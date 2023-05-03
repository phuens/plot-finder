import numpy as np 
import torch 
import yaml
import os
import wandb
from contextlib import nullcontext
import sklearn.metrics as metrics 
import copy
from fastprogress import progress_bar
from torchsummary import summary
from datetime import datetime 
from PIL import Image
import matplotlib.pyplot as plt
import random

from train.model import get_model 
from train.dataset import PrepareDataset


class Classification: 
    def __init__(self, config: object) -> None:
        torch.manual_seed(config["model"]["seed"])
        self.config = config 
        self.model = get_model(self.config["model"]["name"], self.config["model"]["classes"])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print(f"Device: {self.device}")


    def setup_logging(self): 
        os.makedirs(self.config["model"]["save_model"], exist_ok=True)
    

    def count_parameters(self):
        """ count total trainable parameters """

        total_params = 0
        for _, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            total_params+=params
        print(f"Total Trainable Params: {total_params}")
        return total_params

    def save_model(self, model_wts, run): 
        try: 
            path = os.path.join(self.config["model"]["save_model"], f"{run}.pt")
            torch.save(model_wts, path)
        except: 
            print("Save directory not found!")

    def calculate_metrics(self, targets, predicted): 
        predicted = np.concatenate(predicted)
        targets = np.concatenate(targets)
        
        accuracy = (predicted == targets).sum() / len(predicted)

        f1 = metrics.f1_score(targets, predicted)
        precision = metrics.precision_score(targets, predicted)
        recall = metrics.recall_score(targets, predicted)
        bal_acc = metrics.balanced_accuracy_score(targets, predicted)       
        return accuracy, f1, precision, recall, bal_acc

    def setup_optimizer(self):
        optimizer = self.config["model"]["optimizer"]
        
        if optimizer == 'Adam': 
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["model"]["lr"])

        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config["model"]["lr"], momentum=self.config["model"]["momentum"])

        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["model"]["lr"], momentum=self.config["model"]["momentum"])
    
    def setup_loss_fnc(self):
        loss = self.config["model"]["loss"]
        class_weight = self.config["model"]["class_weight"]
        loss_weight = torch.tensor([1.0, class_weight], dtype=torch.float, device='cuda')

        
        if loss == "BCEWithLogitsLoss":
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight = loss_weight[1])
        else: 
            self.criterion = torch.nn.CrossEntropyLoss(weight = loss_weight)
            
    def setup_scheduler(self):
        schedule = self.config["model"]["scheduler"] 
        if schedule == "cycliclr_exp_range": 
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, 
                base_lr = 0.0001, # Initial learning rate which is the lower boundary in the cycle for each parameter group
                max_lr = 1e-3, # Upper learning rate boundaries in the cycle for each parameter group
                step_size_up = 500, # Number of training iterations in the increasing half of a cycle
                cycle_momentum = False,
                mode = "exp_range")

        elif schedule == "cycliclr_triangle":
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, 
                base_lr = 0.0001, # Initial learning rate which is the lower boundary in the cycle for each parameter group
                max_lr = 1e-3, # Upper learning rate boundaries in the cycle for each parameter group
                cycle_momentum = False,
                step_size_up = 500, # Number of training iterations in the increasing half of a cycle
                mode = "triangular")
            
        elif schedule == "cosineannealing": 
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 
                T_0 = 8,# Number of iterations for the first restart
                T_mult = 1, # A factor increases TiTi​ after a restart
                eta_min = 1e-5) # Minimum learning rate
        
        elif schedule == "cosine_onecyclelr": 
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                max_lr = 1e-3, # Upper learning rate boundaries in the cycle for each parameter group
                steps_per_epoch = 1269, # The number of steps per epoch to train for.
                epochs = self.config["model"]["epochs"], # The number of epochs to train for.
                anneal_strategy = 'cos') # Specifies the annealing strategy

        else: 
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                factor=0.1, 
                patience=5, 
                threshold=0.03, 
                threshold_mode='rel', 
                cooldown=0, 
                min_lr=0, eps=1e-08, 
                verbose=True)

    def display_image(self, images, labels, image_name): 
        plt.figure(figsize=(10, 10))
        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            img = images[i]
            img = img.permute(2, 1, 0)
            plt.imshow(img.numpy())
            plt.title([image_name[i], labels[i]], fontsize = 8)
            plt.axis("off")

        plt.show()       
    

    def create_coord(self, batch): 
        width   = self.config["augmentation"]["width"]
        height  = self.config["augmentation"]["height"]


        xx_ones     = torch.ones([batch, width], dtype=torch.int64) # (batch x width)
        xx_ones     = xx_ones.unsqueeze(-1)
        xx_range    = torch.arange(height).unsqueeze(0).repeat(batch, 1) 
        xx_range    = xx_range.unsqueeze(1) # (batch x 1 x height)
        

        xx_channel  = torch.matmul(xx_ones.float(), xx_range.float())
        xx_channel  = xx_channel.unsqueeze(-1)
        
        yy_ones     = torch.ones([batch, height], dtype=torch.int64)
        yy_ones     = yy_ones.unsqueeze(1)

        yy_range    = torch.arange(width).unsqueeze(0).repeat(batch, 1)
        yy_range    = yy_range.unsqueeze(-1)

        yy_channel  = torch.matmul(yy_range.float(), yy_ones.float())
        yy_channel  = yy_channel.unsqueeze(-1)
        
        xx_channel  = xx_channel.float() / (width - 1)
        yy_channel  = yy_channel.float() / (height - 1)
        xx_channel  = xx_channel*2 - 1
        yy_channel  = yy_channel*2 - 1 

        return xx_channel, yy_channel


    def concat_coord(self, images):
        images = images.permute(0, 2, 3, 1)

        xx, yy = self.create_coord(batch=len(images))
        images =  torch.cat([images, xx, yy], dim=-1)
        images = images.permute(0, 3, 1, 2)
        
        return images

    def one_epoch(self, train=True, epoch=0): 
        if train: 
            self.model.train()
            pbar = progress_bar(self.train_loader, leave=False)
            
        else: 
            self.model.eval()
            pbar = progress_bar(self.val_loader, leave=False)

        predicted, targets = [], [] 
        for _, (images, labels, pos_embed, img_name) in enumerate(pbar): 
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()): 
                
                #if not self.displayed: 
                #    self.display_image(images, labels, img_name)
                #    self.displayed = True
                # print(pos_embed.shape)
                # print(images.shape)


                # images = self.concat_coord(images)
                images = images.to(self.device)
                labels = labels.to(self.device)
                pos_embed = pos_embed.unsqueeze(-1).unsqueeze(-1)
                pos_embed = pos_embed.to(self.device)

                outputs = self.model(images)
                if train and self.config["model"]["name"] == "inception_v3": 
                    outputs = outputs[0]

                _, preds = torch.max(outputs, 1)

                predicted.append(preds.cpu().numpy())
                targets.append(labels.cpu().numpy())

                one_hot_targets = torch.nn.functional.one_hot(labels, 2)
                loss = self.criterion(outputs, one_hot_targets.float())
                
                if train:
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    if self.config['wandb']['use']:
                        wandb.log({"learning rate:":self.scheduler.get_last_lr()[0],"epoch":epoch})
                    self.optimizer.zero_grad()

                pbar.comment = f"loss={loss.item():2.3f}"



        accuracy, f1, precision, recall, bal_acc = self.calculate_metrics(predicted=predicted, targets=targets)

        if self.config["wandb"]["use"]: 
            prefix = "train" if train else "validation"

            wandb.log({f"{prefix}_loss": loss.item(), "epoch":epoch})
            wandb.log({f"{prefix}_acc": accuracy, "epoch":epoch})
            wandb.log({f"{prefix}_f1": f1, "epoch":epoch})
            wandb.log({f"{prefix}_precision":precision, "epoch":epoch})
            wandb.log({f"{prefix}_recall":recall, "epoch":epoch})
            wandb.log({f"{prefix}_bal_accuracy":bal_acc, "epoch":epoch})

        model_wts = copy.deepcopy(self.model.state_dict())

        return model_wts, accuracy, f1, precision, recall, bal_acc
    

    def train(self): 
        self.setup_logging()
        self.setup_loss_fnc()
        self.setup_optimizer()
        self.setup_scheduler()
        self.count_parameters()
        
        # GET DATASET
        data = PrepareDataset(self.config)
        self.train_loader = data.prepare_dataset(category = "train")
        self.val_loader = data.prepare_dataset(category = "validation")
        self.displayed = False

        print(f"Train dataset length: {len(self.train_loader.dataset)} \nValidation dataset length: {len(self.val_loader.dataset)}")

        self.model = self.model.to(self.device)
        # print(summary(self.model, (3, 299, 299)))

        # for name, module in self.model.named_children():
        #     print(name)
        #     print(module)

        best_f1 = 0.0

        for epoch in progress_bar(range(self.config["model"]["epochs"]), total=self.config["model"]["epochs"], leave=True):
            # train
            _, epoch_acc, epoch_f1, tprecision, trecall, t_bal_acc  = self.one_epoch(train=True, epoch=epoch)
            
            #validate
            model_wts, accuracy, f1, precision, recall, bal_acc = self.one_epoch(train=False, epoch=epoch)

            print(f"Epoch {epoch}:")
            print(f"training acc: {epoch_acc}, training f1:{epoch_f1}, train recall: {trecall}, training precision: {tprecision}, train_bal_acc: {t_bal_acc}")
            print(f"validation acc: {accuracy} validation f1: {f1},  val preciison: {precision}, val recall:{recall} , val bal_acc: {bal_acc}")

            if f1 > best_f1: 
                run_name = str(self.config["model"]["identifier"])
                self.save_model(model_wts, run_name)

        print(f"Best f1 score: {best_f1}")


def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)
    return config


def run(): 
    config = load_config("config.yml")
    identifier = random.randint(0, 10000000)
    config["model"]["identifier"] = identifier
    classifier = Classification(config)
    

    with wandb.init(project="Plot-finder - detect centered plots", group=config["wandb"]["wandb_group"],name=str(identifier), config=config) if config["wandb"]["use"] else nullcontext():
        classifier.train()


