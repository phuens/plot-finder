import torch
import os
import pandas as pd
from tqdm import tqdm
import numpy as np 
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch import topk
from train.training import Classification
from train.dataset import PrepareDataset


FOLDER = "1"

class Predict(Classification): 
    def __init__(self, config: object) -> None:
        super().__init__(config)
        self.height = self.config["augmentation"]["height"]
        self.width = self.config["augmentation"]["width"]

    def setup_testing(self): 
        get_data =  PrepareDataset(self.config)
        self.test_data = get_data.prepare_dataset(category='validation')

        model_path = os.path.join(self.config["test"]["model_path"], self.config["test"]["name"])
        self.model.load_state_dict(torch.load(model_path))
        
    # ----------------------------------0
    def display_image(self, images, labels, image_name): 
        plt.figure(figsize=(20, 20))
        for i in range(len(images)):
            ax = plt.subplot(3, 2, i + 1)
            img = images[i]
            img = img.permute(1, 2, 0)
            plt.imshow(img.numpy())
            plt.title([image_name[i], labels[i]], fontsize = 8)
            plt.axis("off")

        plt.show() 

    def predict(self): 
        self.model.to(self.device)
        self.model.eval()

        softmax_0, softmax_1, prob_0, prob_1, predicted, target, image_name = [], [], [], [], [], [], []


        with torch.no_grad(): 
            for (images, label, score, pos, img_name) in tqdm(self.test_data): 

                # self.display_image(images, label, img_name)
                
                images = images.to(self.device)
                label = label.to(self.device)
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1).data.squeeze()
                _, preds = torch.max(outputs, 1)

                probs = probs.cpu().numpy()
                outputs = outputs.cpu().numpy().flatten()

                # prob_0.append(outputs[0])
                # prob_1.append(outputs[1])
                softmax_0.append(probs[0])
                softmax_1.append(probs[1])
                
                # predicted_idx = preds.item()
                predicted.append(preds.cpu().numpy())
                target.append(label.cpu().numpy())
                image_name.append(img_name)

              

        _, f1, precision, recall, _ = self.calculate_metrics(predicted=predicted, targets=target)
        
        predicted = np.concatenate(predicted)
        target = np.concatenate(target)
        image_name = np.concatenate(image_name)

        df = pd.DataFrame(list(zip(image_name, predicted, target, softmax_0, softmax_1)), columns=['name', 'predicted', 'target', 'softmax_0', 'softmax_1'])

        
        csv_name = self.config['test']['filename']
        
        df.to_csv('predict/results/unprocessed-files/'+FOLDER+"/"+FOLDER+"_"+csv_name, index=False)
        return f1, precision, recall


def run(config): 
    print(f"current folder: {FOLDER}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config['model']['batch'] = 1
    model = Predict(config)
    model.setup_testing()
    print(f'Using the model {config["test"]["name"]}')
    metric_score = pd.DataFrame(columns=["video", "f1", "precision", "recall"])
    
    test_directory = "/home/phn501/plot-finder/dataset/shuffled/"+FOLDER


    for files in os.listdir(test_directory): 
        print(files)
        config['dataset']['test_csv']   = config['dataset']['validation_csv'] = os.path.join(test_directory, files)
        config['test']['filename']      = files
        config['model']['batch']        = 1
        
        model = Predict(config)
        config['model']['batch']        = 1
        model.setup_testing()
        f1, precision, recall = model.predict()
        
        metric_score = metric_score.append({ 
            "name"      : files,
            "f1"        : f1, 
            "precision" : precision, 
            "recall"    : recall
        }, ignore_index = True)
    

    print("\n\nMEAN SCORES")
    print(f"F1:     {metric_score['f1'].mean()}")
    print(f"PREC:   {metric_score['precision'].mean()}")
    print(f"RECALL: {metric_score['recall'].mean()}")
