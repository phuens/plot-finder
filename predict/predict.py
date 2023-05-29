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
import random

class Predict(Classification): 
    def __init__(self, config: object) -> None:
        super().__init__(config)
        self.height = self.config["augmentation"]["height"]
        self.width = self.config["augmentation"]["width"]

    def setup_testing(self): 
        get_data =  PrepareDataset(self.config)
        self.test_data = get_data.prepare_dataset(category='validation')

        model_path = os.path.join(self.config["test"]["model_path"], self.config["test"]["name"])
        print(f'Using the model {self.config["test"]["name"]}')
        self.model.load_state_dict(torch.load(model_path))
        
        self.features_blobs = []

        self.model._modules.get('Mixed_7c').register_forward_hook(self.hook_feature)

    # -------------- CAM ---------------
    def hook_feature(self, module, input, output):
        self.features_blobs.append(output.data.cpu().numpy())

    def returnCAM(self, feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (self.width, self.height)

        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            f_sghape = feature_conv.reshape((nc, h*w))
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam
    
    def show_cam(self, CAMs, orig_image, class_idx, save_name, target):
        for i, cam in enumerate(CAMs):
            
            heatmap = cv2.applyColorMap(cv2.resize(cam,(self.width, self.height)), cv2.COLORMAP_JET)

            orig_image = orig_image.cpu().detach().numpy()
            orig_image = orig_image.transpose(2, 3, 1, 0).squeeze(axis=3)
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_HSV2RGB)

            orig_image = cv2.normalize(orig_image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            orig_image = orig_image.astype(np.uint8)
            path = f"/home/phuntsho/Desktop/plot-finder/plot-finder/predict/result/orig/{save_name}"

            cv2.imwrite(path,orig_image)
           
            heatmap = np.array(heatmap, dtype="float32")
            result = heatmap * 0.3 + orig_image * 0.9

            # put class label text on the result
            label = "centered" if class_idx[i] == 1 else "reject" 
            cv2.putText(result, label, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            name, ext = save_name.split(".")
            if class_idx[i] == target: 
                name += "_C"
            else: 
                name += "_W"
            
            target = target.cpu().detach().numpy()
            name, ext = save_name.split(".")
            if target[0] == 1: name += "_centered"

            if class_idx[i] != target[0]: 
                name += "_wrong"
                

            save_name = name+"."+ext.lower()
            path = f"/home/phuntsho/Desktop/plot-finder/plot-finder/predict/result/CAM/{save_name}"
            cv2.imwrite(path,result)


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

        params = list(self.model.parameters())

        weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
        counter = 0 
        with torch.no_grad(): 
            for (images, label, score, pos, img_name) in tqdm(self.test_data): 

                # self.display_image(images, label, img_name)
                
                images = images.to(self.device)
                label = label.to(self.device)
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1).data.squeeze()
                
                prob_0.append(outputs[0])
                prob_1.append(outputs[1])
                softmax_0.append(probs[0])
                softmax_1.append(probs[1])
                
                _, preds = torch.max(outputs, 1)
                # predicted_idx = preds.item()
                predicted.append(preds.cpu().numpy())
                target.append(label.cpu().numpy())
                image_name.append(img_name)

                # ******  GENERATE CAM ******
                if 1 == 2: 
                    class_idx = topk(probs, 1)[1].int()
                    # generate class activation mapping for the top1 prediction
                    CAMs = self.returnCAM(self.features_blobs[counter], weight_softmax, class_idx)
                    counter += 1
                    self.show_cam(CAMs, images, class_idx, str(img_name[0]), label)
                # ***********
                

        accuracy, f1, precision, recall, bal_acc = self.calculate_metrics(predicted=predicted, targets=target)
        
        # print(f"f1:{f1}, precision: {precision}, recall: {recall}, bal_acc: {bal_acc} Acc: {accuracy} ")
        print(self.config["dataset"]["validation_csv"])
        print(f"f1: {round(f1, 5)},  precision: {round(precision, 5)}, recall: {round(recall, 5)}, {round(bal_acc, 5)}, {round(accuracy, 5)} ")
        print("\n")
        
        predicted = np.concatenate(predicted)
        target = np.concatenate(target)
        image_name = np.concatenate(image_name)

        df = pd.DataFrame(list(zip(image_name, predicted, target, prob_0, prob_1, softmax_0, softmax_1)), columns=['name', 'predicted', 'target', 'prob_0', 'prob_1', 'softmax_0', 'softmax_1'])

        csv_name = 'model-'+self.config["test"]["name"]
        csv_name = csv_name.replace('.pt', '___file-')
        csv_name += self.config['test']['filename']


        df.to_csv('/home/phn501/plot-finder/predict/result/csv/'+csv_name, index=False)


        return accuracy, f1, precision, recall, bal_acc


def run(config): 
    config['dataset']['test_csv']   = config['dataset']['validation_csv'] = str("/home/phuntsho/Desktop/plot-finder/plot-finder/dataset/validation_range_wise/NUE_1_2019-06-27_range_8.csv")
    config['model']['batch'] = 1
    model = Predict(config)
    model.setup_testing()
    accuracy, f1, precision, recall , bal_acc = model.predict()
            
   
    metric_score = pd.DataFrame(columns=["video", "accuracy", "f1", "precision", "recall", "bal_acc"])
    for files in os.listdir("/home/phn501/plot-finder/dataset/validation_range_wise/emergence"): 
        if files.endswith(".csv"):
            config['dataset']['test_csv']   = config['dataset']['validation_csv'] = str("dataset/validation_range_wise/emergence/"+files)
            config['test']['filename']      = files
            config['model']['batch']        = 1
            
            model = Predict(config)
            model.setup_testing()
            accuracy, f1, precision, recall , bal_acc = model.predict()
            
            metric_score = metric_score.append({
                "accuracy"  : accuracy, 
                "f1"        : f1, 
                "precision" : precision, 
                "recall"    : recall, 
                "bal_acc"   : bal_acc
            }, ignore_index = True)
    
    print("\n\nMEAN SCORES")
    print(f"ACC:    {metric_score['accuracy'].mean()}")
    print(f"F1:     {metric_score['f1'].mean()}")
    print(f"PREC:   {metric_score['precision'].mean()}")
    print(f"RECALL: {metric_score['recall'].mean()}")
    print(f"BAL_ACC:{metric_score['bal_acc'].mean()}")