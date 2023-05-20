import torch
import os
import pandas as pd
import yaml
from tqdm import tqdm
import numpy as np 

from train.training import Classification
from train.dataset import PrepareDataset

class Predict(Classification): 
    def __init__(self, config: object) -> None:
        super().__init__(config)
        

    def setup_testing(self): 
        get_data =  PrepareDataset(self.config)
        self.test_data = get_data.prepare_dataset(category='validation')

        model_path = os.path.join(self.config["test"]["model_path"], self.config["test"]["name"])
        print(f'Using the model {self.config["test"]["name"]}')
        self.model.load_state_dict(torch.load(model_path))

    def predict(self): 
        self.model.to(self.device)
        self.model.eval()

        probability, predicted, target, image_name = [], [], [], []


        with torch.no_grad(): 
            for (images, label, score, pos, img_name) in tqdm(self.test_data): 
                images = images.to(self.device)
                label = label.to(self.device)
                outputs = self.model(images)

                probability.append(outputs.cpu().numpy())
                
                _, preds = torch.max(outputs, 1)
                predicted.append(preds.cpu().numpy())
                target.append(label.cpu().numpy())
                image_name.append(img_name)


        accuracy, f1, precision, recall, bal_acc = self.calculate_metrics(predicted=predicted, targets=target)
        
        # print(f"f1:{f1}, precision: {precision}, recall: {recall}, bal_acc: {bal_acc} Acc: {accuracy} ")
        print(self.config["dataset"]["validation_csv"])
        print(f"{round(f1, 5)} {round(precision, 5)} {round(recall, 5)} {round(bal_acc, 5)} {round(accuracy, 5)} ")
        print("\n")
        return accuracy, f1, precision, recall, bal_acc
        

        probability = np.concatenate(probability)
        predicted = np.concatenate(predicted)
        target = np.concatenate(target)
        image_name = np.concatenate(image_name)

        # df = pd.DataFrame(list(zip(image_name, probability, predicted, target)), columns=['name', 'probability', 'predicted', 'target'])

        # csv_name = self.config["test"]["name"]
        # csv_name = csv_name.replace('.pt', '.csv')


        # df.to_csv('predict/result/'+csv_name, index=False)


def run(config): 
    
    metric_score = pd.DataFrame(columns=["video", "accuracy", "f1", "precision", "recall", "bal_acc"])
    for files in os.listdir("/home/phuntsho/Desktop/plot-finder/plot-finder/dataset/validation_range_wise"): 
        if files.endswith(".csv"):
            config['dataset']['test_csv'] = config['dataset']['validation_csv'] = str("dataset/validation_range_wise/"+files)
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