import torch
import os
import json
import yaml
from tqdm import tqdm
import numpy as np 
from torchsummary import summary
from collections import OrderedDict 
import h5py
import sklearn.metrics as metrics 
import pandas as pd
import random


from train.training import Classification
from train.dataset import PrepareDataset

class Predict(Classification): 
    def __init__(self, config: object) -> None:
        super().__init__(config)
        self.feature        = {} 
        self.global_feature = OrderedDict()
    

    def get_feature(self, name): 
        def feature_hook(model, input, output): 
            self.feature[name] = output.detach()
        return feature_hook


    def setup_testing(self): 
        get_data =  PrepareDataset(self.config)
        self.test_data = get_data.prepare_dataset(category='validation')

        model_path = os.path.join(self.config["test"]["model_path"], self.config["test"]["name"])
        self.model.load_state_dict(torch.load(model_path))


    def append_feature(self, image_name, features): 
        features = features.squeeze().squeeze()
        if len(image_name) > 1: 
            for i in range(len(image_name)):
                self.global_feature[image_name[i]] = features[i].cpu().numpy()
        else: 
            self.global_feature[image_name[0]] = features.cpu().numpy()


    def return_features_as_1D(self): 
        sorted_feature = sorted(self.global_feature.items())
        return [item[1] for item in sorted_feature]


    def predict(self): 
        self.model.to(self.device)
        self.model.eval()
        # print(self.model)
        # print(summary(self.model, (3, 299, 299)))
        self.model.avgpool.register_forward_hook(self.get_feature('avgpool'))
        predicted, position, gtscore, target, image_name = [], [], [], [], []


        with torch.no_grad(): 
            for (images, label, score, pos, img_name) in tqdm(self.test_data): 
                images = images.to(self.device)
                label = label.to(self.device)
                
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                predicted.append(preds.cpu().numpy())
                target.append(label.cpu().numpy())
                image_name.append(img_name)
                gtscore.append(score)
                position.append(pos)
                
                self.append_feature(img_name, self.feature['avgpool'])

        position    = np.concatenate(position).astype(int)
        gtscore     = np.concatenate(gtscore).astype(float)
        target      = np.concatenate(target).astype(int)
        image_name  = np.concatenate(image_name).astype(str)
        features    = np.array(self.return_features_as_1D())
        f1, precision, recall, accuracy, bal_acc = self.calc_metrics(target, predicted)
        
        print(image_name)
        return position, gtscore, target, image_name, features, f1, precision, recall, accuracy, bal_acc


    def calc_metrics(self, targets, predicted):
        predicted = np.concatenate(predicted)
        # targets = np.concatenate(targets)

        accuracy = round(((predicted == targets).sum() / len(predicted)),5)

        f1          = round(metrics.f1_score(targets, predicted), 5)
        precision   = round(metrics.precision_score(targets, predicted), 5)
        recall      = round(metrics.recall_score(targets, predicted), 5)
        bal_acc     = round(metrics.balanced_accuracy_score(targets, predicted), 5)
        print(f"f1:{f1}, precision:{precision}, recall:{recall}, accuracy:{accuracy} bal_acc:{bal_acc}")
        print("-------------------------------------------------------")
        return f1, precision, recall, accuracy, bal_acc

def shuffle_data(gtscore, feature, position, target, name):
    for i in range(len(gtscore) - 1, 0, -1):
        #generate a random number between 0 and index. 
        j = random.randint(0, i)
        # swap the elements at index i and j 
        gtscore[i] , gtscore[j]     = gtscore[j], gtscore[i] 
        feature[i] , feature[j]     = feature[j], feature[i] 
        position[i], position[j]    = position[j], position[i] 
        target[i]  , target[j]      = target[j], target[i]
        name[i]    , name[j]        = name[j], name[i]
    
    return gtscore, feature, position, target, name
    

def run(config): 
    videos  = [] 
    
    metric_score = pd.DataFrame(columns=["video", "f1", "precision", "recall", "bal_acc", "accuracy"])

    directory   = "dataset/score_data/emergence/validation"
    counter     = 0

    model_name = config['test']['name']
    model_name = model_name.split(".")[0]
    print("model name:", model_name)
    category    = directory.split("/")[-1]
    h5_filename = f"{model_name}_{category}_feature.h5"
    h5_file     = h5py.File(h5_filename, 'w')

    shuffle = False

    for file in os.listdir(directory):
        if file.endswith('.csv'):
            config['dataset']['validation_csv'] =  directory+"/"+str(file)

            print(f"\n {config['dataset']['validation_csv']}")

            model = Predict(config)
            model.setup_testing()
            position, gtscore, target, image_name, feature, f1, precision, recall, accuracy, bal_acc = model.predict()

            if counter == 0: 
                print(f"printing out the score just to check: \n{gtscore}")
                counter += 1


            video_name  = file.split('.')[0] 
            videos.append(str(file))

            metric_score = metric_score.append({
                "video"     : video_name,
                "f1"        : f1, 
                "precision" : precision, 
                "recall"    : recall, 
                "bal_acc"   : bal_acc,
                "accuracy"  : accuracy, 
            }, ignore_index = True)

            if shuffle: 
                gtscore, feature, position, target, image_name = shuffle_data(gtscore, feature, position, target, image_name)
            
            df = pd.DataFrame(zip(image_name, gtscore, feature, position, target), columns=["name", "score", "features", "position", "target"])
            df.to_csv("randomized_feature.csv")
            h5_file.create_dataset(f'{video_name}/gtscore',  data=gtscore)
            h5_file.create_dataset(f'{video_name}/features', data=feature)
            h5_file.create_dataset(f'{video_name}/position', data=position)
            h5_file.create_dataset(f'{video_name}/gttarget', data=target)

            # need to encode unicode to ASCII
            asciiList = [n.encode("ascii", "ignore") for n in image_name]

            h5_file.create_dataset(f'{video_name}/img_name', data=asciiList)
    
    h5_file.close()
    print(metric_score)



