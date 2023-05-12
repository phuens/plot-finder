import torch
import os
import json
import yaml
from tqdm import tqdm
import numpy as np 
from torchsummary import summary
from collections import OrderedDict 
import h5py

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
        position, gtscore, target, image_name = [], [], [], []

        # return 
        with torch.no_grad(): 
            for (images, label, score, pos, img_name) in tqdm(self.test_data): 
                images = images.to(self.device)
                label = label.to(self.device)
                _ = self.model(images)

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
        self.calc_metrics(target, predicted)
        return position, gtscore, target, image_name, features


    def calc_metrics(self, targets, predicted):
        predicted = np.concatenate(predicted)
        #targets = np.concatenate(targets)

        accuracy = (predicted == targets).sum() / len(predicted)

        f1 = metrics.f1_score(targets, predicted)
        precision = metrics.precision_score(targets, predicted)
        recall = metrics.recall_score(targets, predicted)
        bal_acc = metrics.balanced_accuracy_score(targets, predicted)
        print(f1, precision, recall, bal_acc)
        return accuracy, f1, precision, recall, bal_acc
            
def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)
    return config

def run(): 
    config  = load_config("config.yml")
    h5_file = h5py.File('hsv_train_emergence_feature.h5', 'w')
    videos  = [] 

    for file in os.listdir('/home/phuntsho/Desktop/plot-finder/raw_data/single_video/score_data/train'):
        if file.endswith('.csv'):
            config['dataset']['validation_csv'] = "/home/phuntsho/Desktop/plot-finder/raw_data/single_video/score_data/train/"+str(file)

            print("\n", config['dataset']['validation_csv'])

            model = Predict(config)
            model.setup_testing()
            position, gtscore, target, image_name, feature = model.predict()

            video_name  = file.split('.')[0] 
            videos.append(str(file))

            h5_file.create_dataset(f'{video_name}/gtscore', data=gtscore)
            h5_file.create_dataset(f'{video_name}/features', data=feature)
            h5_file.create_dataset(f'{video_name}/position', data=position)
            h5_file.create_dataset(f'{video_name}/gttarget', data=target)
            # h5_file.create_dataset(f'{video_name}/img_name', data=image_name)
            

    h5_file.close()

    
    # with open("train_video_names.json", 'w') as f:
    #     json.dump({"train_keys": videos}, f)    

