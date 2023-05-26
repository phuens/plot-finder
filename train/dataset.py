import torch 
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import random
import pandas as pd 
import os 
from PIL import Image
import math
import numpy as np


def position_embedding(max_length:int, embed_size:int):
    pos_enc = torch.zeros(max_length, embed_size)

    # Compute the position encodings
    positions = torch.arange(0, max_length, dtype=torch.float32).unsqueeze(1)
    divisors = torch.exp(torch.arange(0, embed_size, 2, dtype=torch.float32) * -(math.log(10000.0) / embed_size))
    pos_enc[:, 0::2] = torch.sin(positions * divisors)
    pos_enc[:, 1::2] = torch.cos(positions * divisors)

    # Add the position embeddings to the input embeddings
    # input_embeddings = ... # Your input embeddings tensor
    # pos_embeddings = pos_enc.unsqueeze(0).repeat(input_embeddings.shape[0], 1, 1)
    # output_embeddings = input_embeddings + pos_embeddings

    return pos_enc



class Getdata(torch.utils.data.Dataset):
    def __init__(self, csv_file:str, transform:object, root_img_dir:str) -> None:
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.root_img_dir = root_img_dir
        self.position_embed = position_embedding(max_length=1000, embed_size=2048)

    
    
    def __len__(self): 
        return len(self.data.index)
    

    def __getitem__(self, idx):
        image_name  = self.data.loc[idx][5]
        label       = self.data.loc[idx][6]
        position    = self.data.loc[idx][7]
        path        = self.data.loc[idx][8]
        img_path    = os.path.join(self.root_img_dir, path, image_name)

        image = Image.open(img_path)
        image = image.convert("HSV")
        image = self.transform(image)
        
        score    = self.data.loc[idx][-1]
        # pos_embed = self.position_embed[position]
        # score = 1
        return image, label, score, position, image_name
    

class PrepareDataset: 
    def __init__(self, config:object) -> None:
        self.config = config

    def augment(self): 
        """ augment images based on training vs validation """

        transform = {
            "train": transforms.Compose([
                transforms.Resize((self.config["augmentation"]["height"], self.config["augmentation"]["width"])),
                transforms.RandomHorizontalFlip(p=self.config["augmentation"]["horizontal_flip"]),
                transforms.RandomVerticalFlip(p=self.config["augmentation"]["vertical_flip"]),
                transforms.RandomRotation(random.randint(0, self.config["augmentation"]["rotation"])),
                transforms.GaussianBlur(5, sigma=(self.config["augmentation"]["gaussian"], 2.0)),
                transforms.ColorJitter(brightness= 0.5, hue=0.3),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.ToTensor()
                # transforms.Normalize([0.4426, 0.0586, 0.4954], [0.3439, 0.0737, 0.2783])
            ]),

            "validation": transforms.Compose([
                transforms.Resize((self.config["augmentation"]["height"], self.config["augmentation"]["width"])),
                transforms.ToTensor(),
                transforms.Normalize([0.4426, 0.0586, 0.4954], [0.3439, 0.0737, 0.2783])
            ])
        }

        return transform
    

    def prepare_dataset(self, category='Train'):
            transform = self.augment() 
            
            if category == 'test':  
                test_dataset = Getdata(csv_file=self.config["dataset"]["test_csv"], transform=transform["validation"], root_img_dir=self.config["dataset"]["root_img_dir"])
                test_loader = DataLoader(test_dataset, batch_size=self.config["model"]["batch"], num_workers=4)

                return test_loader

            elif category == 'validation': 
                validation_dataset = Getdata(csv_file=self.config["dataset"]["validation_csv"], transform=transform["validation"], root_img_dir=self.config["dataset"]["root_img_dir"])
            
                # Define the dataloaders with the samplers
                val_loader = DataLoader(validation_dataset, batch_size=self.config["model"]["batch"], num_workers=4)
                
                return val_loader


            else: 
                if self.config["augmentation"]["augment"]: 
                    train_dataset = Getdata(csv_file=self.config["dataset"]["train_csv"], transform=transform["validation"], root_img_dir=self.config["dataset"]["root_img_dir"])
                else: 
                    train_dataset = Getdata(csv_file=self.config["dataset"]["train_csv"], transform=transform["validation"], root_img_dir=self.config["dataset"]["root_img_dir"])
                
                # WEIGHTED SAMPLER
                class_labels = []
                for i in range(len(train_dataset)):
                    _, label, _, _, _=  train_dataset[i]
                    class_labels.append(label)
                class_labels = np.array(class_labels)

                class_sample_count = np.array([len(class_labels)-class_labels.sum(), class_labels.sum()])
                weight = 1. / class_sample_count
                samples_weight = np.array([weight[t] for t in class_labels])
                samples_weight = torch.from_numpy(samples_weight)
                
                weighted_sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
                
                train_loader = DataLoader(train_dataset, sampler = weighted_sampler, batch_size=self.config["model"]["batch"], num_workers=4)
                

                # train_loader = DataLoader(train_dataset, batch_size=self.config["model"]["batch"], num_workers=4)

                return train_loader