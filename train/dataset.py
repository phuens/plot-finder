import torch 
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import random
import pandas as pd 
import os 
from PIL import Image


class Getdata(torch.utils.data.Dataset):
    def __init__(self, csv_file:str, transform:object, root_img_dir:str) -> None:
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.root_img_dir = root_img_dir
    
    
    def __len__(self): 
        return len(self.data.index)
    

    def __getitem__(self, idx):

        path    = self.data.loc[idx][-1]
        image_name  = self.data.loc[idx][5]
        img_path    = os.path.join(self.root_img_dir, path, image_name)

        image = Image.open(img_path)
        image = self.transform(image)
        label = self.data.loc[idx][6]
        position    = self.data.loc[idx][7]
        img_name = self.data.loc[idx][5] 

        return image, label, position, img_name



class PrepareDataset: 
    def __init__(self, config:object) -> None:
        self.config = config


    def augment(self): 
        """ augment images based on training vs validation """

        transform = {
            "training": transforms.Compose([
                transforms.Resize((self.config["augmentation"]["width"], self.config["augmentation"]["height"])),
                transforms.RandomHorizontalFlip(p=self.config["augmentation"]["horizontal_flip"]),
                transforms.RandomVerticalFlip(p=self.config["augmentation"]["vertical_flip"]),
                transforms.RandomRotation(random.randint(0, self.config["augmentation"]["rotation"])),
                transforms.GaussianBlur(5, sigma=(self.config["augmentation"]["gaussian"], 2.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.47, 0.47, 0.47], [0.3,0.3,0.3])
            ]),

            "validation": transforms.Compose([
                transforms.Resize((self.config["augmentation"]["width"], self.config["augmentation"]["height"])), 
                transforms.ToTensor(),
                # transforms.Normalize([0.4, 0.4, 0.4], [0.3,0.3,0.3])
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
                train_dataset = Getdata(csv_file=self.config["dataset"]["train_csv"], transform=transform["validation"], root_img_dir=self.config["dataset"]["root_img_dir"])
                
                # train_size = int(self.config["model"]["train_size"] * len(train_dataset))
                # val_size = int(len(train_dataset) - train_size)
                # train_data, val_data = random_split(train_dataset, [train_size, val_size])
                
                # Define the dataloaders with the samplers
                train_loader = DataLoader(train_dataset, batch_size=self.config["model"]["batch"], num_workers=4)
                
                return train_loader


               
