import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
import random

DATASET = "/home/phn501/plot-finder/dataset/seasons/train_emergence+data.csv"
IMG_DIR = "dataset/images"
batch_size = 100

class Getdata(torch.utils.data.Dataset):
    def __init__(self, csv_file:str, transform:object, root_img_dir:str) -> None:
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.root_img_dir = root_img_dir

    
    
    def __len__(self): 
        return len(self.data.index)
    

    def __getitem__(self, idx):
        image_name  = self.data.loc[idx][5]
        path        = self.data.loc[idx][8]
        img_path    = os.path.join(self.root_img_dir, path, image_name)

        image = Image.open(img_path)
        image = image.convert('HSV')
        image = self.transform(image)

        return image
    

def augment(): 
    """ augment images based on training vs validation """

    return transforms.Compose([
            transforms.Resize((400, 300)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(random.randint(0,360)),
            transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
            transforms.ColorJitter(brightness= 0.5, hue=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ToTensor()
    ])
    

    

def prepare_dataset():
    train_dataset = Getdata(csv_file=DATASET, transform=augment(), root_img_dir=IMG_DIR)
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    return train_loader






def batch_mean_and_sd():
    loader = prepare_dataset()
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
                      cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                            cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
      snd_moment - fst_moment ** 2)        
    return mean,std
  
mean, std = batch_mean_and_sd()
print("mean and std: \n", mean, std)

'''
FOR THE WHOLE DATASET
mean and std: 
 tensor([0.2812, 0.1616, 0.5090]) tensor([0.1981, 0.1516, 0.1076]


FOR TRAIN EMERGENCE+ ONLY: 
[0.4426, 0.0586, 0.4954]) tensor([0.3439, 0.0737, 0.2783]

'''


