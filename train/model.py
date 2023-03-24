import torch
from torchvision import models


def get_model(model, classes, pretrained=True):
    model_ft = None
    if model == 'resnet18':
        model_ft = models.resnet18(pretrained=pretrained)
    
    elif model == 'resnet50':
        model_ft = models.resnet50(pretrained=pretrained)

    elif model == "inception_v3": 
        model_ft = models.inception_v3(pretrained=True)

    elif model == "swin_transfomer": 
        model_ft = models.swin_b(weights="DEFAULT")
    
    elif model == "convnext": 
        model_ft = models.convnext_base(weights='DEFAULT')


    model_ft = torch.nn.Sequential(torch.nn.Linear(model_ft.fc.in_features, classes))


    return model_ft
