import torch
from torchvision import models


def get_model(model, classes, pretrained=True):
    model_ft = None
    if model == 'resnet18':
        model_ft = models.resnet18(pretrained=pretrained)
        model_ft.fc = torch.nn.Sequential(torch.nn.Linear(model_ft.fc.in_features, classes))
    
    elif model == 'resnet50':
        model_ft = models.resnet50(pretrained=False)
        model_ft.fc = torch.nn.Sequential(torch.nn.Linear(model_ft.fc.in_features, classes))

    elif model == "inception_v3": 
        model_ft = models.inception_v3(pretrained=True)
        model_ft.fc = torch.nn.Sequential(torch.nn.Linear(model_ft.fc.in_features, classes))
    
    elif model == "convnext": 
        model_ft = models.convnext_base(weights='DEFAULT')
        model_ft.classifier = torch.nn.Sequential(
            # torch.nn.LayerNorm((16, 1024, 1, 1), eps=1e-06, elementwise_affine=True),
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(in_features=1024, out_features=classes, bias=True)
        )
    
    elif model == "vit": 
        model_ft    = models.vit_b_32()
        in_feature  = model_ft.heads.head.in_features
        model_ft.heads = torch.nn.Sequential(
            torch.nn.Linear(in_feature, classes)
        )


    return model_ft
