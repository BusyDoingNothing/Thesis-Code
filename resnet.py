import torch
import torchvision

def load_resnet18():
    
    # creates a resnet18 model
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 4) # 4 classes: 0,1,2,3

    return model
