
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import cv2
import argparse

from data.data_manager import DataManager
from resnet import load_resnet18
from loss.sord_function import sord_function
from eval import evaluate

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), # last convolution
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(9216, 1000) #80: 25600, 90: 30976, 70: 18496 
        self.fc2 = nn.Linear(1000, 4)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def training_model(model, train_loader, optimizer, epoch, writer, opt):
    model.train()
    for i, (images, metadata) in enumerate(train_loader):   
            optimizer.zero_grad()
            loss_label = sord_function(model, images, metadata)
            loss_label.backward()
            optimizer.step()
  
            if (i + 1) % 660 == 0:                
                print('Epoch [{}/{}], Step [{}/{}], Total Loss: {:.4f}'
                     .format((epoch + 1), opt.epochs, (i + 1), len(train_loader), loss_label.item()))

                writer.add_scalar("Training: Loss", loss_label.item(), str(epoch + 1)+'_'+str(i+1)) 
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), "./checkpoints/simpleCNN_50"+str(epoch+1)+".pth")  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument('--hospitals', nargs='+', type=str, default=['Germania', 'Pavia', 'Lucca', 'Brescia', 'Gemelli - Roma', 'Tione', 'Trento'], 
        help='Name of the hospital / folder to be used.')
    parser.add_argument('--dataset_root', default='/home/dataset', type=str, help='Root folder for the datasets.')
    parser.add_argument('--split_file', default='split_0.csv', type=str, help='File defining train and test splits.')
    parser.add_argument('--standard_image_size', nargs='+', type=int, default=[250, 250])
    parser.add_argument('--input_image_size', nargs='+', type=int, default=[50,50]) 
    parser.add_argument('--domains_count', type=int, default=2)
    parser.add_argument('--domain_label', type=str, default="sensor_label")
    parser.add_argument('--affine_sigma', type=float, default=0.0)
    parser.add_argument('--rotation', type=float, default=23.0)
    # Environment
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument('--test_size', default=0.3, type=float, help='Relative size of the test set.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--split', default='patient_hash', type=str, help='The split strategy.')
    parser.add_argument('--stratify', default=None, type=str, help='The field to stratify by.')
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    # Network
    parser.add_argument("--batch_size", default=64, type=int)
    opt = parser.parse_args()

    writer = SummaryWriter("./runs/")
    
    learning_rate = 0.001

    cuda = torch.cuda.is_available()
    if cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU...')

    model = Net() 
    # Start from checkpoint, if specified
    if opt.pretrained_weights:
        model.load_state_dict(torch.load(opt.pretrained_weights))
        print("pretrained model loaded!")    
    if cuda:
        model = model.cuda()
        print('Loaded model on GPU')

    data_manager = DataManager(opt) 
    dataset = data_manager.get_datasets()
    train_dataset = dataset["train"]
    test_dataset = dataset["validation"]

    train_dataloader = data_manager.get_dataloaders()["train"]
    test_dataloader = data_manager.get_dataloaders()["validation"]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(opt.epochs):
        training_model(model, train_dataloader, optimizer, epoch, writer, opt)
        if(epoch+1) % 10 == 0:
            evaluate(model, test_dataloader, writer, epoch)
