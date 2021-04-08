
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import cv2
import argparse
import time

from gradcam import GradCam 
from resnet import load_resnet18
from loss.delu import bwLoss
from data.data_manager import DataManager
from loss.sord_function import sord_function
from eval import evaluate, _compute_scores

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
        self.fc1 = nn.Linear(9216, 1000) # 80: 25600, 70: 18496 40000
        self.fc2 = nn.Linear(1000, 4)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def calculate_measures(groundTs, masks):
    for mask, groundT in zip(masks, groundTs): 
        if(mask.sum() == 0):
            ''' handle the case when the generated explaination is composed of all zeros '''
            precision = 0
            recall = 0
        else:
            precision = np.sum(groundT*mask) / (np.sum(groundT*mask) + np.sum((1-groundT)*mask))
            recall = np.sum(groundT*mask) / (np.sum(groundT*mask) + np.sum(groundT*(1-mask)))

        sumPrec += precision
        sumRec += recall

    return sumPrec, sumRec


def training_model(model, train_loader, bce, optimizer, epoch, num_epochs, n_batches, writer):
    lamb = 0.1
    sumMeanPrec = 0
    sumMeanRec = 0

    model.train()
    for i, (images, metadata) in enumerate(train_loader):
            blackGroundTindx = []
            weightForWhitePixel = torch.tensor([14.0])
            weightForBlackPixel = torch.tensor([1.0])
            masks = grad_cam(images,metadata,training=True)

            model.train()

            tensors_groundTs = []
            for idx, (hospital, expl) in enumerate(zip(metadata["hospital"], metadata["explanation"])):
                npy_groundT = np.load("/home/dataset/segmentation frames/"+hospital+"/"+expl)
                resizedGroundT = cv2.resize(npy_groundT, (14,14), cv2.INTER_AREA) 
                if resizedGroundT.sum() == 0:
                    ''' after resizing, small ground truths may lost all the information.
                    We need to track them, if any'''
                    blackGroundTindx.append(idx)
                    continue
                else:
                    resizedGroundT = np.where(resizedGroundT > 0, 255, 0) # pixels grater than 0 are set to 255
                    normalizedGroundT = resizedGroundT / 255 # normalize expl between 0 and 1
                    tensors_groundTs.append(torch.from_numpy(normalizedGroundT))
            
            groundTs = torch.stack(tensors_groundTs).double()    

            optimizer.zero_grad()
            weightedGroundTs = torch.where(groundTs > 0, weightForWhitePixel, weightForBlackPixel)
            while blackGroundTindx:
                indx = blackGroundTindx.pop()
                masks = torch.cat((masks[:indx,:,:], masks[indx+1:,:,:]))
            batchSize = len(masks)
            lossG = bce(masks, groundTs)
            weightedLoss = lossG * weightedGroundTs
            loss_gradcam = weightedLoss.mean()
            loss_gradcam = loss_gradcam.cuda()
            loss_label = sord_function(model, images, metadata)
            loss = (lamb * loss_label) + ((1 - lamb) * loss_gradcam)
            loss.backward()
            optimizer.step()

            npy_masks = masks.cpu().detach().numpy()
            npy_groundTs = groundTs.cpu().detach().numpy()
            precision, recall = calculate_measures(npy_groundTs, npy_masks)

            sumMeanPrec += prec / batchSize
            sumMeanRec += rec / batchSize
  
            if (i + 1) % 105 == 0:             
                print('Epoch [{}/{}], Step [{}/{}], Total Loss: {:.4f}, EXPLS:[precision: {:.2f}%, recall: {:.2f}%]'
                     .format((epoch + 1), num_epochs, (i + 1), len(train_loader), loss.item(), (sumMeanPrec/n_batches)*100, (sumMeanRec/n_batches)*100))

                # If you use Tensorboard as visualization tool
                writer.add_scalar("Training: Loss", loss.item(), str(epoch + 1))
                writer.add_scalar("Training: Precision", (sumMeanPrec/n_batches)*100, str(epoch + 1))
                writer.add_scalar("Training: Precision", (sumMeanRec/n_batches)*100, str(epoch + 1))
    if (epoch+1) % 10 == 0:           
        torch.save(model.state_dict(), "./checkpoints/SimpleCNN_epoch_"+str(epoch+1)+".pth")  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument('--hospitals', nargs='+', type=str, default=['Germania', 'Pavia', 'Lucca', 'Brescia', 'Gemelli - Roma', 'Tione', 'Trento'], 
        help='Name of the hospital / folder to be used.')
    parser.add_argument('--dataset_root', default='/home/dataset/', type=str, help='Root folder for the datasets.')
    parser.add_argument('--split_file', default='80_20_activeset.csv', type=str, help='File defining train and test splits.')
    parser.add_argument('--standard_image_size', nargs='+', type=int, default=[250, 250])
    parser.add_argument('--input_image_size', nargs='+', type=int, default=[224,224]) 
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
    parser.add_argument("--checkpoint_interval", type=int, default=33, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=3, help="interval evaluations on validation set")
    # Network
    parser.add_argument("--batch_size", default=16, type=int)
    opt = parser.parse_args()

    writer = SummaryWriter("./runs/")
    
    num_epochs = 100
    learning_rate = 0.001

    cuda = torch.cuda.is_available()
    if cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU...')

    model = load_resnet18() # or Net() for Simple CNN
    # Start from checkpoint, if specified
    if opt.pretrained_weights:
        model.load_state_dict(torch.load(opt.pretrained_weights))
        print("pretrained model loaded!")    
    if cuda:
        model = model.cuda()
        print('Loaded model on GPU')

    data_manager = DataManager(opt) 

    train_dataloader = data_manager.get_dataloaders()["train"]
    test_dataloader = data_manager.get_dataloaders()["validation"]

    grad_cam= GradCam(model=model, feature_module=model.layer4, \
                      target_layer_names=["1"], use_cuda=True) # Target layer = last convolutional layer, feature_module = layer where the last conv. layer is

    batch_size = len(train_dataloader)
    bce = nn.BCELoss(reduce=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    
    for epoch in range(num_epochs):
        training_model(model, train_dataloader, bce, cross_entropy, optimizer, epoch, num_epochs, batch_size, writer)
        if (epoch+1) % 10 == 0:
           evaluate(model, grad_cam, calculate_measures, test_dataloader, writer, epoch, batch_size)
