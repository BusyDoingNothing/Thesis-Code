import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]

        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        count_layer = 0
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            #    count_layer += 1                   ## Uncomment to use Simple CNN. Comment to use ResNet18
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
                #count_layer += 1                   ## Uncomment to use Simple CNN. Comment to use ResNet18
            #elif count_layer == 2:                 ## Uncomment to use Simple CNN. Comment to use ResNet18
            #    x = x.reshape(x.size(0), -1)       ## Uncomment to use Simple CNN. Comment to use ResNet18
            #    x = module(x)                      ## Uncomment to use Simple CNN. Comment to use ResNet18
            #    count_layer += 1                   ## Uncomment to use Simple CNN. Comment to use ResNet18
            else: 
                x = module(x)
            #    count_layer += 1                   ## Uncomment to use Simple CNN. Comment to use ResNet18

        return target_activations, x

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.sigmoid = nn.Sigmoid()
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, metadata, training, index=None):

        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=1)

        one_hot = np.zeros((output.shape[0], output.size()[-1]), dtype=np.float32)
        for i, elm in enumerate(index):
            one_hot[i][elm] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        if training == True:
            grads_val = self.extractor.get_gradients()[-1].cpu()

            targets = features[-1].cpu()

            weights = torch.mean(grads_val, axis=(2, 3), keepdims=True)
            cam = torch.zeros(targets.shape[2:], dtype=torch.float32)
            camList = []

            for batch, pesi in enumerate(weights):
                for i, w in enumerate(pesi):
                    cam += w * targets[batch, i, :, :]
                camList += [cam]
                if batch < targets.shape[0]:
                    cam = torch.zeros(targets.shape[2:], dtype=torch.float32)

            tmpCamList = []
            for i, cam in enumerate(camList):
                    cam = torch.where(cam > 0, cam*4.38, cam*1.095)
                    if torch.max(cam) == 0:
                        tmpCamList.append(cam)
                    else:
                        cam = self.sigmoid(cam)
                        tmpCamList.append(cam)
                    
            masks = torch.stack(tmpCamList).double()
            return masks