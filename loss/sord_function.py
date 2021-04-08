import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.data_manager import DataManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sord_function(model, images, metadata):

    wide_gap_loss = True # It could be False as well
    images = images.to(device)  

    label_multiclass = metadata["multiclass_label"].type(torch.LongTensor).to(device, non_blocking=True)
    batch_size = label_multiclass.size(0)
    labels_sord = np.zeros((batch_size, 4))
    for element_idx in range(batch_size):
        current_label = label_multiclass[element_idx].item()
        for class_idx in range(4):
            if wide_gap_loss:
                wide_label = current_label
                wide_class_idx = class_idx

              # Increases the gap between positive and negative
                if wide_label == 0:
                     wide_label = -0.5
                if wide_class_idx == 0:
                    wide_class_idx = -0.5

                labels_sord[element_idx][class_idx] = 2 * abs(wide_label - wide_class_idx) ** 2
            else:
                labels_sord[element_idx][class_idx] = 2 * abs(current_label - class_idx) ** 2

    labels_sord = torch.from_numpy(labels_sord).to(device, non_blocking=True)
    labels_sord = F.softmax(-labels_sord, dim=1)
    
    class_predictions = model(images)
    log_predictions = F.log_softmax(class_predictions, 1)


    # Computes cross entropy
    loss = (-labels_sord * log_predictions).sum(dim=1).mean()

    return loss 
