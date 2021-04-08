import os

import numpy as np
import torch
import cv2
import time
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

from utils.misc import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUMB_OF_CLASSES = 4


def _compute_scores(y_true, y_pred):

    folder = "test"

    labels = list(range(NUMB_OF_CLASSES))
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)

    print(confusion)

    scores = {}
    scores["{}/accuracy".format(folder)] = accuracy
    scores["{}/precision".format(folder)] = precision
    scores["{}/recall".format(folder)] = recall
    scores["{}/f1".format(folder)] = fscore

    precision, recall, fscore, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, labels=labels, average=None)
    for i in range(len(labels)):
        prefix = "{}_{}/".format(folder, i)
        scores[prefix + "precision"] = precision[i]
        scores[prefix + "recall"] = recall[i]
        scores[prefix + "f1"] = fscore[i]

    return scores


def evaluate(model, grad_cam, calculate_measures, data_loader, writer, step, n_batches):

    val_loss_meter = AverageMeter('loss1', 'loss2')
    classification_loss_function = torch.nn.CrossEntropyLoss()

    total_samples = 0

    all_predictions = []
    all_labels = []
    
    sumMeanPrec = 0
    sumMeanRec = 0

    #with torch.no_grad():                  ## Comment this line if you want to use GradCAM
    for (img, metadata) in data_loader:
            blackGroundTindx = []
            start = time.time()
            labels_classification = metadata["multiclass_label"].type(torch.LongTensor).to(device)
            total_samples += img.size()[0]

            img = img.to(device)

            masks = grad_cam(img,metadata,training=True)
            class_probabilities = model(img)
            class_predictions = torch.argmax(class_probabilities, dim=1).cpu().numpy()
            total_time += time.time() - start

            classification_loss = classification_loss_function(class_probabilities, labels_classification)

            npyGroundTList = []
            for idx, (hospital, expl, label) in enumerate(zip(metadata["hospital"], metadata["explanation"], metadata["multiclass_label"])):
                npy_groundT = np.load("/home/dataset/segmentation frames/"+hospital+"/"+expl)
                resizedGroundT = cv2.resize(npy_groundT, (14,14), cv2.INTER_AREA) 
                if resizedGroundT.sum() == 0:
                    blackGroundTindx.append(idx)
                    continue
                else:
                    resizedGroundT = np.where(resizedGroundT > 0, 255, 0)
                    normGroundT = resizedGroundT / 255
                    npyGroundTList.append(normGroundT)

            labels_classification = labels_classification.cpu().numpy()
            while blackGroundTindx:
                idx = blackGroundTindx.pop()
                masks = torch.cat((masks[:idx,:,:], masks[idx+1:,:,:]))
            batchSize = len(masks)
            npy_masks = masks.cpu().detach().numpy()
            precision, recall = calculate_measures(npyGroundTList, npy_masks) 

            sumMeanPrec += precision / batchSize
            sumMeanRec += rececall / batchSize

            val_loss_meter.add({'classification_loss': classification_loss.item()})
            all_labels.append(labels_classification)
            all_predictions.append(class_predictions)

    inference_time = total_time / total_samples
    print("Inference time: {}".format(inference_time))

    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)

    # Computes and logs classification results
    scores = _compute_scores(all_labels, all_predictions)

    avg_classification_loss = val_loss_meter.pop('classification_loss')

    print("- EXPL: precision {:.2f}%, recall: {:.2f}%".format((sumMeanPrec/n_batches)*100, (sumMeanRec/n_batches)*100))    
    print("- accuracy: {:.3f}".format(scores["test/accuracy"]))
    print("- precision: {:.3f}".format(scores["test/precision"]))
    print("- recall: {:.3f}".format(scores["test/recall"]))
    print("- f1: {:.3f}".format(scores["test/f1"]))
    print("- classification_loss: {:.3f}".format(avg_classification_loss))

    # If you use Tensorboard as visualization tool
    writer.add_scalar("Validation_f1/", scores["test/f1"], step)
    writer.add_scalar("Validation_accuracy/", scores["test/accuracy"], step)
    writer.add_scalar("Validation_precision/", scores["test/precision"], step)
    writer.add_scalar("Validation_classification_loss/", avg_classification_loss, step) 
    writer.add_scalar("EXPL: precision", (sumMeanPrec/n_batches)*100, step) 
    writer.add_scalar("EXPL: recall", (sumMeanRec/n_batches)*100, step)         
