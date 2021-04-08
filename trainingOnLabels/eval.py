import os

import numpy as np
import cv2
import torch
import time
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

from utils.misc import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_scores(y_true, y_pred):

    folder = "test"

    labels = list(range(4)) # al posto di 4 ci va args.num_classes
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


def evaluate(model, data_loader, writer, step):

    val_loss_meter = AverageMeter('loss1', 'loss2')
    classification_loss_function = torch.nn.CrossEntropyLoss()

    total_samples = 0

    all_predictions = []
    all_labels = []

    total_time = 0

    with torch.no_grad():
        for (img, metadata) in data_loader:

            start = time.time()
            labels_classification = metadata["multiclass_label"].type(torch.LongTensor).to(device)
            total_samples += img.size()[0]

            img = img.to(device)
            
            class_probabilities = model(img)
            class_predictions = torch.argmax(class_probabilities, dim=1).cpu().numpy()
            total_time += time.time() - start

            classification_loss = classification_loss_function(class_probabilities, labels_classification)

            labels_classification = labels_classification.cpu().numpy()

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

    print("- accuracy: {:.3f}".format(scores["test/accuracy"]))
    print("- precision: {:.3f}".format(scores["test/precision"]))
    print("- recall: {:.3f}".format(scores["test/recall"]))
    print("- f1: {:.3f}".format(scores["test/f1"]))
    print("- classification_loss: {:.3f}".format(avg_classification_loss))
    writer.add_scalar("Validation_f1/", scores["test/f1"], step)
    writer.add_scalar("Validation_accuracy/", scores["test/accuracy"], step)
    writer.add_scalar("Validation_precision/", scores["test/precision"], step)
    writer.add_scalar("Validation_classification_loss/", avg_classification_loss, step)        

    return
