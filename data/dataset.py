from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import os
import torch
import hashlib
import pandas as pd
import numpy as np
from scipy.io import loadmat
from torch.utils.data.dataset import Subset


class COVID19Dataset(Dataset):
    # As soon as the object class is created, it filters the dataset per hospital (given as input).
    def __init__(self, args, transforms=None):
        self.args = args
        self.dataset_root = args.dataset_root
        self.transforms = transforms

        # load metadata
        self.metadata = pd.read_pickle(os.path.join(self.dataset_root, 'activeset.pkl')) # label, fram_pos, sensor, raw_filenames, video_hash, patient, patient hash, hospital, filnename

        # filter desired hospitals
        metadata_mask = self.metadata.hospital.str.contains('|'.join(args.hospitals))
        self.metadata = self.metadata[metadata_mask] # metadata updated with filtered hospitals

        # get labels
        self.labels = self.metadata.label # filtered hospotals' labels

    def __len__(self):
        return len(self.metadata.index)

    def __repr__(self):
        return repr(self.metadata)

    def __getitem__(self, idx):
        frame_file = self.metadata.loc[idx].filename
        frame_path = os.path.join(self.dataset_root, 'frames', frame_file)
        frame = np.load(frame_path)
        if self.transforms:
            frame = self.transforms(frame)
        return frame, self.metadata  # self.labels[idx][0]

    def get_video_dataset(self, video_hash, transforms):

        mask = self.metadata["video_hash"] == video_hash # True or False for each row
        indices = self.metadata[mask].index.values.tolist() # indices of selected video_hash

        return TransformableFullMetadataSubset(self, indices, transforms)

    def get_video_info(self):

        all_hash_values = self.metadata["video_hash"].unique()
        all_video_info = []
        for current_hash in all_hash_values:
            current_record = dict(self.metadata[self.metadata["video_hash"] == current_hash].iloc[0])
            all_video_info.append(current_record)
        return all_video_info

class TransformableSubset(Subset):

    def __init__(self, dataset, indices, transforms=None):
        super(TransformableSubset, self).__init__(dataset, indices)
        self.transforms = transforms

    def __getitem__(self, idx):
        translated_idx = self.indices[idx]
        frame, metadata = self.dataset[translated_idx]
        if self.transforms:
            frame = self.transforms(frame)

        return frame, self.dataset.metadata.label[translated_idx][0]


class TransformableFullMetadataSubset(Subset):

    def __init__(self, dataset, indices, transforms=None):
        super(TransformableFullMetadataSubset, self).__init__(dataset, indices)
        self.transforms = transforms

    def find_last_one_idx(self, sequence):
        index = len(sequence) - 1
        while sequence[index] != 1:
            index -= 1
            assert(index >= 0)
        return index

    def compute_labels(self, full_label_info):
        severity_levels_count = self.dataset.args.num_classes

        # If there are extra annotations discard them
        if len(full_label_info) > severity_levels_count:
            full_label_info = full_label_info[:severity_levels_count]

        severity_levels = np.zeros(severity_levels_count, dtype=np.int)
        for index, value in enumerate(full_label_info):
            severity_levels[index] = value

        # In case of missing annotations where the highest known severity level is present we must flag the others as unknowns
        if full_label_info[-1] == 1 and len(full_label_info) < severity_levels_count:
            for index in range(len(full_label_info), severity_levels_count):
                severity_levels[index] = -1

        classification_label = full_label_info[0]  # 0 if negative, 1 if positive

        return classification_label, severity_levels

    def get_info(self):
        info = {}
        unique_patients = self.dataset.metadata.loc[self.indices][["hospital", "patient"]].drop_duplicates()
        info["patients"] = unique_patients

        levels = [0] * self.dataset.args.num_classes
        annotation_level = [0] * self.dataset.args.num_classes * 2
        for index in self.indices:
            current_metadata = self.dataset.metadata.loc[index]
            current_label = current_metadata['label']
            classification_label, severity_labels = self.compute_labels(current_label)

            for index, value in enumerate(severity_labels):
                if value != -1:
                    annotation_level[index] += 1
                if value == 1:
                    levels[index] += 1

        info["negatives_count"] = annotation_level[0] - levels[0]
        info["positives_count"] = levels[0]
        info["all_counts"] = levels
        info["annotation_counts"] = annotation_level

        return info

    def sensor_to_domain(self, sensor):
        assert (sensor in ["linear", "unknown", "convex"])

        sensor_label = 0
        # Convex is mapped to 1. All the others (linear and unknown) are mapped to 0
        if sensor == "convex":
            sensor_label = 1

        return sensor_label

    def __getitem__(self, idx):
        translated_idx = self.indices[idx]
        frame, metadata = self.dataset[translated_idx]
        if self.transforms:
            frame = self.transforms(frame)

        full_label_info = self.dataset.metadata.label[translated_idx]
        #classification_label, severity_label = self.compute_labels(full_label_info)
        multiclass_label = sum(full_label_info)

        sensor = self.dataset.metadata.sensor[translated_idx]
        sensor_label = self.sensor_to_domain(sensor)

        return frame, {"multiclass_label": multiclass_label,
                       "hospital": self.dataset.metadata.hospital[translated_idx],
                       "sensor": sensor,
                       "sensor_label": sensor_label,
                       "filenames": self.dataset.metadata.raw_filenames[translated_idx],
                       "frame_pos": self.dataset.metadata.frame_pos[translated_idx],
                       "patient": self.dataset.metadata.patient_hash[translated_idx],
                       "patient_name": self.dataset.metadata.patient[translated_idx],
                       "explanation": self.dataset.metadata.explanation[translated_idx]}

def compute_uniform_sampling_weights(metadata, indices):
    '''
    Computes the sampling probability for each element in metadata elencated in indices such that, sampling
    according to that probability yields samples with uniform probability in their label
    :param metadata:
    :param indices: list of indices
    :return:
    '''
    classes_counter = [0, 0]
    for current_index in indices:
        current_sample = metadata.loc[current_index]
        classes_counter[current_sample['label'][0]] += 1
    total_samples = len(indices)

    class_weights = [total_samples, total_samples]
    for i in range(len(class_weights)):
        if classes_counter[i] != 0:
            class_weights[i] /= classes_counter[i]
        else:
            class_weights[i] = 0

    weights = []
    for current_index in indices:
        current_sample = metadata.loc[current_index]
        current_class = current_sample['label'][0]
        weights.append(class_weights[current_class])

    return weights


def covid_train_test_split(args, covid19_dataset, train_transforms, test_transforms):
    # split patients
    metadata = covid19_dataset.metadata

    if args.split_file:
        split_file_path = os.path.join(args.dataset_root, "splits", args.split_file)
        split_metadata = pd.read_csv(split_file_path) 
        train_hashes = split_metadata[split_metadata["split"] == "train"]["patient_hash"].tolist() # Crea lista di pazienti_hash per training
        validation_hashes = split_metadata[split_metadata["split"] == "test"]["patient_hash"].tolist() # Crea lista di pazienti_hash per testing

        train_mask = metadata['patient_hash'].str.contains('|'.join(train_hashes))
        test_mask = metadata['patient_hash'].str.contains('|'.join(validation_hashes))
        train_indices = metadata[train_mask].index.values.tolist()
        test_indices = metadata[test_mask].index.values.tolist()
    else:
        splitter = metadata[args.split].unique()
        stratifier = [metadata[metadata[args.split] == s][args.stratify].iloc[0] for s in
                      splitter] if args.stratify else None
        train_split, test_split = train_test_split(
            splitter,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=stratifier)

        # get frame indices according to patient split
        train_mask = metadata[args.split].str.contains('|'.join(train_split))
        train_indices = metadata[train_mask].index.values.tolist()
        test_mask = metadata[args.split].str.contains('|'.join(test_split))
        test_indices = metadata[test_mask].index.values.tolist()

    train_weights = compute_uniform_sampling_weights(metadata, train_indices)

    # subset the dataset
    train_subset = TransformableFullMetadataSubset(covid19_dataset, train_indices, train_transforms) # train_indeces: frame per training
    test_subset = TransformableFullMetadataSubset(covid19_dataset, test_indices, test_transforms)

    return train_subset, test_subset, train_weights
