import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from data.dataset import COVID19Dataset, covid_train_test_split, TransformableFullMetadataSubset
from data.transformations import ToRGB, NumpyToPIL, Affine, Show, TorchToPIL


class DataManager:
    '''
    Helper class for data loading and preparation
    '''

    def __init__(self, args):

        self.args = args

        self.training_transform = self._get_training_transformations()
        self.validation_transform = self._get_validation_transformations()

        self.covid19_dataset = COVID19Dataset(args, transforms=None)
        self.train_dataset, self.validation_dataset, train_weights = covid_train_test_split(args, 
            self.covid19_dataset, 
            self.training_transform, 
            self.validation_transform)
        self.viz_train_dataset, self.viz_validation_dataset, _ = covid_train_test_split(args, 
            self.covid19_dataset, 
            self.validation_transform, 
            self.validation_transform,
        )

        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=train_weights, 
            num_samples=len(train_weights))

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=args.batch_size, 
            sampler=sampler, 
            num_workers=args.num_workers, 
            pin_memory=True
        )

        self.validation_dataloader = torch.utils.data.DataLoader(
            self.validation_dataset, 
            shuffle=False, 
            batch_size=args.batch_size,
            num_workers=args.num_workers, 
            pin_memory=True
        )

        self.viz_train_dataloader = torch.utils.data.DataLoader(
            self.viz_train_dataset, 
            shuffle=False,
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            pin_memory=True,
        )
        
        self.viz_validation_dataloader = torch.utils.data.DataLoader(
            self.viz_validation_dataset, 
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    
    def get_video_dataset(self, video_hash):
        return self.covid19_dataset.get_video_dataset(
            video_hash, 
            self._get_validation_transformations(),
        )

    def get_video_dataloader(self, video_hash):
            video_dataset = self.get_video_dataset(video_hash)
            video_dataloader = torch.utils.data.DataLoader(
                video_dataset, 
                shuffle=False, 
                batch_size=64, 
                num_workers=self.args.num_workers, 
                pin_memory=True,
            )

            return video_dataloader    

    def get_video_info(self):
        return self.covid19_dataset.get_video_info()                

    def get_datasets(self):
        return {
            "train": self.train_dataset,
            "validation": self.validation_dataset
        }

    def get_dataloaders(self):
        return {
            "train": self.train_dataloader,
            "validation": self.validation_dataloader
        }
    def get_dataloaders(self):
        return {
            "train": self.train_dataloader,
            "validation": self.validation_dataloader,
            "train_viz": self.viz_train_dataloader,
            "validation_viz": self.viz_validation_dataloader
        }

    def _get_training_transformations(self):
        '''
        Computes the transformation to apply during training
        :param args: main script parameters
        :return:
        '''
        transforms_list = [
            NumpyToPIL(),
            transforms.Resize(self.args.standard_image_size),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([transforms.RandomCrop(self.args.input_image_size)]),
            transforms.Resize(self.args.input_image_size),
            transforms.ToTensor(),
            Affine(self.args.affine_sigma),
            transforms.Normalize(mean=[0.15, 0.15, 0.15], std=[0.15, 0.15, 0.15])
        ]
        final_transform = transforms.Compose(transforms_list)
        return final_transform

    def _get_validation_transformations(self):
        '''
        Computes the transformation to apply during validation
        :param args: main script parameters
        :return:
        '''
        transforms_list = [
            NumpyToPIL(),
            transforms.Resize(self.args.input_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.15, 0.15, 0.15], std=[0.15, 0.15, 0.15])]
        final_transform = transforms.Compose(transforms_list)
        return final_transform
