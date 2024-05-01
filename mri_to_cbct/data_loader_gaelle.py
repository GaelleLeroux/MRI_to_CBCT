from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import SimpleITK as sitk
from PIL import Image
# import nrrd
import os
import sys
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import monai
from monai.transforms import (  
    LoadImaged,
    Compose,
    Resize,
    RandZoomd,
    RandRotated,
    RandAffined,
    ToTensord
)   
import math

import pytorch_lightning as pl
    
class LotusTrainTransforms2:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = Compose(
            [   
            LoadImaged(keys=['img_mri', 'img_cbct']),
            RandZoomd(keys=['img_mri', 'img_cbct'], min_zoom=0.8, max_zoom=1.1, mode=['area', 'nearest'], prob=0.9, padding_mode='constant'),
            RandRotated(keys=['img_mri', 'img_cbct'], range_x=math.pi, mode=['bilinear', 'nearest'], prob=1.0),
            RandAffined(keys=['img_mri', 'img_cbct'], prob=0.8, shear_range=(0.1, 0.1), mode=['bilinear', 'nearest'], padding_mode='zeros'),
            ToTensord(keys=['img_mri', 'img_cbct'])
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp) 

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets, use_max=True):
        self.datasets = datasets
        self.use_max = use_max

    def __getitem__(self, i):
        # print(f"Fetching index: {i}") #debug
        # for d in self.datasets:
            # print(f"Dataset length: {len(d)}") #debug
        return tuple(self.check_len(d, i) for d in self.datasets)

    def __len__(self):
        if self.use_max:
            return max(len(d) for d in self.datasets)
        return min(len(d) for d in self.datasets)

    def shuffle(self):
        for d in self.datasets:
            if isinstance(d, monai.data.Dataset):                
                d.data.df = d.data.df.sample(frac=1.0).reset_index(drop=True)                
            else:
                d.df = d.df.sample(frac=1.0).reset_index(drop=True)

    def check_len(self, d, i):
        if i < len(d):
            return d[i]
        else:
            j = i % len(d)
            return d[j]
    
    
    
class ConcatDataModule(pl.LightningDataModule):
    def __init__(self, datasetA_train, datasetA_val, datasetB_train, datasetB_val, batch_size=8, num_workers=4):
        super().__init__()

        self.datasetA_train = datasetA_train
        self.datasetB_train = datasetB_train

        self.datasetA_val = datasetA_val
        self.datasetB_val = datasetB_val
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders        
        self.train_ds = ConcatDataset(self.datasetA_train, self.datasetB_train)
        self.val_ds = ConcatDataset(self.datasetA_val, self.datasetB_val)

    def train_dataloader(self):
        self.train_ds.shuffle()
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)    
    
    
class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Generic dataset for images located in a specified directory.
        Args:
            directory (string): Path to the folder containing images.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform
        self.images = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        # image augmentation functions
        self.eval_transform = Compose(
            [   
                LoadImaged(keys=['img_mri', 'img_cbct']),
                RandZoomd(keys=['img_mri', 'img_cbct'], min_zoom=0.8, max_zoom=1.1, mode=['area', 'nearest'], prob=0.9, padding_mode='constant'),
                RandRotated(keys=['img_mri', 'img_cbct'], range_x=math.pi, mode=['bilinear', 'nearest'], prob=1.0),
                RandAffined(keys=['img_mri', 'img_cbct'], prob=0.8, shear_range=(0.1, 0.1), mode=['bilinear', 'nearest'], padding_mode='zeros'),
                ToTensord(keys=['img_mri', 'img_cbct'])
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # img_path = self.images[idx]
        # # image = Image.open(img_path).convert('L')
        # image = sitk.ReadImage(img_path)
        # image = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        # if self.transform:
        #     image = self.transform(image)
        # return image
        return self.eval_transform(idx)  
    
class LotusTrainTransforms:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = Compose(
            [   
            # EnsureChannelFirstd(keys=['img', 'seg'], channel_dim=-1),
            LoadImaged(keys=['img_mri', 'img_cbct']),
            RandZoomd(keys=['img_mri', 'img_cbct'], min_zoom=0.8, max_zoom=1.1, mode=['area', 'nearest'], prob=0.9, padding_mode='constant'),
            RandRotated(keys=['img_mri', 'img_cbct'], range_x=math.pi, mode=['bilinear', 'nearest'], prob=1.0),
            RandAffined(keys=['img_mri', 'img_cbct'], prob=0.8, shear_range=(0.1, 0.1), mode=['bilinear', 'nearest'], padding_mode='zeros'),
            ToTensord(keys=['img_mri', 'img_cbct'])
            ]

            )

    def __call__(self, inp):
        return self.train_transform(inp) 
    

    
    
class LotusDataModule(pl.LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last        

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = monai.data.Dataset(data=LotusDataset(self.df_train, self.mount_point, img_column=self.img_column), transform=self.train_transform)
        self.val_ds = monai.data.Dataset(data=LotusDataset(self.df_val, self.mount_point, img_column=self.img_column), transform=self.valid_transform)
        self.test_ds = monai.data.Dataset(data=LotusDataset(self.df_test, self.mount_point, img_column=self.img_column), transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)
    

class LotusDataset(Dataset):
    def __init__(self, df, mount_point = "./", img_column="img_path", seg_column=None):
        self.df = df
        self.mount_point = mount_point
        self.img_column = img_column
        self.seg_column = seg_column
        
        self.loader = LoadImaged(keys=["img_mri", "img_cbct"])

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

        d = {"img_mri": img_path, "img_cbct": img_path}
        
        
        d = self.loader(d)
        
        return d