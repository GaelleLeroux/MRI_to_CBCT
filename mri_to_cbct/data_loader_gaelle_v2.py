import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from nets.cut import Cut
from pytorch_lightning import Trainer
# from data_loader_gaelle import ConcatDataset,ImageDataset,LotusDataModule, ConcatDataModule, LotusTrainTransforms2
import pandas as pd
import numpy as np 

from callbackss import logger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

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
    ToTensord,
    EnsureTyped,
    Spacingd,
    Resized,
    EnsureChannelFirstd,
    EnsureChannelFirst,
    RandFlip,
    RandRotate,
    SpatialPad,
    RandSpatialCrop,
    ScaleIntensity,
    RandAdjustContrast,
    RandGaussianNoise,
    RandGaussianSmooth,
    NormalizeIntensity,
    ResizeWithPadOrCrop,
    ToTensor,
    ScaleIntensityRangePercentilesd
)   
import math

import pytorch_lightning as pl

class CleftDataset(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img', class_column='Classification', transform=None):
        self.df = df
        

        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.loc[idx]
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.mount_point, row[self.img_column])))

        if self.transform:
            img = self.transform(img)
        
        return img

    
class CleftDataModule(pl.LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=32, num_workers=4, img_column='img_path', class_column='Classification', train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column = class_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = CleftDataset(self.df_train, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.train_transform)
        self.val_ds = CleftDataset(self.df_val, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.valid_transform)
        self.test_ds = CleftDataset(self.df_test, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last)
    
# class CleftTrainTransforms:
#     def __init__(self, size=128, pad=32):
#         # image augmentation functions        
#         self.train_transform = Compose(
#             [
#                 EnsureChannelFirst(channel_dim='no_channel'),                
#                 RandFlip(prob=0.5),
#                 RandRotate(prob=0.5, range_x=math.pi, range_y=math.pi, range_z=math.pi, mode="nearest", padding_mode='zeros'),
#                 SpatialPad(spatial_size=size + pad),
#                 RandSpatialCrop(roi_size=size, random_size=False),
#                 ScaleIntensity(),
#                 RandAdjustContrast(prob=0.5),
#                 RandGaussianNoise(prob=0.5),
#                 RandGaussianSmooth(prob=0.5),
#                 # NormalizeIntensity(),
#                 ToTensor(dtype=torch.float32, track_meta=False)
#             ]
#         )
#     def __call__(self, inp):
#         return self.train_transform(inp)


class LotusTrainTransforms2:
    def __init__(self, height: int = 256,size=128, pad=16):

        # image augmentation functions
        # self.train_transform = Compose(
        #     [   
        #     LoadImaged(keys=['img_mri', 'img_cbct']),
        #     RandZoomd(keys=['img_mri', 'img_cbct'], min_zoom=0.8, max_zoom=1.1, mode=['area', 'nearest'], prob=0.9, padding_mode='constant'),
        #     RandRotated(keys=['img_mri', 'img_cbct'], range_x=math.pi, mode=['bilinear', 'nearest'], prob=1.0),
        #     RandAffined(keys=['img_mri', 'img_cbct'], prob=0.8, shear_range=(0.1, 0.1), mode=['bilinear', 'nearest'], padding_mode='zeros'),
        #     ToTensord(keys=['img_mri', 'img_cbct'])
        #     ]
        # )
        self.train_transform = Compose(
            [
        EnsureChannelFirstd(keys=['img']),

        EnsureTyped(keys=['img']),
        RandZoomd(keys=['img'], min_zoom=0.8, max_zoom=1.2, mode=['area'], prob=0.8, padding_mode='constant'),
        Resized(keys=['img'], spatial_size=(128, 128, 128)),  # Redimensionne les images
        
        # RandRotated(keys=['img'], range_x=np.pi/16, mode=['bilinear'], prob=1.0),

        RandAffined(keys=['img'], prob=0.8, shear_range=(0.1, 0.5), mode=['bilinear'], padding_mode='zeros'),
        ScaleIntensityRangePercentilesd(keys=['img'], lower=0.0, upper=100.0, b_min=0.0, b_max=1.0), 

        

        ToTensord(keys=['img']),
        ]
        )

        # self.train_transform = Compose(
        #     [
        #         EnsureChannelFirst(channel_dim='no_channel'),                
        #         RandFlip(prob=0.5),
        #         RandRotate(prob=0.5, range_x=math.pi, range_y=math.pi, range_z=math.pi, mode="nearest", padding_mode='zeros'),
        #         # SpatialPad(spatial_size=size + pad),
        #         RandSpatialCrop(roi_size=size, random_size=False),
        #         ScaleIntensity(),
        #         RandAdjustContrast(prob=0.5),
        #         RandGaussianNoise(prob=0.5),
        #         RandGaussianSmooth(prob=0.5),
        #         # NormalizeIntensity(),
        #         # ResizeWithPadOrCrop( spatial_size=(128, 128, 128)),
        #         ToTensor(dtype=torch.float32, track_meta=False)
        #     ]
        # )


    # def __call__(self, inp):
    #     return self.train_transform(inp) 
    def __call__(self, inp):
        try:
            # print("inp : ", inp)
            # print("key inp : ", inp.keys())
            y = self.train_transform(inp)
            return y
            # img_after = [self.train_transform(inp)["img"]]
            # return torch.stack([data for data in img_after])
        
        except Exception as e:
            print("Erreur lors de l'application de la transformation:", e)
            print("Dictionnaire d'entrée :", inp)
            raise e
        
class LotusValidTransforms2:
    def __init__(self, height: int = 256,size=128, pad=16):

        self.train_transform = Compose(
            [
        # LoadImaged(keys=['img']),
        EnsureChannelFirstd(keys=['img']),

        EnsureTyped(keys=['img']),
        

        Resized(keys=['img'], spatial_size=(128, 128, 128)),  # Redimensionne les images
        ScaleIntensityRangePercentilesd(keys=['img'], lower=0.0, upper=100.0, b_min=0.0, b_max=1.0), 

        ToTensord(keys=['img']),
        ]
        )


    # def __call__(self, inp):
    #     return self.train_transform(inp) 
    def __call__(self, inp):
        try:
            # print("inp : ", inp)
            # print("key inp : ", inp.keys())
            y = self.train_transform(inp)
            return y
            # img_after = [self.train_transform(inp)["img"]]
            # return torch.stack([data for data in img_after])
        
        except Exception as e:
            print("Erreur lors de l'application de la transformation:", e)
            print("Dictionnaire d'entrée :", inp)
            raise e

   

class LotusDataset(Dataset):
    def __init__(self, df, mount_point = "./", img_column="img_path", seg_column=None,transform=None):
        self.df = df
        self.mount_point = mount_point
        self.img_column = img_column
        self.seg_column = seg_column
        self.transform = transform
        
        self.loader = LoadImaged(keys=["img"])

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        

        d = {"img": img_path}
        # print("type d : ", type(d))
        
        d = self.loader(d)

        if self.transform:
            # print("Dictionnaire apres load : ", d)
            d = self.transform(d)

        # print("type d : ", type(d))
        # print("size d[img] : ", d["img"].size())
        # print("number key : ", len(d.keys()))
        
        return d["img"]
    
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
        # self.train_ds = monai.data.Dataset(data=LotusDataset(self.df_train, self.mount_point, img_column=self.img_column), transform=self.train_transform)
        # self.val_ds = monai.data.Dataset(data=LotusDataset(self.df_val, self.mount_point, img_column=self.img_column), transform=self.valid_transform)
        # self.test_ds = monai.data.Dataset(data=LotusDataset(self.df_test, self.mount_point, img_column=self.img_column), transform=self.test_transform)
        self.train_ds = LotusDataset(self.df_train, self.mount_point, img_column=self.img_column,transform=self.train_transform)
        self.val_ds = LotusDataset(self.df_val, self.mount_point, img_column=self.img_column,transform=self.valid_transform)
        self.test_ds = LotusDataset(self.df_test, self.mount_point, img_column=self.img_column,transform=self.test_transform)
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)
    
    def train_dataloader(self) :
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False, pin_memory=True, drop_last=self.drop_last)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)
    
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets, use_max=True):
        self.datasets = datasets
        print("OUIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
        print("datasets : ",datasets)
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
        self.train_ds =  ConcatDataset(self.datasetA_train, self.datasetB_train)
        self.val_ds = ConcatDataset(self.datasetA_val, self.datasetB_val)

    def train_dataloader(self):
        # print("train_ds : ",self.train_ds)
        
        self.train_ds.shuffle()
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)   