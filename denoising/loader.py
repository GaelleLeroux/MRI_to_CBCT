import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from skimage.util import random_noise
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, RandRotated, RandFlipd,
    RandZoomd, Resized, ScaleIntensityRangePercentilesd, ScaleIntensityd, Lambda
)
from monai.data import Dataset, DataLoader



class Noise:
    def __init__(self, var_gauss=0.001,var_sp=0.5):
        self.var_gauss = var_gauss
        self.var_sp = var_sp
        
    def __call__(self, d, *args, **kwargs):
        return {
            "orig": d["img"],
            "gaus": torch.tensor(random_noise(d["img"], mode="gaussian", var=self.var_gauss), dtype=torch.float32),
            "speckle": torch.tensor(random_noise(d["img"], mode="speckle", var=self.var_sp), dtype=torch.float32),
        }


class TrainTransform:
    def __init__(self, resize=256,noise=Noise()):
        self.train_transforms = Compose(
    [
        LoadImaged(keys=["img"]),
        EnsureChannelFirstd(keys=["img"]),
        EnsureTyped(keys=['img']),
        RandRotated(keys=["img"], range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlipd(keys=["img"], spatial_axis=0, prob=1),
        RandZoomd(keys=["img"], min_zoom=0.9, max_zoom=1.1, prob=0.5),
        Resized(keys=['img'], spatial_size=(resize, resize, resize)),
        ScaleIntensityRangePercentilesd(keys=['img'], lower=0.0, upper=100.0, b_min=0.0, b_max=1.0), 
        Lambda(noise)
    ]
    )
        
    def __call__(self, inp):
        try:
            y = self.train_transforms(inp)
            return y
        
        except Exception as e:
            print("Erreur lors de l'application de la transformation:", e)
            print("Dictionnaire d'entrée :", inp)
            raise e
        

class ValTransform:
    def __init__(self, resize=256,noise=Noise()):
        self.val_transforms = Compose(
    [
        LoadImaged(keys=["img"]),
        EnsureChannelFirstd(keys=["img"]),
        EnsureTyped(keys=["img"]),
        Resized(keys=['img'], spatial_size=(resize, resize, resize)),
        ScaleIntensityRangePercentilesd(keys=['img'], lower=0.0, upper=100.0, b_min=0.0, b_max=1.0), 
        Lambda(noise)
    ]
    )
        
    def __call__(self, inp):
        try:
            y = self.val_transforms(inp)
            return y
        
        except Exception as e:
            print("Erreur lors de l'application de la transformation:", e)
            print("Dictionnaire d'entrée :", inp)
            raise e
        
        
class TestTransform:
    def __init__(self, resize=256,noise=Noise()):
        self.val_transforms = Compose(
    [
        LoadImaged(keys=["img"]),
        EnsureChannelFirstd(keys=["img"]),
        EnsureTyped(keys=["img"]),
        # Resized(keys=['img'], spatial_size=(resize, resize, resize)),
        ScaleIntensityRangePercentilesd(keys=['img'], lower=0.0, upper=100.0, b_min=0.0, b_max=1.0), 
        Lambda(noise)
    ]
    )
        
    def __call__(self, inp):
        try:
            y = self.val_transforms(inp)
            return y
        
        except Exception as e:
            print("Erreur lors de l'application de la transformation:", e)
            print("Dictionnaire d'entrée :", inp)
            raise e

