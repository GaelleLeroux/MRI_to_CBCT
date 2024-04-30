import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from nets.cut import Cut
from pytorch_lightning import Trainer
from data_loader_gaelle import ConcatDataset,ImageDataset, LotusDataModule, ConcatDataModule, LotusTrainTransforms2
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



df_train_cbct = pd.read_csv("/home/luciacev/Documents/Gaelle/Data/MultimodelReg_2/MRI_to_CBCT/training_CBCT/train.csv") 
df_val_cbct = pd.read_csv("/home/luciacev/Documents/Gaelle/Data/MultimodelReg_2/MRI_to_CBCT/training_CBCT/valid.csv")  
df_test_cbct = pd.read_csv("/home/luciacev/Documents/Gaelle/Data/MultimodelReg_2/MRI_to_CBCT/training_CBCT/test.csv")

df_train_mri = pd.read_csv("/home/luciacev/Documents/Gaelle/Data/MultimodelReg_2/MRI_to_CBCT/training_CBCT/train.csv")
df_val_mri = pd.read_csv("/home/luciacev/Documents/Gaelle/Data/MultimodelReg_2/MRI_to_CBCT/training_CBCT/valid.csv")   
df_test_mri = pd.read_csv("/home/luciacev/Documents/Gaelle/Data/MultimodelReg_2/MRI_to_CBCT/training_CBCT/test.csv") 


print("df_train_cbct : ",df_train_cbct)

transform = transforms.Compose(
    [   
        # EnsureChannelFirstd(keys=['img', 'seg'], channel_dim=-1),
        LoadImaged(keys=['img_mri', 'img_cbct']),
        RandZoomd(keys=['img_mri', 'img_cbct'], min_zoom=0.8, max_zoom=1.1, mode=['area', 'nearest'], prob=0.9, padding_mode='constant'),
        RandRotated(keys=['img_mri', 'img_cbct'], range_x=math.pi, mode=['bilinear', 'nearest'], prob=1.0),
        RandAffined(keys=['img_mri', 'img_cbct'], prob=0.8, shear_range=(0.1, 0.1), mode=['bilinear', 'nearest'], padding_mode='zeros'),
        ToTensord(keys=['img_mri', 'img_cbct'])
    ]

)

train_transform_mri = LotusTrainTransforms2()
valid_transform_mri = LotusTrainTransforms2()
MRI_data = LotusDataModule(df_train_mri, df_val_mri, df_test_mri, mount_point=".", batch_size=2, num_workers=4, img_column="img_path", seg_column="seg_path", train_transform=train_transform_mri, valid_transform=valid_transform_mri, test_transform=valid_transform_mri, drop_last=False)

MRI_data.setup()


train_transform_cbct = LotusTrainTransforms2()
valid_transform_cbct = LotusTrainTransforms2()
CBCT_data = LotusDataModule(df_train_cbct, df_val_cbct, df_test_cbct, mount_point=".", batch_size=2, num_workers=4, img_column="img_path", seg_column="seg_path", train_transform=train_transform_cbct, valid_transform=valid_transform_cbct, test_transform=valid_transform_cbct, drop_last=False)
CBCT_data.setup()

concat_data = ConcatDataModule(MRI_data.train_ds, MRI_data.val_ds, CBCT_data.train_ds, CBCT_data.val_ds, batch_size=2, num_workers=4)

print("concat_data : ",concat_data)

checkpoint_callback = ModelCheckpoint(
            dirpath='/home/luciacev/Documents/Gaelle/Data/MultimodelReg_2/MRI_to_CBCT/output_train/',
            filename='{epoch}-{val_loss:.2f}',
            save_top_k=2,
            monitor='val_loss'
        )

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=30, verbose=True, mode="min")

callbacks=[early_stop_callback, checkpoint_callback]
neptune_logger = None

os.environ['NEPTUNE_API_TOKEN'] = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ZDQ0NTI4Yi03ZWI3LTRiN2UtODAwMi04MThhYzAwNWJhZDgifQ=='

neptune_logger = NeptuneLogger(
    project='gaellel/MRICBCT',
    tags=["v1"],
    api_key=os.environ['NEPTUNE_API_TOKEN']
)

LOGGER = getattr(logger, "CutLogger")    
image_logger = LOGGER(log_steps=100)
callbacks.append(image_logger)

# LOGGER = getattr(logger, "CutLogger")    
# image_logger = LOGGER(log_steps=100)
# callbacks.append(image_logger)
model = Cut()
trainer = Trainer(
    logger=neptune_logger,
    log_every_n_steps=100,
    max_epochs=200,
    max_steps=-1,
    callbacks=callbacks,
    accelerator='gpu', 
    devices=torch.cuda.device_count(),
    strategy=DDPStrategy(find_unused_parameters=False),
    reload_dataloaders_every_n_epochs=1
    # detect_anomaly=True
)
    
trainer.fit(model, datamodule=concat_data)