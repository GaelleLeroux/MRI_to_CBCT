import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from nets.cut import Cut
from pytorch_lightning import Trainer
from data_loader_gaelle_v2 import ConcatDataset,LotusDataModule, ConcatDataModule, LotusTrainTransforms2, CleftDataModule
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
    EnsureChannelFirstd
)   
import math

import pytorch_lightning as pl




df_train_cbct = pd.read_csv("/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_CBCT/train.csv") 
df_val_cbct = pd.read_csv("/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_CBCT/valid.csv")  
df_test_cbct = pd.read_csv("/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_CBCT/test.csv")

df_train_mri = pd.read_csv("/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_MRI/train.csv")
df_val_mri = pd.read_csv("/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_MRI/valid.csv")   
df_test_mri = pd.read_csv("/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_MRI/test.csv") 

# df_train_mri = pd.read_csv("/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_CBCT/train.csv") 
# df_val_mri = pd.read_csv("/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_CBCT/valid.csv")  
# df_test_mri = pd.read_csv("/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_CBCT/test.csv")


print(df_train_mri.columns)  # Affiche les noms de colonnes du DataFrame
print(df_train_mri.head())   # Affiche les premières lignes pour vérifier les données


print("df_train_cbct : ",df_train_cbct)

################################################################################################################################################################3


train_transform_cbct = LotusTrainTransforms2()
valid_transform_cbct = LotusTrainTransforms2()
# CBCT_data = LotusDataModule(df_train_cbct, df_val_cbct, df_test_cbct, mount_point="/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_CBCT/", batch_size=2, num_workers=4, img_column="img_fn", train_transform=train_transform_cbct, valid_transform=valid_transform_cbct, test_transform=valid_transform_cbct, drop_last=False)
CBCT_data = LotusDataModule(df_train_cbct, df_val_cbct, df_test_cbct, mount_point="/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_CBCT/", batch_size=2, num_workers=4, img_column="img_fn", train_transform=train_transform_cbct, valid_transform=valid_transform_cbct, test_transform=valid_transform_cbct, drop_last=False)
CBCT_data.setup()

train_transform_mri = LotusTrainTransforms2()
valid_transform_mri = LotusTrainTransforms2()
MRI_data = LotusDataModule(df_train_mri, df_val_mri, df_test_mri, mount_point="/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_MRI/", batch_size=2, num_workers=4, img_column="img_fn", train_transform=train_transform_mri, valid_transform=valid_transform_mri, test_transform=valid_transform_mri, drop_last=False)
# MRI_data = LotusDataModule(df_train_mri, df_val_mri, df_test_mri, mount_point="/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_CBCT/", batch_size=2, num_workers=4, img_column="img_fn", train_transform=train_transform_mri, valid_transform=valid_transform_mri, test_transform=valid_transform_mri, drop_last=False)
MRI_data.setup()

concat_data = ConcatDataModule(MRI_data.train_ds, MRI_data.val_ds, CBCT_data.train_ds, CBCT_data.val_ds, batch_size=2, num_workers=4)
concat_data.setup()




print("concat_data : ",concat_data)
print("size of concat_data : ",len(concat_data.train_ds))


################################################################################################################################################################3

checkpoint_callback = ModelCheckpoint(
            dirpath='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/output_model/',
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

################################################################################################################################################################3

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
    reload_dataloaders_every_n_epochs=1,
    precision=16
    # detect_anomaly=True
)

torch.cuda.empty_cache()
print("JE SUIS AVANT LE FIT") 
trainer.fit(model, datamodule=concat_data)