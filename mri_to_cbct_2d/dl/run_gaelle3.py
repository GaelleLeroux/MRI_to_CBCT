import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from nets.cut import Cut
from pytorch_lightning import Trainer
from data_loader_gaelle_v2 import ConcatDataset,LotusDataModule, ConcatDataModule, LotusValidTransforms2,LotusTrainTransforms2, CleftDataModule
import pandas as pd
import numpy as np 

from callbacks import logger
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

import argparse

torch.set_float32_matmul_precision('medium')
def main(args):

    df_train_cbct = pd.read_csv(args.train_cbct) 
    df_val_cbct = pd.read_csv(args.val_cbct)  
    df_test_cbct = pd.read_csv(args.test_cbct)

    df_train_mri = pd.read_csv(args.train_mri)
    df_val_mri = pd.read_csv(args.val_mri)   
    df_test_mri = pd.read_csv(args.test_mri) 


    print(df_train_mri.columns)  # Affiche les noms de colonnes du DataFrame
    print(df_train_mri.head())   # Affiche les premières lignes pour vérifier les données


    print("df_train_cbct : ",df_train_cbct)

    ################################################################################################################################################################3


    train_transform_cbct = LotusTrainTransforms2()
    valid_transform_cbct = LotusValidTransforms2()
    # CBCT_data = LotusDataModule(df_train_cbct, df_val_cbct, df_test_cbct, mount_point="/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT_2D/a1_training_CBCT_2D/", batch_size=2, num_workers=4, img_column="img_fn", train_transform=train_transform_cbct, valid_transform=valid_transform_cbct, test_transform=valid_transform_cbct, drop_last=False)
    CBCT_data = LotusDataModule(df_train_cbct, df_val_cbct, df_test_cbct, mount_point=args.mount_point_cbct, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, train_transform=train_transform_cbct, valid_transform=valid_transform_cbct, test_transform=valid_transform_cbct, drop_last=False)
    CBCT_data.setup()

    train_transform_mri = LotusTrainTransforms2()
    valid_transform_mri = LotusValidTransforms2()
    MRI_data = LotusDataModule(df_train_mri, df_val_mri, df_test_mri, mount_point=args.mount_point_mri, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, train_transform=train_transform_mri, valid_transform=valid_transform_mri, test_transform=valid_transform_mri, drop_last=False)
    # MRI_data = LotusDataModule(df_train_mri, df_val_mri, df_test_mri, mount_point="/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT_2D/a1_training_CBCT_2D/", batch_size=2, num_workers=4, img_column="img_fn", train_transform=train_transform_mri, valid_transform=valid_transform_mri, test_transform=valid_transform_mri, drop_last=False)
    MRI_data.setup()

    concat_data = ConcatDataModule(MRI_data.train_ds, MRI_data.val_ds, CBCT_data.train_ds, CBCT_data.val_ds, batch_size=1, num_workers=4)
    concat_data.setup()




    print("concat_data : ",concat_data)
    print("size of concat_data : ",len(concat_data.train_ds))


    ################################################################################################################################################################3

    checkpoint_callback = ModelCheckpoint(
                dirpath=args.out,
                filename='{epoch}-{val_loss:.2f}',
                save_top_k=2,
                monitor='val_loss'
            )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    callbacks=[early_stop_callback, checkpoint_callback]
    neptune_logger = None

    os.environ['NEPTUNE_API_TOKEN'] = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ZDQ0NTI4Yi03ZWI3LTRiN2UtODAwMi04MThhYzAwNWJhZDgifQ=='

    neptune_logger = NeptuneLogger(
        project='gaellel/MRICBCT',
        tags=args.neptune_tags,
        api_key=os.environ['NEPTUNE_API_TOKEN']
    )
    
    print("Hellooo")

    LOGGER = getattr(logger, args.logger)    
    image_logger = LOGGER(log_steps=30)
    callbacks.append(image_logger)

    ################################################################################################################################################################3

    # LOGGER = getattr(logger, "CutLogger")    
    # image_logger = LOGGER(log_steps=100)
    # callbacks.append(image_logger)
    model = Cut(**vars(args))
    trainer = Trainer(
        logger=neptune_logger,
        log_every_n_steps=args.log_steps,
        max_epochs=args.epochs,
        max_steps=args.steps,
        callbacks=callbacks,
        accelerator='gpu', 
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=False),
        reload_dataloaders_every_n_epochs=1
        # precision=16
        # detect_anomaly=True
    )

    torch.cuda.empty_cache()

    # trainer.fit(model, datamodule=concat_data)
    try:
        trainer.fit(model, datamodule=concat_data)
    finally:
        neptune_logger.experiment.wait()


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Diffusion training')

    hparams_group = parser.add_argument_group('Hyperparameters')
    
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience for early stopping', type=int, default=50)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=1)
    
    hparams_group.add_argument('--lr', '--learning-rate', type=float, help='Learning rate', default=1e-4)
    hparams_group.add_argument('--lambda_y', help='betas for optimizer', type=float, default=1.0)
    hparams_group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
    hparams_group.add_argument('--betas', help='betas for optimizer', type=tuple, default=(0.5, 0.999))


    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--mount_point_cbct', help='Dataset mount directory', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT_2D/a1_training_CBCT_2D")    
    input_group.add_argument('--mount_point_mri', help='Dataset mount directory', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT_2D/a1_training_MRI_2D/")    
    input_group.add_argument('--img_column', type=str, default='img_fn', help='Column name for image')
    input_group.add_argument('--num_workers', type=str, default=4, help='number workers')
    


    input_group.add_argument('--train_mri', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT_2D/a1_training_MRI_2D/train.csv", help='path csv mri train')  
    input_group.add_argument('--test_mri', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT_2D/a1_training_MRI_2D/test.csv", help='path csv mri test')  
    input_group.add_argument('--val_mri', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT_2D/a1_training_MRI_2D/valid.csv", help='path csv mri valid')  
    input_group.add_argument('--train_cbct', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT_2D/a1_training_CBCT_2D/train.csv", help='path csv cbct train')  
    input_group.add_argument('--test_cbct', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT_2D/a1_training_CBCT_2D/test.csv", help='path csv cbct test')  
    input_group.add_argument('--val_cbct', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT_2D/a1_training_CBCT_2D/valid.csv", help='path csv cbct valid')  


    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT_2D/output_model")

    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default="2D")
    log_group.add_argument('--logger', help='Neptune tags', type=str, nargs="+", default="CutLogger")
    log_group.add_argument('--log_steps', help='Log every N steps', type=int, default=100)
    


    args = parser.parse_args()
    print("args : ",args)

    main(args)