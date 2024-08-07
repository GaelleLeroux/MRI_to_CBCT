import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.distributed import is_initialized, get_rank

from loaders.ultrasound_dataset import LotusDataModule, USDataset
from loaders.mr_us_dataset import VolumeSlicingProbeParamsDataset, MUSTUSDataModule
from transforms.ultrasound_transforms import LotusEvalTransforms, LotusTrainTransforms, RealUSTrainTransforms, RealEvalTransforms
from callbacks import logger as LOGGER

from nets import lotus

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

from pytorch_lightning.loggers import NeptuneLogger
# from pytorch_lightning.plugins import MixedPrecisionPlugin

import pickle

import SimpleITK as sitk


def main(args):

    if(os.path.splitext(args.csv_train)[1] == ".csv"):
        df_train = pd.read_csv(args.csv_train)
        df_val = pd.read_csv(args.csv_valid)
        df_test = pd.read_csv(args.csv_test)
    else:
        df_train = pd.read_parquet(args.csv_train)
        df_val = pd.read_parquet(args.csv_valid) 
        df_test = pd.read_parquet(args.csv_test)  

    if(os.path.splitext(args.csv_train)[1] == ".csv"):
        df_train_us = pd.read_csv(args.csv_train_us)
        df_val_us = pd.read_csv(args.csv_valid_us)
        df_test_us = pd.read_csv(args.csv_test_us)
    else:
        df_train_us = pd.read_parquet(args.csv_train_us)
        df_val_us = pd.read_parquet(args.csv_valid_us) 
        df_test_us = pd.read_parquet(args.csv_test_us)  

    NN = getattr(lotus, args.nn)    
    model = NN(**vars(args))

    if args.nn_rendering:
        NN_render = getattr(lotus, args.nn_rendering)
        model.us_renderer = NN_render.load_from_checkpoint(args.model_rendering)


    train_transform = LotusTrainTransforms()
    valid_transform = LotusEvalTransforms()
    lotus_data = LotusDataModule(df_train, df_val, df_test, mount_point=args.mount_point, batch_size=args.batch_size, num_workers=4, img_column="img_path", seg_column="seg_path", train_transform=train_transform, valid_transform=valid_transform, test_transform=valid_transform, drop_last=False)
    lotus_data.setup()

    train_transform_us = RealUSTrainTransforms()
    valid_transform_us = RealEvalTransforms()

    us_ds_train = USDataset(df_train_us, args.mount_point, img_column='img_path', transform=train_transform_us, repeat_channel=False)
    us_ds_val = USDataset(df_val_us, args.mount_point, img_column='img_path', transform=valid_transform_us, repeat_channel=False)


    must_us_data = MUSTUSDataModule(lotus_data.train_ds, lotus_data.val_ds, us_ds_train, us_ds_val, batch_size=args.batch_size, num_workers=args.num_workers)


    # must_us_data.setup()

    # train_ds = must_us_data.train_dataloader()
    # for idx, batch in enumerate(train_ds):
    #     label, us = batch

    #     print("__")
    #     print(label['img'].shape)
    #     print(label['seg'].shape)
    #     print(us.shape)
    #     print("..")

    # quit()


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    callbacks=[early_stop_callback, checkpoint_callback]
    logger = None

    if args.neptune_tags:
        logger = NeptuneLogger(
            project='ImageMindAnalytics/Lotus',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN']
        )        
        IMAGE_LOGGER = getattr(LOGGER, args.logger)    
        image_logger = IMAGE_LOGGER(log_steps=args.log_steps)
        callbacks.append(image_logger)

    
    trainer = Trainer(
        logger=logger,
        log_every_n_steps=args.log_steps,
        max_epochs=args.epochs,
        max_steps=args.steps,
        callbacks=callbacks,
        accelerator='gpu', 
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=False)
        # detect_anomaly=True
    )
    
    trainer.fit(model, datamodule=must_us_data, ckpt_path=args.model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Diffusion training')

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience for early stopping', type=int, default=30)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=2)
    hparams_group.add_argument('--num_labels', help='Number of labels in the US model', type=int, default=340)
    hparams_group.add_argument('--grid_w', help='Grid size for the simulation', type=int, default=256)
    hparams_group.add_argument('--grid_h', help='Grid size for the simulation', type=int, default=256)
    hparams_group.add_argument('--center_x', help='Position of the circle that creates the transducer', type=float, default=128.0)
    hparams_group.add_argument('--center_y', help='Position of the circle that creates the transducer', type=float, default=-40.0)
    hparams_group.add_argument('--r1', help='Radius of first circle', type=float, default=20.0)
    hparams_group.add_argument('--r2', help='Radius of second circle', type=float, default=240.0)
    hparams_group.add_argument('--theta', help='Aperture angle of transducer', type=float, default=np.pi/4.0)
    hparams_group.add_argument('--alpha_coeff_boundary_map', help='Lotus model', type=float, default=0.1)
    hparams_group.add_argument('--beta_coeff_scattering', help='Lotus model', type=float, default=10)
    hparams_group.add_argument('--tgc', help='Lotus model', type=int, default=8)
    hparams_group.add_argument('--clamp_vals', help='Lotus model', type=int, default=0)
    
    hparams_group.add_argument('--perceptual_weight', help='Perceptual weight', type=float, default=1.0)
    hparams_group.add_argument('--adversarial_weight', help='Adversarial weight', type=float, default=1.0)    
    hparams_group.add_argument('--warm_up_n_epochs', help='Number of warm up epochs before starting to train with discriminator', type=int, default=1)
    hparams_group.add_argument('--latent_channels', help='Number latent channels', type=int, default=3)
    
    hparams_group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
    hparams_group.add_argument('--kl_weight', help='Weight decay for optimizer', type=float, default=1e-6)    


    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="RealUS")        
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--model_rendering', help='Trained rendering model for lotus', type=str, default= None)
    input_group.add_argument('--nn_rendering', help='Ultrasound rendering Type of neural network', type=str, default=None)        
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')    
    input_group.add_argument('--csv_test', required=True, type=str, help='Test CSV')  
    input_group.add_argument('--csv_train_us', required=True, type=str, help='Train CSV real us')
    input_group.add_argument('--csv_valid_us', required=True, type=str, help='Valid CSV real us')    
    input_group.add_argument('--csv_test_us', required=True, type=str, help='Test CSV real us')  
    input_group.add_argument('--img_column', type=str, default='img_path', help='Column name for image')  
    input_group.add_argument('--seg_column', type=str, default='seg_path', help='Column name for labeled/seg image')  
    # input_group.add_argument('--labeled_img', required=True, type=str, help='Labeled volume to grap slices from')    
    # input_group.add_argument('--csv_train_us', required=True, type=str, help='Train CSV')
    # input_group.add_argument('--csv_valid_us', required=True, type=str, help='Valid CSV')    

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    log_group.add_argument('--logger', help='Neptune tags', type=str, default="ImageLoggerLotusNeptune")
    log_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    log_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="diffusion")
    log_group.add_argument('--log_steps', help='Log every N steps', type=int, default=100)


    args = parser.parse_args()

    main(args)
