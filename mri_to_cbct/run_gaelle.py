from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from nets.cut import Cut
from pytorch_lightning import Trainer
from data_loader_gaelle import ConcatDataset,ImageDataset
import neptune

from monai.transforms import (  
    LoadImaged,      
    Compose,
    Resize,
    RandZoomd,
    RandRotated,
    RandAffined,
    ToTensord
)   

def main(args):
    
    if(os.path.splitext(args.csv_train)[1] == ".csv"):
        df_train = pd.read_csv(args.csv_train)
        df_val = pd.read_csv(args.csv_valid)
        df_test = pd.read_csv(args.csv_test)
    else:
        df_train = pd.read_parquet(args.csv_train)
        df_val = pd.read_parquet(args.csv_valid) 
        df_test = pd.read_parquet(args.csv_test)  

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


    train_transform = ImageDataset()
    valid_transform = ImageDataset()
    MRI_data = LotusDataModule(df_train, df_val, df_test, mount_point=args.mount_point, batch_size=args.batch_size, num_workers=4, img_column="img_path", seg_column="seg_path", train_transform=train_transform, valid_transform=valid_transform, test_transform=valid_transform, drop_last=False)

    MRI_data.setup()


    train_transform_b = ImageDataset()
    valid_transform_b = ImageDataset()
    CBCT_data = LotusDataModule(df_train, df_val, df_test, mount_point=args.mount_point, batch_size=args.batch_size, num_workers=4, img_column="img_path", seg_column="seg_path", train_transform=train_transform, valid_transform=valid_transform, test_transform=valid_transform, drop_last=False)
    CBCT_data.setup()

    concat_data = ConcatDataModule(MRI_data.train_ds, MRI_data.val_ds, CBCT_data.train_ds, CBCT_data.val_ds, batch_size=args.batch_size, num_workers=args.num_workers)
        
        
        
    # mri_dataset = ImageDataset('/home/luciacev/Documents/Gaelle/Data/MultimodelReg_2/MRI_to_CBCT/a01_MRI_left_OR/', transform=transform)
    # cbct_dataset = ImageDataset('/home/luciacev/Documents/Gaelle/Data/MultimodelReg_2/MRI_to_CBCT/a01_CTs_crop_left/', transform=transform)

    # # Concatenate datasets
    # concat_dataset = ConcatDataset(mri_dataset, cbct_dataset)

    checkpoint_callback = ModelCheckpoint(
            dirpath=args.out,
            filename='{epoch}-{val_loss:.2f}',
            save_top_k=2,
            monitor='val_loss'
        )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    callbacks=[early_stop_callback, checkpoint_callback]
    neptune_logger = None

    if args.neptune_tags:
        neptune_logger = NeptuneLogger(
            project='gaellel/MRICBCT',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN']
        )

        LOGGER = getattr(logger, args.logger)    
        image_logger = LOGGER(log_steps=args.log_steps)
        callbacks.append(image_logger)

    model = Cut()
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
        # detect_anomaly=True
    )

    trainer.fit(model, datamodule=concat_data)


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='MRI to CBCT training')
    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')    
    input_group.add_argument('--csv_test', required=True, type=str, help='Test CSV')  
    
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    log_group.add_argument('--logger', help='Neptune tags', type=str, nargs="+", default="CutLogger")
    log_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    log_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="diffusion")
    log_group.add_argument('--log_steps', help='Log every N steps', type=int, default=100)
    
    args = parser.parse_args()

    main(args)
