import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from data_loader_gaelle_v2 import LotusEvalTransforms, LotusDataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
import nibabel as nib

import monai
from monai.data import ImageWriter
# from callbacks import logger as LOGGER

# from nets import lotus
import SimpleITK as sitk

import pickle
from nets.cut import Cut

import nrrd


def main(args):

    if(os.path.splitext(args.csv)[1] == ".csv"):
        df = pd.read_csv(args.csv)
    else:
        df = pd.read_parquet(args.csv)
        
    print("df : ",df)
        
    model = Cut()
    checkpoint = torch.load(args.model, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    
    valid_transform = LotusEvalTransforms()

    test_ds = monai.data.Dataset(data=LotusDataset(df, args.mount_point, img_column=args.img_column), transform=valid_transform)
    print("test ds : ",test_ds)
    test_dl = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    
    with torch.no_grad():
        for idx, batch in enumerate(test_dl):
            row = df.iloc[idx]
            mri_cbct = model(batch['img'])
            
            print("row : ",row["img_fn"])


            out_seg_path = os.path.join(args.out,os.path.basename(row["img_fn"]))
            print("out_seg_path : ",out_seg_path)
            out_seg_dir = os.path.dirname(args.out)

            if not os.path.exists(out_seg_dir):
                os.makedirs(out_seg_dir)
            print('Writing:', out_seg_path)
            output_array = mri_cbct.squeeze().cpu().numpy()

            # Extract metadata
            metadata = batch['img'].meta
            print("metadata : ", metadata)
            spacing = metadata['pixdim'][1:4].tolist()
            origin = [-float(metadata['affine'][0, 3]), -float(metadata['affine'][1, 3]), float(metadata['affine'][2, 3])]
            direction = metadata['affine'][:3, :3].flatten().tolist()
            direction = [-direction[0], -direction[1], direction[2], -direction[3], -direction[4], direction[5], -direction[6], -direction[7], direction[8]]

            
            print("spacing : ",spacing)
            print("origin : ",origin)
            print("direction : ",direction)

            # Convert numpy array to SimpleITK image
            sitk_image = sitk.GetImageFromArray(output_array)
            sitk_image.SetSpacing(spacing)
            sitk_image.SetOrigin(origin)
            sitk_image.SetDirection(direction)
    
            # Save the SimpleITK image
            sitk.WriteImage(sitk_image, out_seg_path)
        


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Lotus prediction')
    input_group = parser.add_argument_group('Input')
      
    input_group.add_argument('--model', help='Model with trained weights', type=str, required=True)
           
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_MRI")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=1)
    input_group.add_argument('--csv', required=True, type=str, help='Test CSV')
    input_group.add_argument('--img_column', type=str, default="img_fn", help='Column name for image')   

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")

    args = parser.parse_args()

    main(args)
