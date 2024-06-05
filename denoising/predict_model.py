import argparse
import os
import pandas as pd
import torch
from loader import TestTransform, ValTransform, Noise
from monai.data import DataLoader, Dataset
from monai.networks.nets import AutoEncoder
import SimpleITK as sitk

def save_image(image, metadata, save_path):
    # Extract metadata with robust handling
    if 'pixdim' in metadata:
        pixdim = metadata['pixdim'][0]  # Access the first row
        spacing = [float(pixdim[i].item()) for i in range(1, 4)]
    else:
        spacing = [1.0, 1.0, 1.0]  # Default spacing if not available

    if 'affine' in metadata:
        affine = metadata['affine'][0]  # Access the first matrix
        origin = [-float(affine[0, 3].item()), -float(affine[1, 3].item()), float(affine[2, 3].item())]
        direction = affine[:3, :3].flatten().tolist()
        direction = [-float(direction[0]) if torch.is_tensor(direction[0]) else float(direction[0]),
                    -float(direction[1]) if torch.is_tensor(direction[1]) else float(direction[1]),
                     float(direction[2]) if torch.is_tensor(direction[2]) else float(direction[2]),
                    -float(direction[3]) if torch.is_tensor(direction[3]) else float(direction[3]),
                    -float(direction[4]) if torch.is_tensor(direction[4]) else float(direction[4]),
                     float(direction[5]) if torch.is_tensor(direction[5]) else float(direction[5]),
                    -float(direction[6]) if torch.is_tensor(direction[6]) else float(direction[6]),
                    -float(direction[7]) if torch.is_tensor(direction[7]) else float(direction[7]),
                     float(direction[8]) if torch.is_tensor(direction[8]) else float(direction[8])]
    else:
        origin = [0.0, 0.0, 0.0]  # Default origin if not available
        direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # Default direction if not available

    # Convert numpy array to SimpleITK image
    sitk_image = sitk.GetImageFromArray(image)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)
    sitk_image.SetDirection(direction)

    # Save the SimpleITK image
    print("save_path : ",save_path)
    sitk.WriteImage(sitk_image, save_path)


def main(args):
    model_path = args.model
    save_path = args.out
    
    df_test_cbct = pd.read_csv(args.csv)
    test_datadict = []
    for i in range(len(df_test_cbct[args.img_column])):
        test_datadict.append({"img": os.path.join(args.mount_point, df_test_cbct[args.img_column][i])})
        
    print("test_datadict:", test_datadict)

    test_transforms = ValTransform(resize=256, noise=Noise(var_gauss=0.001, var_sp=0.5))
    test_ds = Dataset(test_datadict, test_transforms)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(4, 8, 16, 32),
        strides=(2, 2, 2, 2),
    ).to(device)

    # Load the best saved model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loss = 0
    loss_function = torch.nn.MSELoss()

    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            inputs = batch_data["gaus"].to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, batch_data["orig"].to(device))
            test_loss += loss.item()

            # Save the blurred (gaussian noise) image
            blurred_image = inputs.squeeze().cpu().numpy()
            base_filename = os.path.basename(test_datadict[batch_idx]["img"])
            if base_filename.endswith(".nii.gz"):
                blurred_save_path = os.path.join(save_path, base_filename.replace(".nii.gz", "_blurred.nii.gz"))
            else :
                blurred_save_path = os.path.join(save_path, base_filename.replace(".nii", "_blurred.nii"))
            save_image(blurred_image, batch_data['orig'].meta, blurred_save_path)
            
            # Save the deblurred image (model output)
            deblurred_image = outputs.squeeze().cpu().numpy()
            if base_filename.endswith(".nii.gz"):
                deblurred_save_path = os.path.join(save_path, base_filename.replace(".nii.gz", "_deblurred.nii.gz"))
            else :
                deblurred_save_path = os.path.join(save_path, base_filename.replace(".nii", "_deblurred.nii"))
            save_image(deblurred_image, batch_data['orig'].meta, deblurred_save_path)

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model testing')
    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--model', help='Model with trained weights', type=str, required=True)
    input_group.add_argument('--batch_size', help='Batch size for testing', type=int, default=1)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_CBCT")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=1)
    input_group.add_argument('--csv', required=True, type=str, help='Test CSV')
    input_group.add_argument('--img_column', type=str, default="img_fn", help='Column name for image')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")

    args = parser.parse_args()
    main(args)
