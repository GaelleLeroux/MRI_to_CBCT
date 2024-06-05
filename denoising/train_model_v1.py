import argparse
import pandas as pd
import os
from loader import TrainTransform,ValTransform,Noise
from monai.data import DataLoader,Dataset
import torch
import matplotlib.pyplot as plt
import numpy as np

from monai.networks.nets import AutoEncoder
from tqdm import trange
import torch.nn.functional as F

def plot_ims(ims, shape=None, figsize=(10, 10), titles=None):
    shape = (1, len(ims)) if shape is None else shape
    fig, axs = plt.subplots(*shape, figsize=figsize)
    axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs]
    
    for i, im in enumerate(ims):
        im = plt.imread(im) if isinstance(im, str) else torch.squeeze(im).cpu().numpy()
        axs[i].imshow(im, cmap="gray")
        if titles is not None:
            axs[i].set_title(titles[i])
        axs[i].axis("off")
        
    plt.tight_layout()
    plt.show()
    
def train(dict_key_for_training, train_loader, val_loader, max_epochs=10, patience=50,learning_rate=1e-3, model_dir='models'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(4, 8, 16, 32),
        strides=(2, 2, 2, 2),
    ).to(device)

    # Create loss fn and optimiser
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    epoch_loss_values = []
    val_loss_values = []

    best_val_loss = float('inf')
    last_save=0
    best_model_path = False
    t = trange(max_epochs, desc=f"{dict_key_for_training} -- epoch 0, avg loss: inf", leave=True)
    for epoch in t:
        last_save+=1
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs = batch_data[dict_key_for_training].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, batch_data["orig"].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data[dict_key_for_training].to(device)
                val_outputs = model(val_inputs)
                val_loss += F.mse_loss(val_outputs, val_data["orig"].to(device)).item()
        val_loss /= len(val_loader)
        val_loss_values.append(val_loss)
        
        # Save the best model
        if val_loss < best_val_loss:
            last_save = 0
            print("amelioration de la val loss de : ",best_val_loss-val_loss)
            best_val_loss = val_loss
            model_path = os.path.join(model_dir, f'best_model_{dict_key_for_training}_epoch={epoch+1}_loss={val_loss:.4f}.pth')
            torch.save(model.state_dict(), model_path)
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_model_path = model_path
            
        if last_save>patience:
            print(f"No amelioration since {patience} epochs")
        
        t.set_description(  # noqa: B038
            f"{dict_key_for_training} -- epoch {epoch + 1}" + f", avg loss: {epoch_loss:.4f}, val loss: {val_loss:.4f}"
        )
    return model, epoch_loss_values, val_loss_values

def get_single_im(ds):
    loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=10, shuffle=True)
    itera = iter(loader)
    return next(itera)

def main(args):
    mount_point = args.mount_point
    col = args.img_column

    df_train_cbct = pd.read_csv(args.train_cbct) 
    df_val_cbct = pd.read_csv(args.val_cbct)  
    df_test_cbct = pd.read_csv(args.test_cbct)

    train_datadict=[]
    for i in range(len(df_train_cbct[col])):
        train_datadict.append({"img":os.path.join(mount_point,df_train_cbct[col][i])})
    
    val_datadict=[]   
    for i in range(len(df_val_cbct[col])):
        val_datadict.append({"img":os.path.join(mount_point,df_val_cbct[col][i])})
    
    batch_size = args.batch_size
    num_workers = args.num_workers

    train_transforms = TrainTransform(resize=args.resize,noise=Noise(var_gauss=args.var_gauss,var_sp=args.var_sp))
    val_transforms = ValTransform(resize=args.resize,noise=Noise(var_gauss=args.var_gauss,var_sp=args.var_sp))
    
    train_ds = Dataset(train_datadict, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    val_ds = Dataset(val_datadict, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    test_ds = Dataset(df_test_cbct, val_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # DECOMMENT TO PRINT EXAMPLE NOISE
    ##########################################################
    # for data in train_loader:
    #     img_ori = data["orig"]
    #     img_after_gaus = data["gaus"]
    #     img_after_speckle = data["speckle"]

    #     # Print shapes and types for debugging
    #     print("Original Image Shape: ", img_ori.shape)
    #     print("Gaussian Noise Image Shape: ", img_after_gaus.shape)
    #     print("Speckle Noise Image Shape: ", img_after_speckle.shape)

    #     slice = 128  # Adjust the slice value as necessary
    #     plot_ims(
    #         [img_ori[0, 0, slice, :, :], img_ori[0, 0, :, slice, :], img_ori[0, 0, :, :, slice],
    #          img_after_gaus[0, 0, slice, :, :], img_after_gaus[0, 0, :, slice, :], img_after_gaus[0, 0, :, :, slice],
    #          img_after_speckle[0, 0, slice, :, :], img_after_speckle[0, 0, :, slice, :], img_after_speckle[0, 0, :, :, slice]],
    #         shape=(3, 3),
    #         titles=["Original - Axis 1", "Original - Axis 2", "Original - Axis 3",
    #                 "Gaussian Noise - Axis 1", "Gaussian Noise - Axis 2", "Gaussian Noise - Axis 3",
    #                 "Speckle Noise - Axis 1", "Speckle Noise - Axis 2", "Speckle Noise - Axis 3"]
    #     )
    #     break
    ##########################################################
    
    max_epochs = args.epochs
    training_types = ["orig", "gauss", "speckle"]
    models = {"orig":[],"gauss":[],"speckle":[]}
    epoch_losses = {"orig":[],"gauss":[],"speckle":[]}
    val_losses = {"orig":[],"gauss":[],"speckle":[]}
    for training_type in training_types:
        model, epoch_loss,val_loss_values = train(training_type,train_loader=train_loader,val_loader=val_loader, max_epochs=max_epochs,model_dir=args.out,patience=args.patience)
        models[training_type].append(model)
        epoch_losses[training_type].append(epoch_loss)
        val_losses[training_type].append(val_loss_values)
        
    # Plot and save the training loss
    plt.figure()
    plt.title("Epoch Average Training Loss")
    plt.xlabel("epoch")
    for training_type in training_types:
        x = list(range(1, len(epoch_losses[training_type][0]) + 1))
        (line,) = plt.plot(x, epoch_losses[training_type][0])
        line.set_label(training_type)
    plt.legend()
    plt.savefig(os.path.join(args.out, "epoch_training_loss.png"))
    plt.close()
    
    # Plot and save the validation loss
    plt.figure()
    plt.title("Epoch Average Validation Loss")
    plt.xlabel("epoch")
    for training_type in training_types:
        x = list(range(1, len(val_losses[training_type][0]) + 1))
        (line,) = plt.plot(x, val_losses[training_type][0])
        line.set_label(training_type)
    plt.legend()
    plt.savefig(os.path.join(args.out, "epoch_validation_loss.png"))
    plt.close()
    
    data = get_single_im(val_ds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    recons = []
    for model, training_type in zip(models, training_types):
        im = data[training_type]
        recon = model(im.to(device)).detach().cpu()
        recons.append(recon)

    plot_ims(
        [data["orig"], data["gaus"], data["s&p"]] + recons,
        titles=["orig", "Gaussian", "S&P"] + ["recon w/\n" + x for x in training_types],
        shape=(2, len(training_types)),
    )

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Diffusion training')

    hparams_group = parser.add_argument_group('Hyperparameters')
    
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience for early stopping', type=int, default=50)
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=1)
    
    hparams_group.add_argument('--resize', help='Resize used in transform', type=int, default=256)  
    hparams_group.add_argument('--var_gauss', help='Variable gaussian noise', type=float, default=0.001)    
    hparams_group.add_argument('--var_sp', help='Variable speckle noise', type=float, default=0.5)      
    

    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_CBCT")    
    input_group.add_argument('--img_column', type=str, default='img_fn', help='Column name for image')
    input_group.add_argument('--num_workers', type=int, default=4, help='number workers')
    

    input_group.add_argument('--train_cbct', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_CBCT/train.csv", help='path csv cbct train')  
    input_group.add_argument('--val_cbct', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_CBCT/valid.csv", help='path csv cbct valid')  
    input_group.add_argument('--test_cbct', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/training_CBCT/test.csv", help='path csv cbct test')  
    
    output_group = parser.add_argument_group('Ouput')
    output_group.add_argument('--out', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT/output_model_denoising", help='path output model')  



    args = parser.parse_args()
    print("args : ",args)

    main(args)