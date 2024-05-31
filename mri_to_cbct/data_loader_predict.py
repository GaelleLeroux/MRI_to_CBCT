import SimpleITK as sitk
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class LotusEvalTransforms:
    def __init__(self, height: int = 256, size=128, pad=16):
        self.height = height
        self.size = size
        self.pad = pad

    

    def ensure_channel_first(self, image):
        array = sitk.GetArrayFromImage(image)
        if len(array.shape) == 3:  # if the image is grayscale, add a channel dimension
            array = np.expand_dims(array, axis=0)
        image = sitk.GetImageFromArray(array)
        return image

    def ensure_type(self, image):
        # Assuming the image needs to be of type float32
        image = sitk.Cast(image, sitk.sitkFloat32)
        return image

    def resize(self, image, new_size=(128, 128, 128)):
        resample = sitk.ResampleImageFilter()
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()

        new_spacing = [
            original_spacing[0] * (original_size[0] / new_size[0]),
            original_spacing[1] * (original_size[1] / new_size[1]),
            original_spacing[2] * (original_size[2] / new_size[2])
        ]

        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(new_size)
        resample.SetInterpolator(sitk.sitkLinear)
        resampled_image = resample.Execute(image)
        return resampled_image

    def scale_intensity(self, image, lower=0.0, upper=100.0, b_min=0.0, b_max=1.0):
        array = sitk.GetArrayFromImage(image)
        p_low, p_high = np.percentile(array, (lower, upper))
        array = (array - p_low) / (p_high - p_low)
        array = np.clip(array, b_min, b_max)
        image = sitk.GetImageFromArray(array)
        return image

    def to_tensor(self, image):
        array = sitk.GetArrayFromImage(image)
        tensor = np.expand_dims(array, axis=0)
        return tensor

    def __call__(self, inp):
        try:
            # Assuming inp is a dictionary with key 'img' containing the file path
            if 'img' in inp:
                image = inp['img']
                image = self.ensure_channel_first(image)
                image = self.ensure_type(image)
                image = self.resize(image, new_size=(self.size, self.size, self.size))
                image = self.scale_intensity(image, lower=0.0, upper=100.0, b_min=0.0, b_max=1.0)
                tensor = self.to_tensor(image)
                inp['img'] = tensor  # Update the dictionary with the transformed image tensor
                return inp
            else:
                raise ValueError("Le dictionnaire d'entrée ne contient pas la clé 'img'.")

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
        
        # self.loader = (keys=["img"])

    def __len__(self):
        return len(self.df.index)
    
    def load_image(self, filepath):
        image = sitk.ReadImage(filepath)
        return image

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        

        d = {"img": img_path}
        # print("type d : ", type(d))
        
        d = self.load_image(img_path)
        d = {"img": d}

        if self.transform:
            # print("Dictionnaire apres load : ", d)
            d = self.transform(d)

        return d
        
# Example usage:
# transforms = LotusEvalTransforms()
# transformed_image = transforms('path/to/image.nii')
