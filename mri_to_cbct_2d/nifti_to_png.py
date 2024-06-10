import nibabel as nib
import os
from PIL import Image
import argparse

def create_png(nii_gz_file_path,output_dir,filename,slice,transpose,flip_h,min_p,max_p):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Charger les données du fichier NIfTI
    nii_data = nib.load(nii_gz_file_path)

    # Extraire les données d'image sous forme d'un tableau numpy
    image_data = nii_data.get_fdata()

    # Itérer sur la troisième dimension (z) pour sauvegarder chaque tranche
    print("image_data.shape[slice] : ",image_data.shape[slice])
    
    min = int(image_data.shape[slice]*min_p)
    max = int(image_data.shape[slice]*max_p)
    print("min : ",min)
    print("max : ",max)
    for z in range(min,max):
        # Extraire la tranche z
        if slice==0 :
            slice_z = image_data[z, :, :]
        if slice==1 :
            slice_z = image_data[:, z, :]
        if slice==2 :
            slice_z = image_data[:, :, z]
        if slice_z.max() == slice_z.min():
            print("black slice detected")
            continue
        
        # Normaliser les données pour qu'elles soient dans l'intervalle [0, 255]
        final_img_array = ((slice_z - slice_z.min()) / (slice_z.max() - slice_z.min()) * 255).astype('uint8')
        if transpose :
            final_img_array = final_img_array.T
        
        # Convertir en image PIL et sauvegarder
        img = Image.fromarray(final_img_array)
        
        if flip_h :
            img = img.transpose(method=Image.ROTATE_90)
        
        img.save(os.path.join(output_dir, f'{filename}_slice_{z:03}.png'))

    print(f'Toutes les tranches ont été sauvegardées dans {output_dir}')
    
def main(args):
    input_folder = args.input
    output_general = args.output
    # Get all nifti files in the folder
    nifti_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                nifti_files.append(os.path.join(root, file))

    # Get nifti info for every nifti file
    for file in nifti_files:
        base_name = os.path.basename(file)  
        file_name_without_extension = os.path.splitext(os.path.splitext(base_name)[0])[0]
        # output_folder = os.path.join(output_general,file_name_without_extension)
        create_png(file,output_general,file_name_without_extension,args.slice,args.transpose,args.flip_h,args.min,args.max)
    print("All the files has been treated")
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Get nifti info')
    parser.add_argument('--input', type=str, default='/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT_2D/a0_training_CBCT/test', help='Input folder')
    parser.add_argument('--output', type=str, default='/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT_2D/a1_training_CBCT_2D/test', help='Output directory tosave the png')
    parser.add_argument('--slice', type=int, default=0, help='Slice to keep, mri:2, cbct:0')
    parser.add_argument('--transpose',type=bool,default=False,help='if output image need to be transpose (for mri)')
    parser.add_argument('--flip_h',type=bool,default=True,help='if output image need to be flip horizontally (for cbct)')
    parser.add_argument('--min',type=float,default=0.0,help='mri=cbct=0.0')
    parser.add_argument('--max',type=float,default=0.2,help='mri:1.0  cbct:0.5')
    args = parser.parse_args()



    main(args)