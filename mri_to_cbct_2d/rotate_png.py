from PIL import Image
import os
import numpy as np
import argparse

def main(args):
    # Chemins des dossiers
    input_folder = args.input
    output_folder = args.output

    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    # Parcourir toutes les images PNG dans le dossier d'entrée
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # Chemin complet de l'image
            img_path = os.path.join(input_folder, filename)
            
            # Ouvrir l'image
            with Image.open(img_path) as img:
                img_array = np.array(img)
            
                # Appliquer la transposition
                if len(img_array.shape) == 2:  # Image en niveaux de gris
                    transposed_img_array = img_array.T
                else:  # Image en couleur
                    transposed_img_array = img_array.transpose(1, 0, 2)
                
                # Convertir le numpy array transposé en image
                transposed_img = Image.fromarray(transposed_img_array)
                
                # Chemin complet de l'image transposée
                transposed_img_path = os.path.join(output_folder, filename)
                
                # Enregistrer l'image transposée
                transposed_img.save(transposed_img_path)

    print("Transposition des images terminée.")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Get nifti info')
    parser.add_argument('--input', type=str, default='/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT_2D/a1_training_MRI_2D/train/M023_OR', help='Input folder')
    parser.add_argument('--output', type=str, default='/home/lucia/Documents/Gaelle/Data/MultimodelReg/MRI_to_CBCT_2D/a1_training_MRI_2D/train2', help='Output directory tosave the png')
    args = parser.parse_args()



    main(args)
