import os
import torch

# Remplacez l'index par celui de votre GPU si nécessaire
gpu_index = 0

# Définir la variable d'environnement CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
print(os.environ["CUDA_VISIBLE_DEVICES"])

# Vérifier si CUDA est disponible
cuda_available = torch.cuda.is_available()
print(torch.version.cuda)

if cuda_available:
    # Vérifier le nombre de GPU disponibles
    num_gpus = torch.cuda.device_count()
    print("num_gpus : ",num_gpus)

    # Afficher des informations sur chaque GPU disponible
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")

    # Afficher le GPU actuellement utilisé par PyTorch
    current_device = torch.cuda.current_device()
    print(f"GPU actuel: {torch.cuda.get_device_name(current_device)}")

    # Afficher la version de CUDA
    cuda_version = torch.version.cuda
    print(f"Version CUDA: {cuda_version}")
else:
    print("CUDA n'est pas disponible sur cet environnement.")