# datareader.py
"""
datareader.py
Module for loading and preprocessing the ChestMNIST dataset for binary classification between Pneumonia and Edema.
Classes:
    FilteredBinaryDataset(Dataset):
        A PyTorch Dataset that filters the ChestMNIST dataset to only include samples with a single label of either Pneumonia or Edema, and remaps their labels to 0 (Pneumonia) and 1 (Edema).
Functions:
    get_data_loaders(batch_size):
        Returns PyTorch DataLoaders for the filtered binary dataset with data augmentation applied to the training set.
    show_samples(dataset):
        Visualizes 5 sample images from each class (Pneumonia and Edema) in the provided dataset.
Constants:
    CLASS_A_IDX (int): Index for 'pneumonia' class in ChestMNIST labels (6).
    CLASS_B_IDX (int): Index for 'edema' class in ChestMNIST labels (11).
    NEW_CLASS_NAMES (dict): Mapping of new class indices to class names.
ChestMNIST Class Index Reference:
    # 0: 'Atelectasis'
    # 1: 'Cardiomegaly'
    # 2: 'Consolidation'
    # 3: 'Edema'
    # 4: 'Effusion'
    # 5: 'Emphysema'
    # 6: 'Fibrosis'
    # 7: 'Hernia'
    # 8: 'Infiltration'
    # 9: 'Mass'
    # 10: 'Nodule'
    # 11: 'Pneumonia'
    # 12: 'Pleural_Thickening'
    # 13: 'Pneumothorax'
    # 14: 'No Finding'
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from medmnist import ChestMNIST



# --- Konfigurasi Kelas Biner ---
CLASS_A_IDX = 9  # 'mass'
CLASS_B_IDX = 10 # 'nodule'

NEW_CLASS_NAMES = {0: 'Pneumonia', 1: 'Edema'}

class FilteredBinaryDataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        
        # Muat dataset lengkap
        full_dataset = ChestMNIST(split=split, transform=None, download=True)
        original_labels = full_dataset.labels

        # Cari indeks untuk gambar yang HANYA memiliki satu label yang kita inginkan
        indices_a = np.where((original_labels[:, CLASS_A_IDX] == 1) & (original_labels.sum(axis=1) == 1))[0]
        indices_b = np.where((original_labels[:, CLASS_B_IDX] == 1) & (original_labels.sum(axis=1) == 1))[0]

        # Simpan gambar dan label yang sudah dipetakan ulang
        self.images = []
        self.labels = []

        # Tambahkan data untuk kelas Pneumonia (dipetakan ke label 0)
        for idx in indices_a:
            self.images.append(full_dataset[idx][0])
            self.labels.append(0)

        # Tambahkan data untuk kelas Fibrosis (dipetakan ke label 1)
        for idx in indices_b:
            self.images.append(full_dataset[idx][0])
            self.labels.append(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor([label])

def get_data_loaders(batch_size):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # 1. Balik Horizontal dengan probabilitas 50%
        transforms.RandomHorizontalFlip(p=0.5),

        # 2. Rotasi acak dengan sudut kecil
        transforms.RandomRotation(degrees=10),
        
        # 3. Translasi (pergeseran) acak sebesar 10% dari ukuran gambar
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        
        # 4. (Opsional) Perubahan kecerahan dan kontras yang halus
        # Gunakan nilai kecil untuk brightness dan contrast
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])

    train_dataset = FilteredBinaryDataset('train', data_transform)
    val_dataset = FilteredBinaryDataset('test', data_transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    n_classes = 2
    n_channels = 1
    
    print("Dataset ChestMNIST berhasil difilter untuk klasifikasi biner!")
    print(f"Kelas yang digunakan: {NEW_CLASS_NAMES[0]} (Label 0) dan {NEW_CLASS_NAMES[1]} (Label 1)")
    print(f"Jumlah data training: {len(train_dataset)}")
    print(f"Jumlah data validasi: {len(val_dataset)}")
    
    return train_loader, val_loader, n_classes, n_channels

def show_samples(dataset):
    pneumonia_imgs = []
    fibrosis_imgs = []
    
    for img, label in dataset:
        if label.item() == 0 and len(pneumonia_imgs) < 5:
            pneumonia_imgs.append(img)
        elif label.item() == 1 and len(fibrosis_imgs) < 5:
            fibrosis_imgs.append(img)
        
        if len(pneumonia_imgs) == 5 and len(fibrosis_imgs) == 5:
            break
            
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Perbandingan Gambar: Pneumonia (atas) vs Fibrosis (bawah)", fontsize=16)
    
    for i, img in enumerate(pneumonia_imgs):
        ax = axes[0, i]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f"Pneumonia #{i+1}")
        ax.axis('off')
        
    for i, img in enumerate(fibrosis_imgs):
        ax = axes[1, i]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f"Fibrosis #{i+1}")
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    print("Memuat dataset untuk plotting...")
    plot_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = FilteredBinaryDataset('train', transform=plot_transform)
    
    if len(train_dataset) > 0:
        print("\n--- Menampilkan 5 Contoh Gambar per Kelas ---")
        show_samples(train_dataset)
    else:
        print("Dataset tidak berisi sampel untuk kelas yang dipilih.")