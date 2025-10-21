# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import SimpleCNN  # CNN sederhana lama
from model_resnet import resnet18, resnet34  # ResNet models
from model_shufflenet import shufflenet_v1, shufflenet_v1_small  # ShuffleNet models
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameter ---
EPOCHS = 16
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# --- Model Selection ---
# Pilih model yang mau dipake: 
# 'simple_cnn', 'resnet18', 'resnet34', 'shufflenet', 'shufflenet_small'
MODEL_TYPE = 'shufflenet'  # <<< UBAH DI SINI UNTUK GANTI MODEL

#Menampilkan plot riwayat training dan validasi setelah training selesai.

def train():
    # 1. Memuat Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Inisialisasi Model berdasarkan pilihan
    print(f"\n{'='*60}")
    print(f"Model yang dipilih: {MODEL_TYPE.upper()}")
    print(f"{'='*60}\n")
    
    if MODEL_TYPE == 'simple_cnn':
        model = SimpleCNN(in_channels=in_channels, num_classes=num_classes)
        print("✓ Menggunakan SimpleCNN (CNN sederhana)")
    elif MODEL_TYPE == 'resnet18':
        model = resnet18(in_channels=in_channels, num_classes=num_classes)
        print("✓ Menggunakan ResNet-18")
    elif MODEL_TYPE == 'resnet34':
        model = resnet34(in_channels=in_channels, num_classes=num_classes)
        print("✓ Menggunakan ResNet-34")
    elif MODEL_TYPE == 'shufflenet':
        model = shufflenet_v1(num_classes=num_classes, groups=3, in_channels=in_channels)
        print("✓ Menggunakan ShuffleNet V1 (groups=3)")
    elif MODEL_TYPE == 'shufflenet_small':
        model = shufflenet_v1_small(num_classes=num_classes, groups=3, in_channels=in_channels)
        print("✓ Menggunakan ShuffleNet V1 Small (groups=3)")
    else:
        raise ValueError(f"Model type '{MODEL_TYPE}' tidak valid! Pilih: 'simple_cnn', 'resnet18', 'resnet34', 'shufflenet', atau 'shufflenet_small'")
    
    # Hitung total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"\nArsitektur Model:")
    print(model)
    
    # 3. Mendefinisikan Loss Function dan Optimizer
    # Gunakan BCEWithLogitsLoss untuk klasifikasi biner. Ini lebih stabil secara numerik.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Inisialisasi list untuk menyimpan history
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    
    print("\n--- Memulai Training ---")
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images
            # Ubah tipe data label menjadi float untuk BCEWithLogitsLoss
            labels = labels.float()
            
            outputs = model(images)
            loss = criterion(outputs, labels) # Loss dihitung antara output tunggal dan label
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Hitung training accuracy
            predicted = (outputs > 0).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # --- Fase Validasi ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images
                labels = labels.float()
                
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
                
                predicted = (outputs > 0).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Simpan history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    print("--- Training Selesai ---")
    
    # Tampilkan plot
    plot_training_history(train_losses_history, val_losses_history, 
                         train_accs_history, val_accs_history)

    # Visualisasi prediksi pada 10 gambar random dari validation set
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)

if __name__ == '__main__':
    train()
    