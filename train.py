# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from datareader import get_data_loaders
from model import SimpleCNN

# --- Konfigurasi Training ---
EPOCHS = 32
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

def train():
    # 1. Memuat Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Inisialisasi Model
    model = SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    
    # 3. Mendefinisikan Loss Function dan Optimizer
    # Gunakan BCEWithLogitsLoss untuk klasifikasi biner. Ini lebih stabil secara numerik.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Learning Rate Scheduler - OneCycleLR untuk pelatihan yang lebih efektif
    scheduler = OneCycleLR(optimizer, 
                          max_lr=LEARNING_RATE * 10,  # Maksimum LR 10x dari base LR
                          steps_per_epoch=len(train_loader),
                          epochs=EPOCHS,
                          pct_start=0.3)  # 30% pertama untuk warm-up
    
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
            
            # Step scheduler setelah setiap batch untuk OneCycleLR
            scheduler.step()
            
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
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Hapus scheduler.step() dari sini karena OneCycleLR di-step setelah setiap batch

    print("--- Training Selesai ---")

if __name__ == '__main__':
    train()