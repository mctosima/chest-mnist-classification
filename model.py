# model.py

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Arsitektur CNN yang lebih baik dengan beberapa perbaikan:
    1.  Lebih Dalam (Deeper): Tiga lapisan konvolusi untuk menangkap fitur yang lebih kompleks.
    2.  Batch Normalization: Menstabilkan dan mempercepat proses training.
    3.  Dropout: Mengurangi overfitting dengan "mematikan" beberapa neuron secara acak saat training.
    4.  Penggunaan nn.Sequential: Membuat kode lebih rapi dan terstruktur.
    """
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()
        
        # Blok Konvolusi 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Blok Konvolusi 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Blok Konvolusi 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Blok Konvolusi 4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier (Fully Connected Layers)
        # Ukuran input gambar: 28x28
        # Setelah Pool 1: 14x14
        # Setelah Pool 2: 7x7
        # Setelah Pool 3: 3x3 (karena 7 // 2 = 3)
        # Ukuran input flattened: 64 (channels) * 3 * 3 = 576
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.2), # Regularisasi untuk mencegah overfitting
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2), # Regularisasi untuk mencegah overfitting
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Regularisasi untuk mencegah overfitting
            nn.Linear(64, 1 if num_classes == 2 else num_classes)
        )

    def forward(self, x):
        """Mendefinisikan alur data (forward pass)."""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x

# --- Bagian untuk pengujian ---
# Ganti nama model yang diuji dari SimpleCNN menjadi BetterCNN
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    
    print("--- Menguji Model 'SimpleCNN' ---")
    
    model = SimpleCNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    print("Arsitektur Model:")
    print(model)
    
    dummy_input = torch.randn(64, IN_CHANNELS, 28, 28) # Uji dengan batch size 64
    output = model(dummy_input)
    
    print(f"\nUkuran input: {dummy_input.shape}")
    print(f"Ukuran output: {output.shape}") # Harusnya [64, 1]
    print("Pengujian model 'SimpleCNN' berhasil.")