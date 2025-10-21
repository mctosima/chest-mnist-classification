# model.py

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Residual Block untuk ResNet"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # Skip connection
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """ResNet architecture untuk klasifikasi gambar"""
    
    def __init__(self, block, layers, in_channels=1, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling dan classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 
                           1 if num_classes == 2 else num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def resnet18(in_channels=1, num_classes=10):
    """ResNet-18 model"""
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels, num_classes)


def resnet34(in_channels=1, num_classes=10):
    """ResNet-34 model"""
    return ResNet(BasicBlock, [3, 4, 6, 3], in_channels, num_classes)


# Backward compatibility - gunakan ResNet18 sebagai default
class SimpleCNN(nn.Module):
    """Wrapper untuk backward compatibility - sekarang menggunakan ResNet18"""
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.model = resnet18(in_channels, num_classes)
    
    def forward(self, x):
        return self.model(x)

# --- Bagian untuk pengujian ---
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    
    print("=" * 60)
    print("--- Menguji Model ResNet ---")
    print("=" * 60)
    
    # Test ResNet18
    print("\n1. Testing ResNet-18:")
    model18 = resnet18(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    dummy_input = torch.randn(4, IN_CHANNELS, 224, 224)
    output18 = model18(dummy_input)
    print(f"   Input size: {dummy_input.shape}")
    print(f"   Output size: {output18.shape}")
    print(f"   Total parameters: {sum(p.numel() for p in model18.parameters()):,}")
    
    # Test ResNet34
    print("\n2. Testing ResNet-34:")
    model34 = resnet34(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    output34 = model34(dummy_input)
    print(f"   Input size: {dummy_input.shape}")
    print(f"   Output size: {output34.shape}")
    print(f"   Total parameters: {sum(p.numel() for p in model34.parameters()):,}")
    
    # Test backward compatibility dengan SimpleCNN
    print("\n3. Testing SimpleCNN (ResNet-18 wrapper):")
    model_simple = SimpleCNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    output_simple = model_simple(dummy_input)
    print(f"   Input size: {dummy_input.shape}")
    print(f"   Output size: {output_simple.shape}")
    print(f"   Total parameters: {sum(p.numel() for p in model_simple.parameters()):,}")
    
    print("\n" + "=" * 60)
    print("✓ Semua pengujian model ResNet berhasil!")
    print("=" * 60)