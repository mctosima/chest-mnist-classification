# model_shufflenet.py

import torch
import torch.nn as nn

class ShuffleBlock(nn.Module):
    """Channel Shuffle operation untuk ShuffleNet"""
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        """Channel shuffle: [N, C, H, W] -> [N, C, H, W]"""
        N, C, H, W = x.size()
        g = self.groups
        
        # Reshape: N, C, H, W -> N, g, C/g, H, W
        x = x.view(N, g, C // g, H, W)
        # Transpose: N, g, C/g, H, W -> N, C/g, g, H, W
        x = x.transpose(1, 2).contiguous()
        # Flatten: N, C/g, g, H, W -> N, C, H, W
        x = x.view(N, C, H, W)
        
        return x


class ShuffleUnit(nn.Module):
    """ShuffleNet Unit dengan depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, groups=3, stride=1):
        super(ShuffleUnit, self).__init__()
        self.stride = stride
        self.groups = groups
        
        mid_channels = out_channels // 4
        
        # Pointwise group convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, 
                               groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Channel shuffle
        self.shuffle = ShuffleBlock(groups)
        
        # Depthwise convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                               stride=stride, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # Pointwise group convolution
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, 
                               groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shortcut
        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.shuffle(out)
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        
        shortcut = self.shortcut(x)
        
        if self.stride == 2:
            # Concatenate untuk stride=2
            out = torch.cat([out, shortcut], 1)
        else:
            # Add untuk stride=1
            out = out + shortcut
        
        out = self.relu(out)
        return out


class ShuffleNet(nn.Module):
    """ShuffleNet architecture untuk klasifikasi gambar"""
    
    def __init__(self, num_blocks, num_classes=10, groups=3, in_channels=1):
        super(ShuffleNet, self).__init__()
        self.groups = groups
        
        # Stage 1: Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stage 2-4: ShuffleNet units
        # Output channels untuk setiap stage berdasarkan groups
        if groups == 1:
            out_channels = [144, 288, 576]
        elif groups == 2:
            out_channels = [200, 400, 800]
        elif groups == 3:
            out_channels = [240, 480, 960]
        elif groups == 4:
            out_channels = [272, 544, 1088]
        elif groups == 8:
            out_channels = [384, 768, 1536]
        else:
            raise ValueError(f"Groups {groups} tidak didukung. Pilih dari [1, 2, 3, 4, 8]")
        
        in_channels = 24
        
        # Stage 2
        self.stage2 = self._make_layer(in_channels, out_channels[0], 
                                       num_blocks[0], groups)
        in_channels = out_channels[0]
        
        # Stage 3
        self.stage3 = self._make_layer(in_channels, out_channels[1], 
                                       num_blocks[1], groups)
        in_channels = out_channels[1]
        
        # Stage 4
        self.stage4 = self._make_layer(in_channels, out_channels[2], 
                                       num_blocks[2], groups)
        
        # Global average pooling dan classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[2], 1 if num_classes == 2 else num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, groups):
        layers = []
        
        # First unit dengan stride=2
        layers.append(ShuffleUnit(in_channels, out_channels - in_channels, 
                                  groups=groups, stride=2))
        
        # Remaining units dengan stride=1
        for i in range(num_blocks - 1):
            layers.append(ShuffleUnit(out_channels, out_channels, 
                                     groups=groups, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def shufflenet_v1(num_classes=10, groups=3, in_channels=1):
    """
    ShuffleNet V1 model
    Args:
        num_classes: jumlah kelas output
        groups: jumlah groups untuk group convolution (1, 2, 3, 4, atau 8)
        in_channels: jumlah channel input (1 untuk grayscale, 3 untuk RGB)
    """
    return ShuffleNet([4, 8, 4], num_classes=num_classes, groups=groups, 
                     in_channels=in_channels)


def shufflenet_v1_small(num_classes=10, groups=3, in_channels=1):
    """
    ShuffleNet V1 Small - versi lebih kecil untuk dataset kecil
    Args:
        num_classes: jumlah kelas output
        groups: jumlah groups untuk group convolution (1, 2, 3, 4, atau 8)
        in_channels: jumlah channel input (1 untuk grayscale, 3 untuk RGB)
    """
    return ShuffleNet([2, 4, 2], num_classes=num_classes, groups=groups, 
                     in_channels=in_channels)


# --- Bagian untuk pengujian ---
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    
    print("=" * 70)
    print("--- Menguji Model ShuffleNet ---")
    print("=" * 70)
    
    # Test ShuffleNet V1
    print("\n1. Testing ShuffleNet V1 (groups=3):")
    model_v1 = shufflenet_v1(num_classes=NUM_CLASSES, groups=3, 
                             in_channels=IN_CHANNELS)
    dummy_input = torch.randn(4, IN_CHANNELS, 224, 224)
    output_v1 = model_v1(dummy_input)
    print(f"   Input size: {dummy_input.shape}")
    print(f"   Output size: {output_v1.shape}")
    print(f"   Total parameters: {sum(p.numel() for p in model_v1.parameters()):,}")
    
    # Test ShuffleNet V1 Small
    print("\n2. Testing ShuffleNet V1 Small (groups=3):")
    model_small = shufflenet_v1_small(num_classes=NUM_CLASSES, groups=3, 
                                      in_channels=IN_CHANNELS)
    output_small = model_small(dummy_input)
    print(f"   Input size: {dummy_input.shape}")
    print(f"   Output size: {output_small.shape}")
    print(f"   Total parameters: {sum(p.numel() for p in model_small.parameters()):,}")
    
    # Test dengan different groups
    print("\n3. Testing ShuffleNet V1 dengan groups=2:")
    model_g2 = shufflenet_v1(num_classes=NUM_CLASSES, groups=2, 
                            in_channels=IN_CHANNELS)
    output_g2 = model_g2(dummy_input)
    print(f"   Input size: {dummy_input.shape}")
    print(f"   Output size: {output_g2.shape}")
    print(f"   Total parameters: {sum(p.numel() for p in model_g2.parameters()):,}")
    
    print("\n" + "=" * 70)
    print("✓ Semua pengujian model ShuffleNet berhasil!")
    print("=" * 70)
    print("\nInfo: ShuffleNet menggunakan channel shuffle dan depthwise")
    print("      separable convolution untuk efisiensi komputasi yang tinggi!")
