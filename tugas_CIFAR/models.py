"""
Model Architectures: VGG and ResNet with different pooling strategies
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBase(nn.Module):
    """
    Base VGG class with customizable pooling
    """
    def __init__(self, num_classes=10, pooling_type='max'):
        super(VGGBase, self).__init__()
        self.pooling_type = pooling_type
        self.num_classes = num_classes
        
        # Determine pooling layer
        if pooling_type == 'max':
            pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pooling_type == 'avg':
            pool_layer = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
        
        # Feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            pool_layer,
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            pool_layer,
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            pool_layer,
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            pool_layer,
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNetBlock(nn.Module):
    """
    Residual block for ResNet
    """
    def __init__(self, in_channels, out_channels, stride=1, pooling_type='max'):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetBase(nn.Module):
    """
    ResNet with customizable pooling
    """
    def __init__(self, num_classes=10, pooling_type='max', depth=18):
        super(ResNetBase, self).__init__()
        self.pooling_type = pooling_type
        self.num_classes = num_classes
        
        # Determine pooling layer
        if pooling_type == 'max':
            pool_layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif pooling_type == 'avg':
            pool_layer = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = pool_layer
        
        # Residual blocks
        if depth == 18:
            num_blocks = [2, 2, 2, 2]
        elif depth == 34:
            num_blocks = [3, 4, 6, 3]
        else:
            num_blocks = [2, 2, 2, 2]  # Default to 18
        
        self.layer1 = self._make_layer(64, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride, self.pooling_type))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, stride=1, pooling_type=self.pooling_type))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def create_model(architecture='vgg', pooling_type='max', num_classes=10):
    """
    Factory function to create models
    
    Args:
        architecture: 'vgg' or 'resnet'
        pooling_type: 'max' or 'avg'
        num_classes: Number of classes
    
    Returns:
        Model instance
    """
    if architecture == 'vgg':
        return VGGBase(num_classes=num_classes, pooling_type=pooling_type)
    elif architecture == 'resnet':
        return ResNetBase(num_classes=num_classes, pooling_type=pooling_type)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


if __name__ == "__main__":
    # Test models
    import torch
    
    x = torch.randn(2, 3, 32, 32)
    
    print("Testing VGG with Max Pooling:")
    vgg_max = create_model('vgg', 'max')
    out = vgg_max(x)
    print(f"  Output shape: {out.shape}")
    
    print("\nTesting VGG with Avg Pooling:")
    vgg_avg = create_model('vgg', 'avg')
    out = vgg_avg(x)
    print(f"  Output shape: {out.shape}")
    
    print("\nTesting ResNet with Max Pooling:")
    resnet_max = create_model('resnet', 'max')
    out = resnet_max(x)
    print(f"  Output shape: {out.shape}")
    
    print("\nTesting ResNet with Avg Pooling:")
    resnet_avg = create_model('resnet', 'avg')
    out = resnet_avg(x)
    print(f"  Output shape: {out.shape}")
