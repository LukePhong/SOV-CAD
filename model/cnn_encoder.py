import torch
import torch.nn as nn
import math

class CNNSketchEncoder(nn.Module):
    """CNN encoder for sketch images"""
    def __init__(self, img_height, img_width, in_channels=3, model_dim=256):
        super(CNNSketchEncoder, self).__init__()
        
        # Encoder network with reduced parameters
        self.encoder = nn.Sequential(
            # First block: 384x384x3 -> 96x96x16
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block: 96x96x16 -> 32x32x32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            
            # Third block: 32x32x32 -> 16x16x64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth block: 16x16x64 -> 8x8x96
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global pooling: 8x8x96 -> 1x1x96
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # Compact bottleneck: 96 -> 256
            nn.Linear(96, model_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Expect input in shape (B, H, W, C)
        # Convert to (B, C, H, W) for conv layers
        x = x.permute(0, 3, 1, 2)
        return self.encoder(x)

class CNNThreeViewEncoder(nn.Module):
    """CNN encoder for three-view images (front, side, top views)"""
    def __init__(self, img_height, img_width, in_channels=3, model_dim=256):
        super(CNNThreeViewEncoder, self).__init__()
        
        # Encoder network optimized for three-view images
        # Using GroupNorm instead of BatchNorm to avoid batch-dependent statistics
        self.encoder = nn.Sequential(
            # First block: 384x384x3 -> 96x96x32
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(4, 32),  # 4 groups for 32 channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block: 96x96x32 -> 32x32x64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),  # 8 groups for 64 channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            
            # Third block: 32x32x64 -> 16x16x128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 128),  # 16 groups for 128 channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth block: 16x16x128 -> 8x8x128
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 128),  # 16 groups for 128 channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global pooling: 8x8x128 -> 1x1x128
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # Projection to model dimension: 128 -> model_dim
            nn.Linear(128, model_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Expect input in shape (B, H, W, C)
        # Convert to (B, C, H, W) for conv layers
        x = x.permute(0, 3, 1, 2)
        return self.encoder(x) 