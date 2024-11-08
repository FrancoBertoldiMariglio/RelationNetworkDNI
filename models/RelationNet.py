import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class ConvBlock(nn.Module):
    """Optimized convolutional block with memory-efficient operations"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, pool=True):
        super().__init__()
        # Use memory-efficient group normalization instead of batch normalization
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding,
                              bias=False)  # Remove bias when using normalization
        self.norm = nn.GroupNorm(8, out_channels)  # More memory efficient than BatchNorm
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        # Fused operations where possible
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)  # inplace ReLU saves memory
        x = self.pool(x)
        return x


class EmbeddingNet(nn.Module):
    """Memory-efficient CNN for feature extraction"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, 64, padding=0, pool=True),
            ConvBlock(64, 64, padding=0, pool=True),
            ConvBlock(64, 64, padding=1, pool=True),
            ConvBlock(64, 64, padding=1, pool=False)
        )

    @torch.autocast('cuda')  # Enable automatic mixed precision
    def forward(self, x):
        return self.encoder(x)


class RelationModule(nn.Module):
    """Optimized relation computation module"""

    def __init__(self, input_size=256, hidden_size=8):
        super().__init__()

        self.conv_net = nn.Sequential(
            ConvBlock(128, 64, padding=1, pool=True),
            ConvBlock(64, 64, padding=1, pool=True)
        )

        # Memory-efficient linear layers
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),  # Remove bias when followed by normalization
            nn.LayerNorm(hidden_size),  # Add normalization for better training stability
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )

    @torch.autocast("cuda")
    def forward(self, x):
        # Add memory-efficient forward pass
        x = self.conv_net(x)
        # More efficient flattening for large batches
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.classifier(x)