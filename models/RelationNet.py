import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from typing import Optional
from torch import Tensor

class ConvBlock(nn.Module):
    """
    Optimized convolutional block with GPU support
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 padding: int = 0, pool: bool = True) -> None:
        super().__init__()
        if not all(isinstance(x, int) for x in [in_channels, out_channels, kernel_size, padding]):
            raise TypeError("Channel and kernel parameters must be integers")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channel dimensions must be positive")
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm = nn.GroupNorm(8, out_channels)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with shape validation
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Processed tensor of shape (batch_size, out_channels, height', width')
        """
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")
            
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        return self.pool(x)

class EmbeddingNet(nn.Module):
    """
    CNN for feature extraction with GPU optimization
    """
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, 64, padding=0, pool=True),
            ConvBlock(64, 64, padding=0, pool=True),
            ConvBlock(64, 64, padding=1, pool=True),
            ConvBlock(64, 64, padding=1, pool=False)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with input validation
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        Returns:
            Embedded features
        """
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 input channels, got {x.shape[1]}")
        
        return self.encoder(x)

class RelationModule(nn.Module):
    """
    Relation computation module with GPU optimization
    """
    def __init__(self, input_size: int = 256, hidden_size: int = 8) -> None:
        super().__init__()
        if not isinstance(input_size, int) or not isinstance(hidden_size, int):
            raise TypeError("Size parameters must be integers")
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("Size parameters must be positive")

        self.conv_net = nn.Sequential(
            ConvBlock(128, 64, padding=1, pool=True),
            ConvBlock(64, 64, padding=1, pool=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with dimension validation
        Args:
            x: Input tensor of shape (batch_size, 128, height, width)
        Returns:
            Relation scores
        """
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.shape[1] != 128:
            raise ValueError(f"Expected 128 input channels, got {x.shape[1]}")

        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)