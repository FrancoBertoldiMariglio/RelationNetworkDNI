import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, momentum=1, affine=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class EmbeddingNet(nn.Module):
    """Improved CNN for feature extraction"""

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(3, 64, padding=0, pool=True),  # Output: 64 x 41 x 41
            ConvBlock(64, 64, padding=0, pool=True),  # Output: 64 x 20 x 20
            ConvBlock(64, 64, padding=1, pool=True),  # Output: 64 x 10 x 10
            ConvBlock(64, 64, padding=1, pool=False)  # Output: 64 x 10 x 10
        )

    def forward(self, x):
        return self.encoder(x)


class RelationModule(nn.Module):
    """Improved relation computation module"""

    def __init__(self, input_size=256, hidden_size=8):
        super().__init__()

        self.conv_net = nn.Sequential(
            ConvBlock(128, 64, padding=1, pool=True),
            ConvBlock(64, 64, padding=1, pool=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.flatten(x, 1)  # Flatten features
        return self.classifier(x)