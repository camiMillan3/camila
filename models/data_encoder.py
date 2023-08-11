import torch
from torch import nn


class DataEncoder(nn.Module):
    # input data is 16x16x2
    # output data is 16x16x1 (depends on the image size and the number of channels)
    def __init__(self, channels=(16, 32, 64), bottleneck_channels=1):
        super().__init__()
        blocks = []
        channels = [2] + list(channels)
        for i in range(len(channels) - 1):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.ReLU(),
                )
            )

        blocks.append(
            nn.Sequential(
                nn.Conv2d(channels[-1], bottleneck_channels, kernel_size=1),
                nn.BatchNorm2d(bottleneck_channels),
                nn.ReLU(),
            )
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x

