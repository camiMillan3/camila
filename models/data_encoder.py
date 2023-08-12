import torch
from torch import nn


def min_max_scale(x, min_value, max_value):
    return (x - min_value) / (max_value - min_value)


class RunningStatistics(nn.Module):
    def __init__(self, channels, min_value=0, max_value=5e4):
        super().__init__()
        self.register_buffer("running_max", torch.ones(1, channels, 16, 16) * max_value)
        self.register_buffer("running_min", torch.ones(1, channels, 16, 16) * min_value)

    def forward(self, x):
        if self.training:
            self.running_max.data = torch.max(self.running_max, x).max(dim=0, keepdim=True)[0]
            self.running_min.data = torch.min(self.running_min, x).min(dim=0, keepdim=True)[0]

        x = min_max_scale(x, self.running_min, self.running_max)

        return x


class SensorDataEncoderConv(nn.Module):
    # input data is 16x16x2
    # output data is 16x16x1 (depends on the image size and the number of channels)
    def __init__(self, channels=(16, 32, 64), bottleneck_channels=1, input_channels=2,
                 kernels=(3, 3, 3)):
        super().__init__()
        blocks = []
        skip_blocks = []

        self.in_block = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
        )

        for i in range(len(channels) - 1):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernels[i], padding="same"),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        for i in range(len(channels) - 1):
            skip_blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=1),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.out_block = nn.Sequential(
            nn.Conv2d(channels[-1], bottleneck_channels, kernel_size=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.Tanh(),
        )

        self.blocks = nn.Sequential(*blocks)
        self.skip_blocks = nn.Sequential(*skip_blocks)

        self.running_stats = RunningStatistics(input_channels)

    def forward(self, x):
        x = self.running_stats(x)

        x = self.in_block(x)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x) + self.skip_blocks[i](x)

        x = self.out_block(x)

        return x


class SensorDataEncoderDense(nn.Module):
    # input data is 16x16x2
    # output data is 16x16x1 (depends on the image size and the number of channels)
    def __init__(self, hidden_units=(512, 256, 128), bottleneck_units=256, input_units=16 * 16 * 2):
        super().__init__()

        layers = []
        units = [input_units] + list(hidden_units)

        for i in range(len(units) - 1):
            layers.extend([
                nn.Linear(units[i], units[i + 1]),
                nn.BatchNorm1d(units[i + 1]),
                nn.ReLU(),
            ])

        layers.append(nn.Linear(units[-1], bottleneck_units))
        layers.append(nn.BatchNorm1d(bottleneck_units))
        layers.append(nn.ReLU())

        self.fc_blocks = nn.Sequential(*layers)

        self.running_stats = RunningStatistics(input_units)

    def forward(self, x):
        # Flatten the input for dense processing
        x = x.view(x.size(0), -1)
        x = self.running_stats(x)
        x = self.fc_blocks(x)

        # Reshape the output to be spatial (for compatibility with the earlier design)
        x = x.view(x.size(0), 1, 16, 16)

        return x
