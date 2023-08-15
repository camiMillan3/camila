import torch
from torch import nn
import torch.nn.Functional as F 


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

class Interpolate(nn.Module)

    def __init__(self, size, mode):
        self.size = size
        self.mode = mode

    def forward(x)
        return F.interpolate(x, size=self.size, mode=self.mode)


class SensorDataEncoderConv(nn.Module):
    # input data is 2x16x16
    # output data is 1x16x16 (depends on the image size and the number of channels) or 3x8x8
    def __init__(self, channels=(16, 32, 64), bottleneck_shape=(1, 16, 16), input_channels=2,
                 kernels=(3, 3, 3)):
        super().__init__()
        blocks = []
        skip_blocks = []

        self.in_block = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=1),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(inplace=True)
        )

        for i in range(len(channels) - 1):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernels[i], padding="same"),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.LeakyReLU(inplace=True)
                )
            )

        for i in range(len(channels) - 1):
            skip_blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=1),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.LeakyReLU(inplace=True)
                )
            )

        if bottleneck_shape[1] == 16:
            self.downsample = nn.Identity()
        elif bottleneck_shape[1] == 8:
            self.downsample = nn.AvgPool2d(2)
        elif bottleneck_shape[1] == 4:
            self.downsample = nn.AvgPool2d(4)
        else:
            self.downsample = Interpolate(size=(bottleneck_shape[0], bottleneck_shape[1]), mode='nearest')


        self.out_block = nn.Sequential(
            nn.Conv2d(channels[-1], bottleneck_shape[0], kernel_size=1),
            nn.BatchNorm2d(bottleneck_shape[0]),
        )
        self.out_fc = nn.Linear(bottleneck_shape[0] * bottleneck_shape[1] * bottleneck_shape[2],
                                bottleneck_shape[0] * bottleneck_shape[1] * bottleneck_shape[2])

        self.out_act = nn.LeakyReLU(inplace=True)
        self.out_fc2 = nn.Linear(bottleneck_shape[0] * bottleneck_shape[1] * bottleneck_shape[2],
                                bottleneck_shape[0] * bottleneck_shape[1] * bottleneck_shape[2])
        self.out_act2 = nn.Sigmoid()

        self.blocks = nn.Sequential(*blocks)
        self.skip_blocks = nn.Sequential(*skip_blocks)
        self.bottleneck_shape = bottleneck_shape

        self.running_stats = RunningStatistics(input_channels)

    def forward(self, x):
        x = self.running_stats(x)

        x = self.in_block(x)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x) + self.skip_blocks[i](x)

        x = self.downsample(x)

        x = self.out_block(x)
        x = x.view(x.size(0), -1)

        x = self.out_fc(x)
        x = self.out_act(x)
        x = self.out_fc2(x)
        x = self.out_act2(x)
        x = x.view(x.size(0), *self.bottleneck_shape)

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
                nn.LeakyReLU(inplace=True)
            ])

        layers.append(nn.Linear(units[-1], bottleneck_units))
        layers.append(nn.BatchNorm1d(bottleneck_units))
        layers.append(nn.LeakyReLU(inplace=True))

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
