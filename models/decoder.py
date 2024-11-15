from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.base.modules import Activation
from torch import nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ReconstructionHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same")
        activation = Activation(activation)
        super().__init__(conv2d, activation)


class Decoder(nn.Module):
    def __init__(
            self,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            bottleneck_shape=(1, 16, 16),
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # computing blocks input and output channels
        head_channels = bottleneck_shape[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        blocks = [
            DecoderBlock(in_ch, out_ch, use_batchnorm=use_batchnorm)
            for in_ch, out_ch in zip(in_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.bottleneck_shape = bottleneck_shape
        self.in_conv = nn.Sequential(
            nn.Conv2d(bottleneck_shape[0], bottleneck_shape[0], kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, output_size=(128, 128)):
        n_upsample_blocks = len(self.blocks)
        h, w = output_size
        x = F.interpolate(x, size=(h // 2 ** n_upsample_blocks, w // 2 ** n_upsample_blocks), mode="nearest")
        skip = x
        x = self.in_conv(x)
        x = x + skip

        for i, decoder_block in enumerate(self.blocks):
            x = decoder_block(x)

        return x
