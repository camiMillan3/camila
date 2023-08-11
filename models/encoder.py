from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.encoders._base import EncoderMixin
from torch import nn


class Encoder(nn.Module):
    def __init__(self, encoder_name, in_channels, depth, weights,
                 bottleneck_shape=(1, 16, 16),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._encoder = get_encoder(encoder_name, in_channels=in_channels, depth=depth, weights=weights)

        self._bottleneck = nn.Sequential(
            nn.Conv2d(self._encoder.out_channels[-1], bottleneck_shape[0], kernel_size=1),
            nn.BatchNorm2d(bottleneck_shape[0]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(bottleneck_shape[1:]),
        )

    def forward(self, x):
        features = self._encoder(x)
        x = features[-1]
        x = self._bottleneck(x)
        return x

    @property
    def output_stride(self):
        return self._encoder.output_stride
