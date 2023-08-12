from segmentation_models_pytorch.encoders import get_encoder
from torch import nn


class Encoder(nn.Module):
    def __init__(self, encoder_name, in_channels, depth, weights,
                 bottleneck_shape=(1, 16, 16),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._encoder = get_encoder(encoder_name, in_channels=in_channels, depth=depth, weights=weights)

        #self._bottleneck = nn.Sequential(
        #    nn.Conv2d(self._encoder.out_channels[-1], self._encoder.out_channels[-1], kernel_size=3, padding="same"),
        #    nn.BatchNorm2d(self._encoder.out_channels[-1]),
        #    nn.LeakyReLU(inplace=True),
        #)

        self._out_block = nn.Sequential(
            nn.Conv2d(self._encoder.out_channels[-1], bottleneck_shape[0], kernel_size=1),
            nn.BatchNorm2d(bottleneck_shape[0]),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(bottleneck_shape[1:]),
            #nn.Flatten(),
            #nn.Linear(bottleneck_shape[0] * bottleneck_shape[1] * bottleneck_shape[2],
            #          bottleneck_shape[0] * bottleneck_shape[1] * bottleneck_shape[2]),
            #nn.Tanh(),
        )

        self.bottleneck_shape = bottleneck_shape



    def forward(self, x):
        features = self._encoder(x)
        x = features[-1]
        x = self._out_block(x)
        x = x.view(x.size(0), *self.bottleneck_shape)
        return x

    @property
    def output_stride(self):
        return self._encoder.output_stride
