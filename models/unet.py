from typing import Optional, List, Union
from segmentation_models_pytorch.base import initialization
from torch import nn

from models.decoder import Decoder, ReconstructionHead
from models.encoder import Encoder


# We need a AutoEncoder with bottleneck. For the Encoder, we can use a pretrained model.

class Unet(nn.Module):
    """Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**


    Returns:
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        activation: Optional[Union[str, callable]] = None,
        bottleneck_channels: int = 128,
    ):
        super().__init__()

        self.encoder = Encoder(
            encoder_name=encoder_name,
            in_channels=1,
            depth=encoder_depth,
            weights=encoder_weights,
            bottleneck_channels=bottleneck_channels,
        )

        self.decoder = Decoder(
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            bottleneck_channels=bottleneck_channels,
        )

        self.reconstruction_head = ReconstructionHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            activation=activation,
            kernel_size=3,
        )

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        initialization.initialize_decoder(self.decoder)
        initialization.initialize_head(self.reconstruction_head)

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        x = self.encoder(x)

        x = self.decoder(x)

        x = self.reconstruction_head(x)

        return x

