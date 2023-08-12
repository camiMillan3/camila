import os

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, ColorJitter
from torchvision.transforms.v2 import RandomAffine

from models.data_encoder import min_max_scale


class ObservationDataset(Dataset):

    # data under path is organized as follows:
    # Y_<id>.npy
    # X_<id>.npy

    def __init__(self, path, x_transform=None,
                 y_transform=None, y_min=0, y_max=6):
        super().__init__()

        self.path = os.path.abspath(path)
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.y_min = y_min
        self.y_max = y_max

        self.data = []
        self.data_lookup = {}
        assert os.path.isdir(path), f"ObservationDataset: path={path} is not a directory"
        assert len(os.listdir(path)) > 0, f"ObservationDataset: path={path} is empty"

        for file in os.listdir(path):
            if file.startswith('Y_'):
                _id = file[2:-4]
                self.data.append((_id, os.path.join(path, file)))
                self.data_lookup[_id] = len(self.data) - 1

        print(f"ObservationDataset: path={path}, len={len(self.data)}")
        assert len(self.data) > 0, f"ObservationDataset: path={path} contains no data"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _id, _file = self.data[idx]
        y = np.load(_file)
        x = np.load(os.path.join(self.path, f'X_{_id}.npy'))

        y = np.expand_dims(y, axis=-1)
        # normalize to [0, 1]
        y = min_max_scale(y, self.y_min, self.y_max)

        if self.y_transform is not None:
            y = self.y_transform(y)
        if self.x_transform is not None:
            x0 = self.x_transform(x[..., 0])
            x1 = self.x_transform(x[..., 1])

            x = torch.cat([x0, x1], dim=0)

        return y, x, _id


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        device = tensor.device
        return tensor + torch.randn(tensor.size(), device=device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class TensorMaskedColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, except_values=(0,)):
        self.color_jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.except_values = except_values

    def __call__(self, tensor):
        # Apply the color jitter transformation to the PIL Image
        output_tensor = self.color_jitter(tensor)

        for except_value in self.except_values:
            # Create a mask of pixels that are zero in the original tensor
            mask = (tensor == except_value).all(axis=0)

            # Apply the mask to keep zero values unchanged
            output_tensor = mask * tensor + (1 - mask) * output_tensor

        return output_tensor


def get_y_train_transforms(image_size):
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((image_size, image_size)),
            RandomAffine(degrees=180, fill=0, interpolation=InterpolationMode.BILINEAR),
            # TensorMaskedColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            # AddGaussianNoise(0., 0.01),
        ]
    )


def get_y_test_transforms(image_size):
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((image_size, image_size)),
        ]
    )
