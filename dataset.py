import os

import numpy as np
import torch
import torchvision
from einops import rearrange
from torch.utils.data import Dataset
from torchvision.transforms import RandomAffine


class ObservationDataset(Dataset):

    # data under path is organized as follows:
    # Y_<id>.npy
    # X_<id>.npy

    def __init__(self, path, x_transform=None,
                    y_transform=None):
        super().__init__()

        self.path = os.path.abspath(path)
        self.x_transform = x_transform
        self.y_transform = y_transform

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
        # x = rearrange(x, 'h w c -> c h w')

        y = np.expand_dims(y, axis=-1) / 6 # normalize to [0, 1]

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
        return tensor + torch.randn(tensor.size(),device=device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_y_train_transforms(image_size):
    return torchvision.transforms.Compose(
    [
        #RandomAffine(degrees=5, translate=(0.3, 0.3), scale=(0.7, 1.3), shear=5, fill=0),
        torchvision.transforms.Resize((image_size, image_size)),
        AddGaussianNoise(0., 0.01),
     ]
)

def get_y_test_transforms(image_size):
    return torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((image_size, image_size)),
     ]
)


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())