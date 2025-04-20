import os
import torch
import numpy as np
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import struct

def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train')

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(256),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test')),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'hazy/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        try:
            # Try loading the images
            image = Image.open(os.path.join(self.image_dir, 'hazy', self.image_list[idx]))
            label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0] + '.png'))
            
            # Load image data to forcefully catch truncated or corrupted images
            image.load()
            label.load()

        except (OSError, struct.error) as e:
            # If the image is corrupted, skip it and return the next valid one
            print(f"Skipping corrupted image {self.image_list[idx]}: {e}")
            # You can handle this by recursively calling the next image
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)
        
        
        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)

        if self.is_test:
            name = self.image_list[idx]
            return image, label, name

        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError
