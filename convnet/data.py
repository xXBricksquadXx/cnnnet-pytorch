from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


@dataclass
class DataConfig:
    data_dir: str = "data"
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 0
    demo_aug: bool = True
    grayscale: bool = False  # converts on load; does not require grayscale files


def _build_transforms(image_size: int, demo_aug: bool, grayscale: bool):
    t = []

    # Optional grayscale conversion:
    # - num_output_channels=1 yields 1-channel tensors (C=1)
    # - num_output_channels=3 yields 3-channel tensors but grayscale-looking
    if grayscale:
        t.append(transforms.Grayscale(num_output_channels=1))

    if demo_aug:
        t.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        )
    else:
        t.append(transforms.Resize((image_size, image_size)))

    t.append(transforms.ToTensor())

    # Simple normalization; keep consistent between train/predict
    if grayscale:
        t.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    else:
        t.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    return transforms.Compose(t)


def build_dataloaders(cfg: DataConfig):
    train_dir = os.path.join(cfg.data_dir, "train")
    val_dir = os.path.join(cfg.data_dir, "val")

    train_tf = _build_transforms(cfg.image_size, demo_aug=cfg.demo_aug, grayscale=cfg.grayscale)
    val_tf = _build_transforms(cfg.image_size, demo_aug=False, grayscale=cfg.grayscale)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    in_channels = 1 if cfg.grayscale else 3
    return train_loader, val_loader, train_ds.class_to_idx, in_channels
