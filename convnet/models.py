from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
import torch.nn as nn


@dataclass
class CNNNetConfig:
    """
    AlexNet-ish CNN with:
    - Conv2d stacks
    - MaxPool2d
    - Dropout in classifier

    Notes:
    - For RGB images, in_channels=3.
    - If you enable grayscale transforms, set in_channels=1.
    """
    num_classes: int = 2
    in_channels: int = 3
    dropout: float = 0.5
    # AdaptiveAvgPool2d output; AlexNet uses 6x6 with 224 input
    pool_out: int = 6
    # Keep the chapter-style big head, but allow shrinking for tiny datasets
    head_dim: int = 4096


class CNNNet(nn.Module):
    def __init__(self, cfg: CNNNetConfig):
        super().__init__()
        self.cfg = cfg

        self.features = nn.Sequential(
            nn.Conv2d(cfg.in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((cfg.pool_out, cfg.pool_out))

        flat_dim = 256 * cfg.pool_out * cfg.pool_out
        self.classifier = nn.Sequential(
            nn.Dropout(p=cfg.dropout),
            nn.Linear(flat_dim, cfg.head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(cfg.head_dim, cfg.head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.head_dim, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(model_name: str, model_cfg: Dict[str, Any]) -> nn.Module:
    """
    Model factory. Training loop is model-agnostic:
    it only requires an nn.Module that returns logits.
    """
    name = model_name.lower().strip()
    if name == "cnnnet":
        cfg = CNNNetConfig(**model_cfg)
        return CNNNet(cfg)
    raise ValueError(f"Unknown model_name={model_name!r}")


def default_model_cfg(num_classes: int, in_channels: int) -> Dict[str, Any]:
    return asdict(CNNNetConfig(num_classes=num_classes, in_channels=in_channels))
