from __future__ import annotations

import os
from typing import Any, Dict

import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    torch.save(payload, path)


def load_checkpoint(path: str, device: str):
    return torch.load(path, map_location=device)
