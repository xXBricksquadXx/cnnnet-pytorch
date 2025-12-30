from __future__ import annotations

import argparse
import os
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.optim import Adam, SGD

from convnet.data import DataConfig, build_dataloaders
from convnet.io import save_checkpoint, load_checkpoint
from convnet.models import build_model, default_model_cfg
from convnet.utils import AvgMeter, accuracy_top1, set_seed


def _build_optimizer(name: str, params, lr: float, momentum: float, weight_decay: float):
    name = name.lower().strip()
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def train_one_epoch(model: nn.Module, loader, optimizer, device: str, criterion):
    model.train()
    loss_m = AvgMeter()
    acc_m = AvgMeter()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_m.update(loss.item(), n=x.size(0))
        acc_m.update(accuracy_top1(logits, y), n=x.size(0))

    return loss_m.avg, acc_m.avg


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str, criterion):
    model.eval()
    loss_m = AvgMeter()
    acc_m = AvgMeter()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        loss_m.update(loss.item(), n=x.size(0))
        acc_m.update(accuracy_top1(logits, y), n=x.size(0))

    return loss_m.avg, acc_m.avg


def main() -> int:
    p = argparse.ArgumentParser(description="Train CNNNet on ImageFolder data/")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224, help="Recommended 224 for this CNNNet")
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--demo-aug", type=int, default=1, help="1 enables tiny-dataset augmentation")
    p.add_argument("--grayscale", type=int, default=0, help="1 converts images to 1-channel on load")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--head-dim", type=int, default=4096, help="Shrink for tiny datasets (e.g., 512 or 1024)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--runs-dir", default="runs")
    args = p.parse_args()

    set_seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to cpu.")
        device = "cpu"

    data_cfg = DataConfig(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        demo_aug=bool(args.demo_aug),
        grayscale=bool(args.grayscale),
    )
    train_loader, val_loader, class_to_idx, in_channels = build_dataloaders(data_cfg)
    num_classes = len(class_to_idx)

    model_name = "cnnnet"
    model_cfg: Dict[str, Any] = default_model_cfg(num_classes=num_classes, in_channels=in_channels)
    model_cfg["dropout"] = float(args.dropout)
    model_cfg["head_dim"] = int(args.head_dim)

    model = build_model(model_name, model_cfg).to(device)

    optimizer = _build_optimizer(
        args.optimizer, model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.runs_dir, exist_ok=True)

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, criterion)
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f}"
        )

        # Always save "latest" checkpoint
        ckpt = {
            "model_name": model_name,
            "model_cfg": model_cfg,
            "model_state": model.state_dict(),
            "optimizer_name": args.optimizer,
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "class_to_idx": class_to_idx,
            "data_cfg": {
                "image_size": args.image_size,
                "grayscale": bool(args.grayscale),
                "normalize_mean": [0.5] if bool(args.grayscale) else [0.5, 0.5, 0.5],
                "normalize_std": [0.5] if bool(args.grayscale) else [0.5, 0.5, 0.5],
            },
        }
        save_checkpoint(os.path.join(args.runs_dir, "convnet_checkpoint.pt"), ckpt)

        # Save weights-only too
        torch.save(model.state_dict(), os.path.join(args.runs_dir, "convnet_state_dict.pt"))

        # Save best snapshot
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            save_checkpoint(os.path.join(args.runs_dir, "convnet_best_checkpoint.pt"), ckpt)

    # Full-model demo (brittle)
    torch.save(model, os.path.join(args.runs_dir, "convnet_full_model.pt"))
    return 0


def predict_main() -> int:
    p = argparse.ArgumentParser(description="Predict single image using saved checkpoint")
    p.add_argument("--image", required=True, help="Path to image file OR directory (first image will be used)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to cpu.")
        device = "cpu"

    ckpt = load_checkpoint(args.checkpoint, device=device)

    model = build_model(ckpt["model_name"], ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Resolve image path
    img_path = args.image
    if os.path.isdir(img_path):
        exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        files = [f for f in os.listdir(img_path) if f.lower().endswith(exts)]
        if not files:
            raise SystemExit(f"No images found in directory: {img_path}")
        img_path = os.path.join(img_path, sorted(files)[0])

    from PIL import Image
    from torchvision import transforms

    data_cfg = ckpt.get("data_cfg", {})
    image_size = int(data_cfg.get("image_size", 224))
    grayscale = bool(data_cfg.get("grayscale", False))
    mean = data_cfg.get("normalize_mean", [0.5, 0.5, 0.5])
    std = data_cfg.get("normalize_std", [0.5, 0.5, 0.5])

    t = []
    if grayscale:
        t.append(transforms.Grayscale(num_output_channels=1))
    t.append(transforms.Resize((image_size, image_size)))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=mean, std=std))
    tf = transforms.Compose(t)

    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}
    pred_idx = int(probs.argmax().item())
    pred_class = idx_to_class[pred_idx]
    pred_prob = float(probs[pred_idx].item())

    print(f"image: {img_path}")
    print(f"pred:  {pred_class} ({pred_prob:.4f})")
    print("probs:")
    for i in range(len(probs)):
        print(f"  {idx_to_class[i]}: {float(probs[i].item()):.4f}")

    return 0
