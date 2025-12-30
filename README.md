## ![Header](assets/header-banner.png)

# CNNNet (PyTorch) — conv training, prediction, saving/loading

A compact, practical reference for this chapter’s workflow:

- train a convolutional network (**CNNNet**) on images (ImageFolder)
- validate and compute accuracy
- run single-image predictions
- save and restore models:

  - **state_dict** (weights-only)
  - **checkpoint dict** (resume + inference)
  - **full-model demo** (brittle)

This repo is intentionally small so you can iterate on:

- convolution knobs (kernel size / stride / padding)
- pooling choices
- dropout strength
- optimizer + learning rate
- batch size
- input image size (affects feature map sizes + parameter count)

## ![Header](assets/header-banner.png)

# CNNNet (PyTorch) — conv training, prediction, saving/loading

A compact, practical reference for this chapter’s workflow:

- train a convolutional network (**CNNNet**) on images (ImageFolder)
- validate and compute accuracy
- run single-image predictions
- save and restore models:

  - **state_dict** (weights-only)
  - **checkpoint dict** (resume + inference)
  - **full-model demo** (brittle)

This repo is intentionally small so you can iterate on:

- convolution knobs (kernel size / stride / padding)
- pooling choices
- dropout strength
- optimizer + learning rate
- batch size
- input image size (affects feature map sizes + parameter count)

## ![Header](assets/header-banner.png)

# CNNNet (PyTorch) — conv training, prediction, saving/loading

A compact, practical reference for this chapter’s workflow:

- train a convolutional network (**CNNNet**) on images (ImageFolder)
- validate and compute accuracy
- run single-image predictions
- save and restore models:

  - **state_dict** (weights-only)
  - **checkpoint dict** (resume + inference)
  - **full-model demo** (brittle)

This repo is intentionally small so you can iterate on:

- convolution knobs (kernel size / stride / padding)
- pooling choices
- dropout strength
- optimizer + learning rate
- batch size
- input image size (affects feature map sizes + parameter count)

---

## Baseline demo (screen recording)

▶ **Baseline video:** [assets/cnnet-ptorch.mp4](assets/cnnet-ptorch.mp4)

[![Watch the video](https://img.shields.io/badge/▶_Watch-Baseline_Video-blue?style=for-the-badge)](https://github.com/user-attachments/assets/51a44a67-081d-430c-adef-fff21dd63f14)

<div align="center">
  <a href="github.com/user-attachments/assets/51a44a67-081d-430c-adef-fff21dd63f14f">
  </a>
</div>

Notes:

- GitHub may not autoplay MP4 inside the README; the link should open/download the file.

---

## Repo layout

```
cnnnet-pytorch/
  assets/                  # visuals only (optional)
  data/                    # training-only images
    train/
      cat/
      fish/
    val/
      cat/
      fish/
  runs/                    # outputs/checkpoints (gitignored)
  convnet/                 # package
    __init__.py
    data.py
    engine.py
    io.py
    models.py
    utils.py
  train.py                 # CLI entrypoint
  predict.py               # CLI entrypoint
  requirements.txt
  .gitignore
```

Notes:

- `data/` is for model training only.
- `assets/` is for banners/icons/screenshots/video so they never leak into training.

---

## 1) Setup

```bash
python -m venv .venv

# Windows (PowerShell)
./.venv/Scripts/Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Sanity:

```bash
python -c "import torch, torchvision; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

---

## 2) Dataset (ImageFolder)

This uses `torchvision.datasets.ImageFolder`.

Expected folder structure:

```
data/
  train/
    cat/*.png
    fish/*.png
  val/
    cat/*.png
    fish/*.png
```

Convention:

- `data/train/*` = **clean** examples
- `data/val/*` = **challenge** examples

---

## 3) Baseline run (CNNNet)

This CNNNet is AlexNet-ish (stride-4 early), so use a larger image size (recommended: **224**).

```bash
python train.py \
  --data-dir data \
  --device cpu \
  --epochs 25 \
  --batch-size 16 \
  --image-size 224 \
  --optimizer adam \
  --lr 0.001 \
  --demo-aug 1
```

Expected behavior (small data + domain shift):

- training accuracy can hit ~1.0 quickly
- validation accuracy may hover around chance if there’s a clean→challenge shift
- augmentation + dropout can reduce collapse/overconfidence

Outputs:

- `runs/convnet_state_dict.pt` (recommended weights-only)
- `runs/convnet_checkpoint.pt` (recommended for resume + inference)
- `runs/convnet_full_model.pt` (brittle demo; breaks if code structure changes)

---

## 4) Predict

Single file:

```bash
python predict.py \
  --image "data/val/cat/cat-challenge-001.png" \
  --checkpoint "runs/convnet_checkpoint.pt" \
  --device cpu
```

Directory input (picks the first image in the folder):

```bash
python predict.py \
  --image "data/val/cat" \
  --checkpoint "runs/convnet_checkpoint.pt" \
  --device cpu
```

---

## 5) Saving & loading (what to remember)

### A) Full model object (works, but brittle)

```py
torch.save(model, "runs/convnet_full_model.pt")
model = torch.load("runs/convnet_full_model.pt", map_location=device)
```

### B) Weights only (recommended)

```py
torch.save(model.state_dict(), "runs/convnet_state_dict.pt")

model = CNNNet(cfg)
model.load_state_dict(torch.load("runs/convnet_state_dict.pt", map_location=device))
```

### C) Checkpoint dict (recommended for real work)

Includes:

- model weights
- optimizer state
- epoch
- class_to_idx
- config metadata (image_size, normalization, etc.)

This is what `predict.py` uses.

---

## 6) Convolutions / kernels / pooling / dropout (mapped to code)

- **Conv2d**: `nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`

  - `kernel_size` = receptive field size (the “kernel” window)
  - `stride` controls downsampling
  - `padding` preserves spatial sizes (when set appropriately)

- **Pooling**: `nn.MaxPool2d(kernel_size, stride)`

  - reduces spatial resolution, adds some translation tolerance

- **Dropout**: `nn.Dropout(p)`

  - randomly zeros activations during training to reduce co-adaptation

### Grayscale note

You do **not** need grayscale images.

- If you have normal RGB images, keep `Conv2d(3, ...)` (default).
- If you want to demonstrate a 1-channel conv anyway, run with:

  - `--grayscale 1`

This converts RGB → grayscale **on load** (no special files required) and switches the first conv to `in_channels=1`.

---

## 7) Controlled experiments (one variable at a time)

Suggested sequence (keep everything else fixed when testing one change):

1. Augmentation: `--demo-aug 0/1`
2. Dropout: `--dropout 0.2` → `0.5` → `0.6`
3. Head size: `--head-dim 512` vs `1024` vs `4096`
4. Optimizer: Adam vs SGD (`--optimizer sgd --lr 0.01`)
5. Image size: 128 vs 224

---

## Findings (runs so far)

### Environment

- PyTorch: `2.9.1+cpu`
- CUDA available: `False` (CPU training)

### Dataset + split (current)

- train/cat: 12
- train/fish: 12
- val/cat: 8
- val/fish: 8

Note: validation has 16 total images, so:

- `val acc 0.562` = 9/16 correct
- `val acc 0.500` = 8/16 correct
- `val acc 0.438` = 7/16 correct
- `val acc 0.375` = 6/16 correct
- `val acc 0.312` = 5/16 correct
- `val acc 0.250` = 4/16 correct
- `val acc 0.188` = 3/16 correct

### Run 0 — baseline (very tiny set)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 4 --image-size 224 --optimizer adam --lr 0.001 --demo-aug 1
```

Observed:

- training accuracy reached ~1.0 quickly (memorization)
- validation accuracy bounced between ~0.50 and ~0.75 (too few samples to be stable)
- prediction on val examples collapsed to a single class (`fish`)

### Run 1 — Experiment A (shrink classifier head)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 16 --image-size 224 --optimizer adam --lr 0.001 --demo-aug 1 --head-dim 512
```

Observed:

- final epoch: train acc ~0.833, val acc ~0.500
- best val acc seen: ~0.75 (9/12 correct) at epoch 18 (earlier, smaller val)
- predictions on val folders still collapsed to `fish` with probability ~1.0

### Run 2 — Experiment B (lower LR + weight decay)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 16 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 1 --head-dim 512
```

Observed:

- final epoch: train acc ~1.0, val acc ~0.500
- best val acc seen: ~0.583 (7/12 correct) at epoch 20 (earlier, smaller val)
- predictions on val folders still heavily favored `fish` (≈0.9998–0.9999)

### Run 3 — Experiment B (batch size 4)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 4 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 1 --head-dim 512
```

Observed:

- early epochs hovered near chance (train/val ~0.50)
- train accuracy reached ~1.0 by ~epoch 14 (memorization)
- validation loss was volatile and spiked when the model became confidently wrong
- best val acc seen: ~0.667 (8/12 correct) at epochs 17 / 22 / 23 (earlier, smaller val)
- predictions on val folders still favored `fish` strongly (cat val example predicted fish at ~0.9988)

Sanity check (train distribution):

- `predict.py --image data/train/cat` → predicted **cat** (~0.8837)
- `predict.py --image data/train/fish` → predicted **fish** (~1.0000)

### Run 4 — New baseline on larger dataset (demo_aug ON)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 8 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 1 --head-dim 512
```

Observed:

- val accuracy peaked at **~0.562** (9/16 correct) around epochs 21–22
- single-example predictions (same checkpoint) correctly classified one cat and one fish sample

### Run 5 — Augmentation ablation (demo_aug OFF)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 8 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 0 --head-dim 512
```

Observed:

- train accuracy hit **1.0** quickly
- validation loss exploded over time (became extremely overconfident on the challenge split)
- predictions collapsed toward a single class on challenge

### Run 6 — Dropout sweep (demo_aug ON, dropout=0.6)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 8 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 1 --head-dim 512 --dropout 0.6
```

Observed:

- val loss stayed much more controlled than the no-augmentation run
- best val accuracy reached **~0.562** (9/16 correct) at epochs 18 and 22
- best val loss occurred around epoch 18 (~0.696), suggesting an **early-stop** window

### Run 7 — Weight decay sweep (demo_aug ON, weight_decay=0.001)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 8 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.001 --demo-aug 1 --head-dim 512
```

Observed:

- val accuracy intermittently reached **~0.562** (9/16 correct) but remained unstable
- val loss later spiked again, indicating overconfidence can still happen

### Run 8 — README demo run (dropout=0.6, epochs=18)

```bash
python train.py --data-dir data --device cpu --epochs 18 --batch-size 8 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 1 --head-dim 512 --dropout 0.6
```

Observed:

- epoch 18/18: **val loss 0.6961**, **val acc 0.562** (9/16 correct)
- predictions from the saved checkpoint (first image in each folder):

  - `data/val/cat` → **cat** (0.5513)
  - `data/val/fish` → **fish** (0.8891)
  - `data/train/cat` → **cat** (0.9842)
  - `data/train/fish` → **fish** (0.9483)

### Checkpoint sanity

`class_to_idx` from the saved checkpoint:

```py
{'cat': 0, 'fish': 1}
```

### Takeaway (so far)

- The training workflow and checkpointing are functioning.
- The clean→challenge split is the main bottleneck.
- Keeping `demo_aug=1` helps prevent collapse; adding **dropout (0.6)** stabilizes validation loss.
- With 12 clean/train per class and 8 challenge/val per class, best validation accuracy is still modest (~9/16).
- Next lever is **more challenge data** (and/or a smaller CNN), while keeping the training workflow constant.

---

## Further reading (chapter concepts)

- `Krizhevsky, Sutskever, Hinton (2012)`: _ImageNet Classification with Deep Convolutional Neural Networks_ (AlexNet)
- `Srivastava et al. (2014)`: _Dropout: A Simple Way to Prevent Neural Networks from Overfitting_
- `LeCun et al. (1998)`: _Gradient-Based Learning Applied to Document Recognition_ (early CNNs / LeNet)
- `Goodfellow, Bengio, Courville (2016)`: _Deep Learning_ (textbook reference on convs/pooling/regularization)

---

## Troubleshooting

VS Code shows missing imports but terminal runs:

- select the correct interpreter:

  - Ctrl+Shift+P → Python: Select Interpreter → choose `.venv/Scripts/python.exe`

If you get a shape error:

- this architecture expects larger images; try `--image-size 224` (or higher)

## Repo layout

```
cnnnet-pytorch/
  assets/                  # visuals only (optional)
  data/                    # training-only images
    train/
      cat/
      fish/
    val/
      cat/
      fish/
  runs/                    # outputs/checkpoints (gitignored)
  convnet/                 # package
    __init__.py
    data.py
    engine.py
    io.py
    models.py
    utils.py
  train.py                 # CLI entrypoint
  predict.py               # CLI entrypoint
  requirements.txt
  .gitignore
```

Notes:

- `data/` is for model training only.
- `assets/` is for banners/icons/screenshots/video so they never leak into training.

---

## 1) Setup

```bash
python -m venv .venv

# Windows (PowerShell)
./.venv/Scripts/Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Sanity:

```bash
python -c "import torch, torchvision; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

---

## 2) Dataset (ImageFolder)

This uses `torchvision.datasets.ImageFolder`.

Expected folder structure:

```
data/
  train/
    cat/*.png
    fish/*.png
  val/
    cat/*.png
    fish/*.png
```

Convention:

- `data/train/*` = **clean** examples
- `data/val/*` = **challenge** examples

---

## 3) Baseline run (CNNNet)

This CNNNet is AlexNet-ish (stride-4 early), so use a larger image size (recommended: **224**).

```bash
python train.py \
  --data-dir data \
  --device cpu \
  --epochs 25 \
  --batch-size 16 \
  --image-size 224 \
  --optimizer adam \
  --lr 0.001 \
  --demo-aug 1
```

Expected behavior (small data + domain shift):

- training accuracy can hit ~1.0 quickly
- validation accuracy may hover around chance if there’s a clean→challenge shift
- augmentation + dropout can reduce collapse/overconfidence

Outputs:

- `runs/convnet_state_dict.pt` (recommended weights-only)
- `runs/convnet_checkpoint.pt` (recommended for resume + inference)
- `runs/convnet_full_model.pt` (brittle demo; breaks if code structure changes)

---

## 4) Predict

Single file:

```bash
python predict.py \
  --image "data/val/cat/cat-challenge-001.png" \
  --checkpoint "runs/convnet_checkpoint.pt" \
  --device cpu
```

Directory input (picks the first image in the folder):

```bash
python predict.py \
  --image "data/val/cat" \
  --checkpoint "runs/convnet_checkpoint.pt" \
  --device cpu
```

---

## 5) Saving & loading (what to remember)

### A) Full model object (works, but brittle)

```py
torch.save(model, "runs/convnet_full_model.pt")
model = torch.load("runs/convnet_full_model.pt", map_location=device)
```

### B) Weights only (recommended)

```py
torch.save(model.state_dict(), "runs/convnet_state_dict.pt")

model = CNNNet(cfg)
model.load_state_dict(torch.load("runs/convnet_state_dict.pt", map_location=device))
```

### C) Checkpoint dict (recommended for real work)

Includes:

- model weights
- optimizer state
- epoch
- class_to_idx
- config metadata (image_size, normalization, etc.)

This is what `predict.py` uses.

---

## 6) Convolutions / kernels / pooling / dropout (mapped to code)

- **Conv2d**: `nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`

  - `kernel_size` = receptive field size (the “kernel” window)
  - `stride` controls downsampling
  - `padding` preserves spatial sizes (when set appropriately)

- **Pooling**: `nn.MaxPool2d(kernel_size, stride)`

  - reduces spatial resolution, adds some translation tolerance

- **Dropout**: `nn.Dropout(p)`

  - randomly zeros activations during training to reduce co-adaptation

### Grayscale note

You do **not** need grayscale images.

- If you have normal RGB images, keep `Conv2d(3, ...)` (default).
- If you want to demonstrate a 1-channel conv anyway, run with:

  - `--grayscale 1`

This converts RGB → grayscale **on load** (no special files required) and switches the first conv to `in_channels=1`.

---

## 7) Controlled experiments (one variable at a time)

Suggested sequence (keep everything else fixed when testing one change):

1. Augmentation: `--demo-aug 0/1`
2. Dropout: `--dropout 0.2` → `0.5` → `0.6`
3. Head size: `--head-dim 512` vs `1024` vs `4096`
4. Optimizer: Adam vs SGD (`--optimizer sgd --lr 0.01`)
5. Image size: 128 vs 224

---

## Findings (runs so far)

### Environment

- PyTorch: `2.9.1+cpu`
- CUDA available: `False` (CPU training)

### Dataset + split (current)

- train/cat: 12
- train/fish: 12
- val/cat: 8
- val/fish: 8

Note: validation has 16 total images, so:

- `val acc 0.562` = 9/16 correct
- `val acc 0.500` = 8/16 correct
- `val acc 0.438` = 7/16 correct
- `val acc 0.375` = 6/16 correct
- `val acc 0.312` = 5/16 correct
- `val acc 0.250` = 4/16 correct
- `val acc 0.188` = 3/16 correct

### Run 0 — baseline (very tiny set)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 4 --image-size 224 --optimizer adam --lr 0.001 --demo-aug 1
```

Observed:

- training accuracy reached ~1.0 quickly (memorization)
- validation accuracy bounced between ~0.50 and ~0.75 (too few samples to be stable)
- prediction on val examples collapsed to a single class (`fish`)

### Run 1 — Experiment A (shrink classifier head)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 16 --image-size 224 --optimizer adam --lr 0.001 --demo-aug 1 --head-dim 512
```

Observed:

- final epoch: train acc ~0.833, val acc ~0.500
- best val acc seen: ~0.75 (9/12 correct) at epoch 18 (earlier, smaller val)
- predictions on val folders still collapsed to `fish` with probability ~1.0

### Run 2 — Experiment B (lower LR + weight decay)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 16 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 1 --head-dim 512
```

Observed:

- final epoch: train acc ~1.0, val acc ~0.500
- best val acc seen: ~0.583 (7/12 correct) at epoch 20 (earlier, smaller val)
- predictions on val folders still heavily favored `fish` (≈0.9998–0.9999)

### Run 3 — Experiment B (batch size 4)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 4 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 1 --head-dim 512
```

Observed:

- early epochs hovered near chance (train/val ~0.50)
- train accuracy reached ~1.0 by ~epoch 14 (memorization)
- validation loss was volatile and spiked when the model became confidently wrong
- best val acc seen: ~0.667 (8/12 correct) at epochs 17 / 22 / 23 (earlier, smaller val)
- predictions on val folders still favored `fish` strongly (cat val example predicted fish at ~0.9988)

Sanity check (train distribution):

- `predict.py --image data/train/cat` → predicted **cat** (~0.8837)
- `predict.py --image data/train/fish` → predicted **fish** (~1.0000)

### Run 4 — New baseline on larger dataset (demo_aug ON)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 8 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 1 --head-dim 512
```

Observed:

- val accuracy peaked at **~0.562** (9/16 correct) around epochs 21–22
- single-example predictions (same checkpoint) correctly classified one cat and one fish sample

### Run 5 — Augmentation ablation (demo_aug OFF)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 8 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 0 --head-dim 512
```

Observed:

- train accuracy hit **1.0** quickly
- validation loss exploded over time (became extremely overconfident on the challenge split)
- predictions collapsed toward a single class on challenge

### Run 6 — Dropout sweep (demo_aug ON, dropout=0.6)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 8 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 1 --head-dim 512 --dropout 0.6
```

Observed:

- val loss stayed much more controlled than the no-augmentation run
- best val accuracy reached **~0.562** (9/16 correct) at epochs 18 and 22
- best val loss occurred around epoch 18 (~0.696), suggesting an **early-stop** window

### Run 7 — Weight decay sweep (demo_aug ON, weight_decay=0.001)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 8 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.001 --demo-aug 1 --head-dim 512
```

Observed:

- val accuracy intermittently reached **~0.562** (9/16 correct) but remained unstable
- val loss later spiked again, indicating overconfidence can still happen

### Run 8 — README demo run (dropout=0.6, epochs=18)

```bash
python train.py --data-dir data --device cpu --epochs 18 --batch-size 8 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 1 --head-dim 512 --dropout 0.6
```

Observed:

- epoch 18/18: **val loss 0.6961**, **val acc 0.562** (9/16 correct)
- predictions from the saved checkpoint (first image in each folder):

  - `data/val/cat` → **cat** (0.5513)
  - `data/val/fish` → **fish** (0.8891)
  - `data/train/cat` → **cat** (0.9842)
  - `data/train/fish` → **fish** (0.9483)

### Checkpoint sanity

`class_to_idx` from the saved checkpoint:

```py
{'cat': 0, 'fish': 1}
```

### Takeaway (so far)

- The training workflow and checkpointing are functioning.
- The clean→challenge split is the main bottleneck.
- Keeping `demo_aug=1` helps prevent collapse; adding **dropout (0.6)** stabilizes validation loss.
- With 12 clean/train per class and 8 challenge/val per class, best validation accuracy is still modest (~9/16).
- Next lever is **more challenge data** (and/or a smaller CNN), while keeping the training workflow constant.

---

## Further reading (chapter concepts)

- `Krizhevsky, Sutskever, Hinton (2012)`: _ImageNet Classification with Deep Convolutional Neural Networks_ (AlexNet)
- `Srivastava et al. (2014)`: _Dropout: A Simple Way to Prevent Neural Networks from Overfitting_
- `LeCun et al. (1998)`: _Gradient-Based Learning Applied to Document Recognition_ (early CNNs / LeNet)
- `Goodfellow, Bengio, Courville (2016)`: _Deep Learning_ (textbook reference on convs/pooling/regularization)

---

## Troubleshooting

VS Code shows missing imports but terminal runs:

- select the correct interpreter:

  - Ctrl+Shift+P → Python: Select Interpreter → choose `.venv/Scripts/python.exe`

If you get a shape error:

- this architecture expects larger images; try `--image-size 224` (or higher)

## Repo layout

```
cnnnet-pytorch/
  assets/                  # visuals only (optional)
  data/                    # training-only images
    train/
      cat/
      fish/
    val/
      cat/
      fish/
  runs/                    # outputs/checkpoints (gitignored)
  convnet/                 # package
    __init__.py
    data.py
    engine.py
    io.py
    models.py
    utils.py
  train.py                 # CLI entrypoint
  predict.py               # CLI entrypoint
  requirements.txt
  .gitignore
```

Notes:

- `data/` is for model training only.
- `assets/` is for banners/icons/screenshots/video so they never leak into training.

---

## 1) Setup

```bash
python -m venv .venv

# Windows (PowerShell)
./.venv/Scripts/Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Sanity:

```bash
python -c "import torch, torchvision; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

---

## 2) Dataset (ImageFolder)

This uses `torchvision.datasets.ImageFolder`.

Expected folder structure:

```
data/
  train/
    cat/*.png
    fish/*.png
  val/
    cat/*.png
    fish/*.png
```

Convention:

- `data/train/*` = **clean** examples
- `data/val/*` = **challenge** examples

---

## 3) Baseline run (CNNNet)

This CNNNet is AlexNet-ish (stride-4 early), so use a larger image size (recommended: **224**).

```bash
python train.py \
  --data-dir data \
  --device cpu \
  --epochs 25 \
  --batch-size 16 \
  --image-size 224 \
  --optimizer adam \
  --lr 0.001 \
  --demo-aug 1
```

Expected behavior (small data + domain shift):

- training accuracy can hit ~1.0 quickly
- validation accuracy may hover around chance if there’s a clean→challenge shift
- augmentation + dropout can reduce collapse/overconfidence

Outputs:

- `runs/convnet_state_dict.pt` (recommended weights-only)
- `runs/convnet_checkpoint.pt` (recommended for resume + inference)
- `runs/convnet_full_model.pt` (brittle demo; breaks if code structure changes)

---

## 4) Predict

Single file:

```bash
python predict.py \
  --image "data/val/cat/cat-challenge-001.png" \
  --checkpoint "runs/convnet_checkpoint.pt" \
  --device cpu
```

Directory input (picks the first image in the folder):

```bash
python predict.py \
  --image "data/val/cat" \
  --checkpoint "runs/convnet_checkpoint.pt" \
  --device cpu
```

---

## 5) Saving & loading (what to remember)

### A) Full model object (works, but brittle)

```py
torch.save(model, "runs/convnet_full_model.pt")
model = torch.load("runs/convnet_full_model.pt", map_location=device)
```

### B) Weights only (recommended)

```py
torch.save(model.state_dict(), "runs/convnet_state_dict.pt")

model = CNNNet(cfg)
model.load_state_dict(torch.load("runs/convnet_state_dict.pt", map_location=device))
```

### C) Checkpoint dict (recommended for real work)

Includes:

- model weights
- optimizer state
- epoch
- class_to_idx
- config metadata (image_size, normalization, etc.)

This is what `predict.py` uses.

---

## 6) Convolutions / kernels / pooling / dropout (mapped to code)

- **Conv2d**: `nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`

  - `kernel_size` = receptive field size (the “kernel” window)
  - `stride` controls downsampling
  - `padding` preserves spatial sizes (when set appropriately)

- **Pooling**: `nn.MaxPool2d(kernel_size, stride)`

  - reduces spatial resolution, adds some translation tolerance

- **Dropout**: `nn.Dropout(p)`

  - randomly zeros activations during training to reduce co-adaptation

### Grayscale note

You do **not** need grayscale images.

- If you have normal RGB images, keep `Conv2d(3, ...)` (default).
- If you want to demonstrate a 1-channel conv anyway, run with:

  - `--grayscale 1`

This converts RGB → grayscale **on load** (no special files required) and switches the first conv to `in_channels=1`.

---

## 7) Controlled experiments (one variable at a time)

Suggested sequence (keep everything else fixed when testing one change):

1. Augmentation: `--demo-aug 0/1`
2. Dropout: `--dropout 0.2` → `0.5` → `0.6`
3. Head size: `--head-dim 512` vs `1024` vs `4096`
4. Optimizer: Adam vs SGD (`--optimizer sgd --lr 0.01`)
5. Image size: 128 vs 224

---

## Findings (runs so far)

### Environment

- PyTorch: `2.9.1+cpu`
- CUDA available: `False` (CPU training)

### Dataset + split (current)

- train/cat: 12
- train/fish: 12
- val/cat: 8
- val/fish: 8

Note: validation has 16 total images, so:

- `val acc 0.562` = 9/16 correct
- `val acc 0.500` = 8/16 correct
- `val acc 0.438` = 7/16 correct
- `val acc 0.375` = 6/16 correct
- `val acc 0.312` = 5/16 correct
- `val acc 0.250` = 4/16 correct
- `val acc 0.188` = 3/16 correct

### Run 0 — baseline (very tiny set)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 4 --image-size 224 --optimizer adam --lr 0.001 --demo-aug 1
```

Observed:

- training accuracy reached ~1.0 quickly (memorization)
- validation accuracy bounced between ~0.50 and ~0.75 (too few samples to be stable)
- prediction on val examples collapsed to a single class (`fish`)

### Run 1 — Experiment A (shrink classifier head)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 16 --image-size 224 --optimizer adam --lr 0.001 --demo-aug 1 --head-dim 512
```

Observed:

- final epoch: train acc ~0.833, val acc ~0.500
- best val acc seen: ~0.75 (9/12 correct) at epoch 18 (earlier, smaller val)
- predictions on val folders still collapsed to `fish` with probability ~1.0

### Run 2 — Experiment B (lower LR + weight decay)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 16 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 1 --head-dim 512
```

Observed:

- final epoch: train acc ~1.0, val acc ~0.500
- best val acc seen: ~0.583 (7/12 correct) at epoch 20 (earlier, smaller val)
- predictions on val folders still heavily favored `fish` (≈0.9998–0.9999)

### Run 3 — Experiment B (batch size 4)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 4 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 1 --head-dim 512
```

Observed:

- early epochs hovered near chance (train/val ~0.50)
- train accuracy reached ~1.0 by ~epoch 14 (memorization)
- validation loss was volatile and spiked when the model became confidently wrong
- best val acc seen: ~0.667 (8/12 correct) at epochs 17 / 22 / 23 (earlier, smaller val)
- predictions on val folders still favored `fish` strongly (cat val example predicted fish at ~0.9988)

Sanity check (train distribution):

- `predict.py --image data/train/cat` → predicted **cat** (~0.8837)
- `predict.py --image data/train/fish` → predicted **fish** (~1.0000)

### Run 4 — New baseline on larger dataset (demo_aug ON)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 8 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 1 --head-dim 512
```

Observed:

- val accuracy peaked at **~0.562** (9/16 correct) around epochs 21–22
- single-example predictions (same checkpoint) correctly classified one cat and one fish sample

### Run 5 — Augmentation ablation (demo_aug OFF)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 8 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 0 --head-dim 512
```

Observed:

- train accuracy hit **1.0** quickly
- validation loss exploded over time (became extremely overconfident on the challenge split)
- predictions collapsed toward a single class on challenge

### Run 6 — Dropout sweep (demo_aug ON, dropout=0.6)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 8 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 1 --head-dim 512 --dropout 0.6
```

Observed:

- val loss stayed much more controlled than the no-augmentation run
- best val accuracy reached **~0.562** (9/16 correct) at epochs 18 and 22
- best val loss occurred around epoch 18 (~0.696), suggesting an **early-stop** window

### Run 7 — Weight decay sweep (demo_aug ON, weight_decay=0.001)

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 8 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.001 --demo-aug 1 --head-dim 512
```

Observed:

- val accuracy intermittently reached **~0.562** (9/16 correct) but remained unstable
- val loss later spiked again, indicating overconfidence can still happen

### Run 8 — README demo run (dropout=0.6, epochs=18)

```bash
python train.py --data-dir data --device cpu --epochs 18 --batch-size 8 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 1 --head-dim 512 --dropout 0.6
```

Observed:

- epoch 18/18: **val loss 0.6961**, **val acc 0.562** (9/16 correct)
- predictions from the saved checkpoint (first image in each folder):

  - `data/val/cat` → **cat** (0.5513)
  - `data/val/fish` → **fish** (0.8891)
  - `data/train/cat` → **cat** (0.9842)
  - `data/train/fish` → **fish** (0.9483)

### Checkpoint sanity

`class_to_idx` from the saved checkpoint:

```py
{'cat': 0, 'fish': 1}
```

### Takeaway (so far)

- The training workflow and checkpointing are functioning.
- The clean→challenge split is the main bottleneck.
- Keeping `demo_aug=1` helps prevent collapse; adding **dropout (0.6)** stabilizes validation loss.
- With 12 clean/train per class and 8 challenge/val per class, best validation accuracy is still modest (~9/16).
- Next lever is **more challenge data** (and/or a smaller CNN), while keeping the training workflow constant.

---

## Further reading (chapter concepts)

- `Krizhevsky, Sutskever, Hinton (2012)`: _ImageNet Classification with Deep Convolutional Neural Networks_ (AlexNet)
- `Srivastava et al. (2014)`: _Dropout: A Simple Way to Prevent Neural Networks from Overfitting_
- `LeCun et al. (1998)`: _Gradient-Based Learning Applied to Document Recognition_ (early CNNs / LeNet)
- `Goodfellow, Bengio, Courville (2016)`: _Deep Learning_ (textbook reference on convs/pooling/regularization)

---

## Troubleshooting

VS Code shows missing imports but terminal runs:

- select the correct interpreter:

  - Ctrl+Shift+P → Python: Select Interpreter → choose `.venv/Scripts/python.exe`

If you get a shape error:

- this architecture expects larger images; try `--image-size 224` (or higher)
