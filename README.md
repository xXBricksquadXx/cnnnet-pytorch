## ![Header](assets/header-banner.png)

# CNNNet (PyTorch) — conv training, prediction, saving/loading

A compact, practical reference for this chapter’s workflow:

- train a convolutional network (**CNNNet**) on images (ImageFolder)
- validate and compute accuracy
- run single-image predictions
- save and restore models:

  - **state_dict** (weights-only)
  - **checkpoint dict** (resume + inference)
  - **best-checkpoint** selection (recommended)
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

▶ **Baseline video:** [assets/cnnet-ptorch.mp4](assets/final-test.mp4)

[![Watch the video](https://img.shields.io/badge/▶_Watch-Baseline_Video-blue?style=for-the-badge)](https://github.com/user-attachments/assets/326fe341-f867-4bdf-8f42-96ebb2cd37d1)

<div align="center">
  <a href="https://github.com/user-attachments/assets/326fe341-f867-4bdf-8f42-96ebb2cd37d1">
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
logs/                    # transcripts (optional)
runs/                    # outputs/checkpoints (gitignored)
convnet/                 # package
**init**.py
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
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('cuda:', torch.cuda.is_available())"
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
- `data/val/*` = **challenge** examples (harder / different distribution)

### Current dataset size (full upgrade)

- train/cat: **20**
- train/fish: **20**
- val/cat: **16**
- val/fish: **16**

Total validation images: **32**

---

## 3) Final run (CNNNet) — recommended config

This CNNNet is AlexNet-ish (stride-4 early), so use a larger image size (recommended: **224**).

```bash
python train.py \
  --data-dir data \
  --device cpu \
  --epochs 25 \
  --batch-size 8 \
  --image-size 224 \
  --optimizer adam \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --demo-aug 1 \
  --head-dim 512 \
  --dropout 0.6
```

### Outputs

- `runs/convnet_state_dict.pt` (recommended weights-only)
- `runs/convnet_checkpoint.pt` (last epoch checkpoint)
- `runs/convnet_best_checkpoint.pt` (**best** checkpoint on validation; recommended for inference)
- `runs/convnet_full_model.pt` (brittle demo; breaks if code structure changes)

---

## 4) Predict

Single file:

```bash
python predict.py \
  --image "data/val/cat/cat-challenge-001.png" \
  --checkpoint "runs/convnet_best_checkpoint.pt" \
  --device cpu
```

Directory input (picks the first image in the folder):

```bash
python predict.py \
  --image "data/val/cat" \
  --checkpoint "runs/convnet_best_checkpoint.pt" \
  --device cpu
```

---

## 5) Why BEST checkpoint matters

This run clearly shows **late-epoch drift / overfit** on the challenge split.

On the _same_ validation image (`data/val/cat/cat-challenge-001.png`):

- **Last epoch checkpoint** (`convnet_checkpoint.pt`, epoch 25) predicted **fish (0.7770)**
- **Best checkpoint** (`convnet_best_checkpoint.pt`, epoch 15) predicted **cat (0.5619)**

So for demos + inference, prefer:

- `runs/convnet_best_checkpoint.pt`

---

## 6) Final results (full dataset)

Environment:

- PyTorch: `2.9.1+cpu`
- torchvision: `0.24.1+cpu`
- CUDA available: `False` (CPU training)

Final training log highlights:

- **Peak validation accuracy:** **0.875** at **epoch 15** (28/32 correct)
- Next best: **0.844** at epoch 16 and epoch 20 (27/32 correct)

After epoch ~15–20, validation loss becomes volatile again (classic small-data behavior), even while train accuracy stays high.

### Random-sample sanity (best checkpoint)

A quick random check shows the model is usually confident on clean/train images, but still confuses a few hard challenge images.

Examples from the final run:

- `val/cat` sample mistakes included:

  - `cat-challenge-005.png` predicted fish (0.5104)
  - `cat-challenge-003.png` predicted fish (0.7814)

- `val/fish` sample mistake included:

  - `fish-challenge-004.png` predicted cat (0.8428)

Takeaway: the pipeline is correct; the remaining limitation is **domain shift + small data**;
`demo_aug=1` + `dropout=0.6` + **best checkpoint** made the run stable and usable.

---

## 7) Saving & loading (what to remember)

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

## 8) Convolutions / kernels / pooling / dropout (mapped to code)

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

## 9) Troubleshooting

VS Code shows missing imports but terminal runs:

- select the correct interpreter:

  - Ctrl+Shift+P → Python: Select Interpreter → choose `.venv/Scripts/python.exe`

If you get a shape error:

- this architecture expects larger images; try `--image-size 224` (or higher)

If you see:

- `pin_memory is set as true but no accelerator is found`

That’s expected on CPU-only PyTorch; it’s safe to ignore.

---

## Close-out analysis

- **full dataset upgrade** (20/20 train per class; 16/16 val per class) materially improved stability.
- The model hits its best generalization around **epoch 15** (best checkpoint saved at epoch 15). After that, the model keeps fitting the train set while **val behavior becomes noisy**.
- The “same image flips label” comparison (last vs best checkpoint) is the cleanest proof that:
  - the training loop works
  - the dataset is the bottleneck (domain shift)
  - the correct practice is **best-checkpoint + early stop**.
- Random predictions show only a handful of confusing “hard” images; that’s expected with small data and mixed-style images.

---
