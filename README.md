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
- `assets/` is for banners/icons/screenshots so they never leak into training.

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

Expected behavior (tiny data):

- training accuracy can hit ~1.0 quickly
- validation accuracy may hover around chance if there’s a domain shift
- augmentation + smaller heads can stabilize overfitting

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
2. Dropout: `--dropout 0.2` → `0.5`
3. Head size: `--head-dim 512` vs `1024` vs `4096`
4. Optimizer: Adam vs SGD (`--optimizer sgd --lr 0.01`)
5. Image size: 128 vs 224

---

## Findings (runs so far)

### Environment

- PyTorch: `2.9.1+cpu`
- CUDA available: `False` (CPU training)

### Dataset + split (current)

- train/cat: 6
- train/fish: 6
- val/cat: 6
- val/fish: 6

### Run 0 — baseline (very tiny set)

Command:

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 4 --image-size 224 --optimizer adam --lr 0.001 --demo-aug 1
```

Observed:

- training accuracy reached ~1.0 quickly (memorization)
- validation accuracy bounced between ~0.50 and ~0.75 (too few samples to be stable)
- prediction on val examples collapsed to a single class (`fish`)

### Run 1 — Experiment A (shrink classifier head)

Command:

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 16 --image-size 224 --optimizer adam --lr 0.001 --demo-aug 1 --head-dim 512
```

Observed:

- final epoch: train acc ~0.833, val acc ~0.500
- best val acc seen: ~0.75 (9/12 correct) at epoch 18
- predictions on val folders still collapsed to `fish` with probability ~1.0

### Run 2 — Experiment B (lower LR + weight decay)

Command:

```bash
python train.py --data-dir data --device cpu --epochs 25 --batch-size 16 --image-size 224 --optimizer adam --lr 0.0003 --weight-decay 0.0001 --demo-aug 1 --head-dim 512
```

Observed:

- final epoch: train acc ~1.0, val acc ~0.500
- best val acc seen: ~0.583 (7/12 correct) at epoch 20
- predictions on val folders still heavily favored `fish` (≈0.9998–0.9999)

### Takeaway (so far)

- The training workflow and checkpointing are functioning.
- With ~6 images/class and a clean→challenge shift, metrics are still noisy and the model can collapse to one class.
- Next lever is **more data** (and/or a smaller/simple CNN head), while keeping the workflow constant.

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
