# 🧠 CNN on CIFAR-10 — Hyperparameter Tuning (PyTorch)

This repository contains a simple **Convolutional Neural Network (CNN)** built in **PyTorch** for the **CIFAR-10** dataset.  
It is designed as a **teaching + learning repo** for experimenting with **hyperparameters** such as:

- Learning rate
- Optimizer (Adam / SGD)
- Dropout
- Epochs
- Batch size
- Device (CPU, CUDA GPU, Apple MPS)

The code is intentionally minimal and extendable so students can practice deeper experiments after exercises.

---

## 📂 Project Structure

.
├─ README.md
├─ requirements.txt
├─ Makefile
├─ src/
│ ├─ init.py
│ ├─ models.py # Defines SimpleCNN architecture
│ ├─ utils.py # Data loaders (CIFAR-10, transforms)
│ └─ train.py # Training loop + CLI hyperparameters
└─ tests/
└─ test_model_smoke.py

---

## ⚙️ Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/)
- Torchvision
- TQDM (progress bars)
- PyTest (for tests)

Install dependencies:
```bash
pip install -r requirements.txt
```
|⚠️ The CIFAR-10 dataset (~170MB) will be automatically downloaded into ./data on first run.

### 🚀 Quick Start

Train using default parameters (Adam, learning rate 0.001, dropout 0.5, batch size 64, epochs 2):

```python3 src.train.py```
Or via Makefile shortcut:

make train

### 🎛️ Hyperparameters (CLI Options)

```bash python3 src.train.py \
  --lr 0.0005 \
  --optimizer adam \        # {adam, sgd}
  --dropout 0.4 \
  --epochs 5 \
  --batch_size 128 \
  --device cpu              # e.g. "cuda", "mps", or "cpu"
```

### Examples
Adam, smaller LR, more epochs:

```python3  src.train.py --lr 5e-4 --optimizer adam --epochs 5```

SGD with momentum, higher LR, larger batch, lower dropout:

```python3 -m src.train.py --lr 0.01 --optimizer sgd --batch_size 128 --dropout 0.3```

Run on GPU (if available):

```python3 -m srctrain.py --device cuda```

Make targets
```bash
make train        # Adam, lr=0.001, epochs=1 (quick run)
make train-sgd    # SGD,  lr=0.01,  epochs=1
```

### 📊 Output

The script prints:

Epoch loss (training)

Validation accuracy after training

Example output:
- Epoch 1: Loss = 1.5623
- Epoch 2: Loss = 1.2381
- Validation Accuracy: 55.40%

### 🧪 Tests
There’s a smoke test for the CNN forward pass:

```PYTHONPATH=src pytest -v```

Expected output:

tests/test_model_smoke.py::test_forward_pass PASSED

### 🛠️ Implementation Notes

Model: SimpleCNN with 2 convolutional layers (ReLU + MaxPool), dropout, and 2 fully connected layers.

Dataset: CIFAR-10 (32×32 RGB images, 10 classes).

Transforms:

- Convert to tensor
- Normalize to [-1, 1] per channel

Optimizers:

- Adam with configurable learning rate
- SGD with momentum (0.9) and configurable learning rate

Device: Choose between cpu, cuda (NVIDIA GPU), or mps (Apple Silicon GPU).


📜 License
MIT — free to use for learning, teaching, and portfolio building.

