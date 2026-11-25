# T12 Dataset Array & Split Index Format (Member A Deliverable)

## 1. Single Array File (`results/data/dataset.npz`)

Generated from `T12_Splits/` via:

```bash
python3 scripts/build_dataset.py
```

Archive format: compressed NumPy `.npz` containing the exact four keys expected by `load()` in `train_vae.py`:

| Key | Dtype | Shape | Layout / Semantics | Value Range |
| :--- | :--- | :--- | :--- | :--- |
| `x` | `uint8` | `(12653, 64, 64, 3)` | NHWC, RGB channel order | `0`–`255` |
| `y` | `int64` | `(12653,)` | Contiguous class labels (`train_label`) | `0`–`9` |
| `train_idx` | `int64` | `(11388,)` | 0-based indices into `x` and `y` for TRAIN (`0..11387`) | `0`–`11387` |
| `val_idx` | `int64` | `(1265,)` | 0-based indices into `x` and `y` for VAL (`11388..12652`) | `11388`–`12652` |

> **TEST is never touched:** `T12_Splits/test/` is empty and excluded. `x`, `y`, `train_idx`, and `val_idx` contain only TRAIN (11,388 images) and VAL (1,265 images).

## 2. Confirmation for `load()` in `train_vae.py`

The key names `x, y, train_idx, val_idx` match `load()` directly. Images are stored as `uint8` NHWC `(N, 64, 64, 3)` in `[0, 255]` so the archive stays compact (~132 MB compressed vs. ~620 MB as `float32`). If your PyTorch model expects NCHW `float32` in `[0.0, 1.0]`:

```python
import numpy as np
import torch

def load(npz_path="results/data/dataset.npz"):
    data = np.load(npz_path)
    x = data["x"]              # uint8, (12653, 64, 64, 3), RGB [0, 255]
    y = data["y"]              # int64, (12653,), labels 0..9
    train_idx = data["train_idx"]  # int64, (11388,)
    val_idx = data["val_idx"]      # int64, (1265,)

    # Convert to NCHW float32 in [0.0, 1.0] for PyTorch:
    x_train = torch.from_numpy(x[train_idx]).permute(0, 3, 1, 2).float() / 255.0
    y_train = torch.from_numpy(y[train_idx]).long()
    x_val = torch.from_numpy(x[val_idx]).permute(0, 3, 1, 2).float() / 255.0
    y_val = torch.from_numpy(y[val_idx]).long()
    return x_train, y_train, x_val, y_val
```

## 3. TRAIN / VAL Split Index Files

- **Array index files (0-based positions into `x` and `y`):**
  - `results/data/train_idx.npy` (`int64`, shape `(11388,)`) and `results/data/train_idx.json`
  - `results/data/val_idx.npy` (`int64`, shape `(1265,)`) and `results/data/val_idx.json`
  - `results/data/array_index.csv` (maps each `array_idx` `0..12652` to `split`, `train_label`, `wnid`, `path`, `sha256`)
  - `results/data/dataset_format.json` (machine-readable format specification)
- **Image file split indices (existing in `T12_Splits/`):**
  - `T12_Splits/train_index.csv` / `T12_Splits/train_index.json` (11,388 entries)
  - `T12_Splits/val_index.csv` / `T12_Splits/val_index.json` (1,265 entries)
  - `T12_Splits/test_index.csv` / `T12_Splits/test_index.json` (0 entries — untouched)

## 4. Class Label Mapping (`0`–`9`)

| `y` (`train_label`) | Folder / WNID | Train Count | Val Count | Total |
| :--- | :--- | :---: | :---: | :---: |
| `0` | `0_n01667778` | 1,170 | 130 | 1,300 |
| `1` | `1_n01855032` | 1,027 | 114 | 1,141 |
| `2` | `2_n02099267` | 1,170 | 130 | 1,300 |
| `3` | `3_n02099429` | 1,096 | 122 | 1,218 |
| `4` | `4_n02115913` | 1,170 | 130 | 1,300 |
| `5` | `5_n03000684` | 1,075 | 119 | 1,194 |
| `6` | `6_n03127747` | 1,170 | 130 | 1,300 |
| `7` | `7_n03496892` | 1,170 | 130 | 1,300 |
| `8` | `8_n04557648` | 1,170 | 130 | 1,300 |
| `9` | `9_n07717556` | 1,170 | 130 | 1,300 |
| **Total** | **10 classes** | **11,388** | **1,265** | **12,653** |
