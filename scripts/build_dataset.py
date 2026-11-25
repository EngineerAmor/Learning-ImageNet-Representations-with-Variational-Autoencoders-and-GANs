#!/usr/bin/env python3
"""Build the single `.npz` array file and TRAIN/VAL split index artifacts from `T12_Splits`.

Produces:
  - `dataset.npz` containing keys: `x`, `y`, `train_idx`, `val_idx`
      * `x`: uint8, shape (N, 64, 64, 3), RGB in [0, 255]
      * `y`: int64, shape (N,), class labels 0..9
      * `train_idx`: int64, shape (N_train,), 0-based indices into x/y
      * `val_idx`: int64, shape (N_val,), 0-based indices into x/y
  - Standalone index files: `train_idx.npy`, `val_idx.npy`, `train_idx.json`,
    `val_idx.json`, `array_index.csv`, and `dataset_format.json`.

TEST is never read or included. Uses NumPy if installed, with a pure-stdlib
`.npy`/`.npz` fallback so it runs in minimal Python + Pillow environments.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence
from PIL import Image

NPY_MAGIC = b"\x93NUMPY"
IMAGE_SIZE = (64, 64)
NUM_CHANNELS = 3
NUM_CLASSES = 10


def encode_npy_header(descr: str, shape: tuple[int, ...]) -> bytes:
    """Build a NumPy NPY v1.0 header aligned to a 64-byte boundary."""
    dict_str = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {repr(shape)}, }}"
    base = dict_str.encode("ascii")
    # 10 bytes = 6 magic + 2 version + 2 header_len; +1 for trailing '\n'
    total_unpadded = 10 + len(base) + 1
    pad_len = (64 - (total_unpadded % 64)) % 64
    header = base + (b" " * pad_len) + b"\n"
    return NPY_MAGIC + b"\x01\x00" + struct.pack("<H", len(header)) + header


def make_npy_bytes(descr: str, shape: tuple[int, ...], payload: bytes) -> bytes:
    """Serialize raw C-contiguous bytes into a complete `.npy` byte stream."""
    return encode_npy_header(descr, shape) + payload


def pack_int64_npy(values: Sequence[int]) -> bytes:
    """Serialize a 1-D sequence of integers into an `<i8` `.npy` byte stream."""
    payload = struct.pack(f"<{len(values)}q", *values)
    return make_npy_bytes("<i8", (len(values),), payload)


def parse_npy_header(data: bytes) -> tuple[str, bool, tuple[int, ...], int]:
    """Parse an NPY v1.0/v2.0 header and return (descr, fortran_order, shape, data_offset)."""
    if not data.startswith(NPY_MAGIC):
        raise ValueError("Invalid NPY magic bytes")
    major, minor = data[6], data[7]
    if major == 1:
        header_len = struct.unpack("<H", data[8:10])[0]
        offset = 10 + header_len
        header_str = data[10:offset].decode("latin1")
    elif major == 2:
        header_len = struct.unpack("<I", data[8:12])[0]
        offset = 12 + header_len
        header_str = data[12:offset].decode("latin1")
    else:
        raise ValueError(f"Unsupported NPY version {(major, minor)}")

    import ast

    meta = ast.literal_eval(header_str.strip())
    return meta["descr"], bool(meta["fortran_order"]), tuple(meta["shape"]), offset


def _load_split_rows(splits_dir: Path, filename: str, expected_prefix: str) -> list[dict]:
    index_path = splits_dir / filename
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing split index file: {index_path}")
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected non-empty list in {index_path}")
    for row in rows:
        rel_path = row["path"]
        if not rel_path.startswith(expected_prefix):
            raise ValueError(f"Unexpected path {rel_path!r} in {filename}; expected prefix {expected_prefix!r}")
        label = int(row["train_label"])
        if not (0 <= label < NUM_CLASSES):
            raise ValueError(f"Label out of range 0..9: {label} for {rel_path}")
    return rows


def build_dataset(
    splits_dir: Path,
    output_dir: Path,
    npz_name: str = "dataset.npz",
    compress: bool = True,
) -> dict:
    """Build `dataset.npz` and standalone TRAIN/VAL index files from `splits_dir`."""
    train_rows = _load_split_rows(splits_dir, "train_index.json", "train/")
    val_rows = _load_split_rows(splits_dir, "val_index.json", "val/")

    all_rows: list[tuple[str, dict]] = [("train", r) for r in train_rows] + [
        ("val", r) for r in val_rows
    ]
    total = len(all_rows)
    n_train = len(train_rows)
    n_val = len(val_rows)

    expected_img_bytes = IMAGE_SIZE[0] * IMAGE_SIZE[1] * NUM_CHANNELS
    x_buf = bytearray(total * expected_img_bytes)
    labels: list[int] = [0] * total

    for idx, (_split, row) in enumerate(all_rows):
        img_path = splits_dir / row["path"]
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing image file: {img_path}")
        with Image.open(img_path) as im:
            if im.size != IMAGE_SIZE:
                raise ValueError(f"Image {img_path} has size {im.size}, expected {IMAGE_SIZE}")
            rgb = im.convert("RGB") if im.mode != "RGB" else im
            raw = rgb.tobytes()
        if len(raw) != expected_img_bytes:
            raise ValueError(f"Image {img_path} produced {len(raw)} bytes, expected {expected_img_bytes}")
        start = idx * expected_img_bytes
        x_buf[start : start + expected_img_bytes] = raw
        labels[idx] = int(row["train_label"])

    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, total))

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / npz_name
    train_npy_path = output_dir / "train_idx.npy"
    val_npy_path = output_dir / "val_idx.npy"
    train_json_path = output_dir / "train_idx.json"
    val_json_path = output_dir / "val_idx.json"
    array_index_path = output_dir / "array_index.csv"
    format_json_path = output_dir / "dataset_format.json"

    x_npy = make_npy_bytes("|u1", (total, IMAGE_SIZE[1], IMAGE_SIZE[0], NUM_CHANNELS), bytes(x_buf))
    y_npy = pack_int64_npy(labels)
    train_idx_npy = pack_int64_npy(train_idx)
    val_idx_npy = pack_int64_npy(val_idx)

    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(npz_path, "w", compression=compression) as zf:
        zf.writestr("x.npy", x_npy)
        zf.writestr("y.npy", y_npy)
        zf.writestr("train_idx.npy", train_idx_npy)
        zf.writestr("val_idx.npy", val_idx_npy)

    train_npy_path.write_bytes(train_idx_npy)
    val_npy_path.write_bytes(val_idx_npy)
    train_json_path.write_text(json.dumps(train_idx), encoding="utf-8")
    val_json_path.write_text(json.dumps(val_idx), encoding="utf-8")

    with array_index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["array_idx", "split", "train_label", "wnid", "path", "sha256"])
        for idx, (split, row) in enumerate(all_rows):
            writer.writerow(
                [idx, split, row["train_label"], row["wnid"], row["path"], row["sha256"]]
            )

    class_counts_train = dict(sorted(Counter(r["train_label"] for r in train_rows).items()))
    class_counts_val = dict(sorted(Counter(r["train_label"] for r in val_rows).items()))

    metadata_path = splits_dir / "metadata.json"
    class_mappings = []
    if metadata_path.is_file():
        class_mappings = json.loads(metadata_path.read_text(encoding="utf-8")).get(
            "class_mappings", []
        )

    format_spec = {
        "dataset_file": npz_name,
        "archive_format": "numpy_npz",
        "compressed": compress,
        "test_included": False,
        "test_note": "TEST split is intentionally untouched and excluded from this archive.",
        "total_images": total,
        "splits": {
            "train": {"count": n_train, "index_range": [0, n_train - 1], "per_class": class_counts_train},
            "val": {"count": n_val, "index_range": [n_train, total - 1], "per_class": class_counts_val},
        },
        "arrays": {
            "x": {
                "dtype": "uint8",
                "npy_descr": "|u1",
                "shape": [total, IMAGE_SIZE[1], IMAGE_SIZE[0], NUM_CHANNELS],
                "layout": "NHWC",
                "color_space": "RGB",
                "value_range": [0, 255],
            },
            "y": {
                "dtype": "int64",
                "npy_descr": "<i8",
                "shape": [total],
                "value_range": [0, NUM_CLASSES - 1],
            },
            "train_idx": {
                "dtype": "int64",
                "npy_descr": "<i8",
                "shape": [n_train],
                "description": "0-based indices into x and y for TRAIN images",
            },
            "val_idx": {
                "dtype": "int64",
                "npy_descr": "<i8",
                "shape": [n_val],
                "description": "0-based indices into x and y for VAL images",
            },
        },
        "standalone_index_files": [
            "train_idx.npy",
            "val_idx.npy",
            "train_idx.json",
            "val_idx.json",
            "array_index.csv",
        ],
        "class_mappings": class_mappings,
    }
    format_json_path.write_text(json.dumps(format_spec, indent=2) + "\n", encoding="utf-8")
    return format_spec


def main(argv: Iterable[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build T12 64x64 RGB dataset.npz and TRAIN/VAL index files.")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=repo_root / "T12_Splits",
        help="Path to T12_Splits directory (default: <repo>/T12_Splits)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "data",
        help="Output directory for dataset.npz and index files (default: <repo>/results/data)",
    )
    parser.add_argument(
        "--npz-name",
        default="dataset.npz",
        help="Filename for the single array archive (default: dataset.npz)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Store arrays uncompressed inside the .npz archive",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    spec = build_dataset(
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        npz_name=args.npz_name,
        compress=not args.no_compress,
    )
    print(
        f"Built {args.output_dir / args.npz_name}: "
        f"x={tuple(spec['arrays']['x']['shape'])} ({spec['arrays']['x']['dtype']}), "
        f"y={tuple(spec['arrays']['y']['shape'])} ({spec['arrays']['y']['dtype']}), "
        f"train_idx={tuple(spec['arrays']['train_idx']['shape'])}, "
        f"val_idx={tuple(spec['arrays']['val_idx']['shape'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
