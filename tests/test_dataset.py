"""Tests for the T12 dataset builder and generated array / index artifacts."""

from __future__ import annotations

import csv
import json
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_dataset  # noqa: E402


class TestNpyEncoding(unittest.TestCase):
    def test_header_alignment_and_roundtrip(self) -> None:
        raw_x = bytes(range(12))
        npy_x = build_dataset.make_npy_bytes("|u1", (1, 2, 2, 3), raw_x)
        descr, fortran, shape, offset = build_dataset.parse_npy_header(npy_x)
        self.assertEqual(descr, "|u1")
        self.assertFalse(fortran)
        self.assertEqual(shape, (1, 2, 2, 3))
        self.assertEqual(offset % 64, 0)
        self.assertEqual(npy_x[offset:], raw_x)

        npy_idx = build_dataset.pack_int64_npy([0, 5, 12652])
        descr, fortran, shape, offset = build_dataset.parse_npy_header(npy_idx)
        self.assertEqual(descr, "<i8")
        self.assertFalse(fortran)
        self.assertEqual(shape, (3,))
        self.assertEqual(offset % 64, 0)
        self.assertEqual(struct.unpack("<3q", npy_idx[offset:]), (0, 5, 12652))


class TestSyntheticDatasetBuild(unittest.TestCase):
    def test_synthetic_build_excludes_test_and_matches_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            splits = tmp_path / "T12_Splits"
            (splits / "train" / "0_cls").mkdir(parents=True)
            (splits / "val" / "1_cls").mkdir(parents=True)
            (splits / "test" / "0_cls").mkdir(parents=True)

            img_train = Image.new("RGB", (64, 64), (10, 20, 30))
            img_val = Image.new("RGB", (64, 64), (200, 210, 220))
            # Even if test has an image, builder must never include it
            img_test = Image.new("RGB", (64, 64), (255, 0, 0))
            img_train.save(splits / "train" / "0_cls" / "a.png")
            img_val.save(splits / "val" / "1_cls" / "b.png")
            img_test.save(splits / "test" / "0_cls" / "c.png")

            (splits / "train_index.json").write_text(
                json.dumps([{"path": "train/0_cls/a.png", "train_label": 0, "wnid": "n0", "sha256": "s0"}]),
                encoding="utf-8",
            )
            (splits / "val_index.json").write_text(
                json.dumps([{"path": "val/1_cls/b.png", "train_label": 1, "wnid": "n1", "sha256": "s1"}]),
                encoding="utf-8",
            )

            out_dir = tmp_path / "out"
            build_dataset.build_dataset(splits, out_dir)

            with zipfile.ZipFile(out_dir / "dataset.npz", "r") as zf:
                self.assertEqual(set(zf.namelist()), {"x.npy", "y.npy", "train_idx.npy", "val_idx.npy"})
                x_bytes = zf.read("x.npy")
                y_bytes = zf.read("y.npy")
                tr_bytes = zf.read("train_idx.npy")
                va_bytes = zf.read("val_idx.npy")

            descr, _, shape, off = build_dataset.parse_npy_header(x_bytes)
            self.assertEqual((descr, shape), ("|u1", (2, 64, 64, 3)))
            pixels = x_bytes[off:]
            self.assertEqual(len(pixels), 2 * 64 * 64 * 3)
            self.assertEqual(tuple(pixels[:3]), (10, 20, 30))
            self.assertEqual(tuple(pixels[64 * 64 * 3 : 64 * 64 * 3 + 3]), (200, 210, 220))

            _, _, y_shape, y_off = build_dataset.parse_npy_header(y_bytes)
            self.assertEqual(y_shape, (2,))
            self.assertEqual(struct.unpack("<2q", y_bytes[y_off:]), (0, 1))

            _, _, tr_shape, tr_off = build_dataset.parse_npy_header(tr_bytes)
            _, _, va_shape, va_off = build_dataset.parse_npy_header(va_bytes)
            self.assertEqual(tr_shape, (1,))
            self.assertEqual(va_shape, (1,))
            self.assertEqual(struct.unpack("<1q", tr_bytes[tr_off:]), (0,))
            self.assertEqual(struct.unpack("<1q", va_bytes[va_off:]), (1,))


class TestBuiltProjectArtifacts(unittest.TestCase):
    def test_project_dataset_npz_and_indices(self) -> None:
        data_dir = REPO_ROOT / "results" / "data"
        npz_path = data_dir / "dataset.npz"
        self.assertTrue(npz_path.is_file(), f"Missing {npz_path}")

        with zipfile.ZipFile(npz_path, "r") as zf:
            self.assertEqual(set(zf.namelist()), {"x.npy", "y.npy", "train_idx.npy", "val_idx.npy"})

            x_info = zf.getinfo("x.npy")
            with zf.open("x.npy") as fh:
                x_header_chunk = fh.read(256)
            descr, fortran, shape, offset = build_dataset.parse_npy_header(x_header_chunk)
            self.assertEqual(descr, "|u1")
            self.assertFalse(fortran)
            self.assertEqual(shape, (12653, 64, 64, 3))
            self.assertEqual(x_info.file_size, offset + 12653 * 64 * 64 * 3)

            y_bytes = zf.read("y.npy")
            tr_bytes = zf.read("train_idx.npy")
            va_bytes = zf.read("val_idx.npy")

        y_descr, _, y_shape, y_off = build_dataset.parse_npy_header(y_bytes)
        self.assertEqual((y_descr, y_shape), ("<i8", (12653,)))
        y_vals = struct.unpack("<12653q", y_bytes[y_off:])
        self.assertEqual(set(y_vals), set(range(10)))

        tr_descr, _, tr_shape, tr_off = build_dataset.parse_npy_header(tr_bytes)
        self.assertEqual((tr_descr, tr_shape), ("<i8", (11388,)))
        tr_vals = struct.unpack("<11388q", tr_bytes[tr_off:])

        va_descr, _, va_shape, va_off = build_dataset.parse_npy_header(va_bytes)
        self.assertEqual((va_descr, va_shape), ("<i8", (1265,)))
        va_vals = struct.unpack("<1265q", va_bytes[va_off:])

        self.assertEqual(len(set(tr_vals) & set(va_vals)), 0)
        self.assertEqual(set(tr_vals) | set(va_vals), set(range(12653)))

        # Verify standalone index files match archive contents
        self.assertEqual((data_dir / "train_idx.npy").read_bytes(), tr_bytes)
        self.assertEqual((data_dir / "val_idx.npy").read_bytes(), va_bytes)
        self.assertEqual(json.loads((data_dir / "train_idx.json").read_text()), list(tr_vals))
        self.assertEqual(json.loads((data_dir / "val_idx.json").read_text()), list(va_vals))

        # Verify array_index.csv has zero test rows and exact counts
        with (data_dir / "array_index.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 12653)
        self.assertEqual(sum(1 for r in rows if r["split"] == "train"), 11388)
        self.assertEqual(sum(1 for r in rows if r["split"] == "val"), 1265)
        self.assertEqual(sum(1 for r in rows if r["split"] == "test"), 0)

        # Spot-check pixel fidelity of first train image, last train image, first val, last val
        splits_dir = REPO_ROOT / "T12_Splits"
        stride = 64 * 64 * 3
        spot_indices = [0, 11387, 11388, 12652]
        with zipfile.ZipFile(npz_path, "r") as zf:
            with zf.open("x.npy") as fh:
                fh.read(offset)  # skip header
                x_raw = fh.read()
        for i in spot_indices:
            row = rows[i]
            with Image.open(splits_dir / row["path"]) as im:
                expected_pixels = im.convert("RGB").tobytes()
            actual_pixels = x_raw[i * stride : (i + 1) * stride]
            self.assertEqual(actual_pixels, expected_pixels, f"Pixel mismatch at array_idx={i}")
            self.assertEqual(y_vals[i], int(row["train_label"]))


if __name__ == "__main__":
    unittest.main()
