## Dataset Array & Format (Member A Deliverable)

Build the single `.npz` dataset archive and standalone TRAIN/VAL split index files from `T12_Splits/` (TEST is never touched):

```bash
python3 scripts/build_dataset.py
python3 -m unittest discover -s tests -v
```

This creates `results/data/dataset.npz` (gitignored because it exceeds GitHub's 100 MB limit) with keys `x`, `y`, `train_idx`, `val_idx`, along with `train_idx.npy`/`.json`, `val_idx.npy`/`.json`, `array_index.csv`, and `dataset_format.json`. See `DATA_FORMAT.md` for exact dtypes, shapes, and a `train_vae.py` `load()` example.

## Windows local commands

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH="$PWD\src"
python -m milestone1.cli audit --source DataSet\T12_Images --output results\data_audit.json
python -m milestone1.cli build --source DataSet\T12_Images --output results\data
pytest -q
python -m milestone1.train --data results\data\dataset.npz --checkpoint results\checkpoints\vae_baseline_last.pt --epochs 1 --batch-size 16 --run-record results\runs\smoke_vae.json
python scripts\index_checkpoints.py
python -m milestone1.evaluate --checkpoint results\checkpoints\vae_baseline_last.pt --data results\data\dataset.npz --split val --latent-dim 64
```

For official TEST, provide the untouched official validation directory and a CSV with
`filename,original_label` (optionally `wnid`), then rerun `build` with both options.
Do not use a random split as TEST.

## Colab

Upload/clone this project, mount Drive, set `PYTHONPATH=src`, and point `--output` and
`--checkpoint` to Drive. The same commands apply. A T4 is recommended for the full six-run
study; this environment reports CPU-only PyTorch. The repository intentionally does not include
raw images, checkpoints, or generated figures.

## Protocol

`configs/milestone1.json` contains exactly six controlled VAE runs. Baseline is latent 64,
beta 1, warm-up on; alternatives change one setting. Full runs, classifier studies, official
TEST, figures and tables are generated only from actual records and are not represented by
fabricated values. See `docs/milestone1_requirements.md`.

The assignment card is a required human input: confirm the original ImageNet label numbers,
WNIDs, and official validation mapping before real-data TEST evaluation. Predictions must be
written and committed by the named human author before main studies. Git contribution statistics
are unavailable until a repository history exists.
