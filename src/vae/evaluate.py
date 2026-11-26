"""Evaluate a trained VAE checkpoint on VAL (never TEST unless explicitly requested in M2).

Reproduces every metric required for Table 1.2:
  - val_recon (summed SE / (2*sigma2), same metric for every run)
  - val_kl (summed over latent dimensions, nats)
  - val_neg_elbo (val_recon + 1.0 * val_kl, computed with beta = 1)
  - val_mse_per_pixel (MSE per pixel-channel in [0, 1], also needed for M2 Table 2.3)
  - active_units (count of dims with Var_val(mu_k) > 0.01)

Usage:
  PYTHONPATH=src python -m vae.evaluate --checkpoint path/to/vae_default_final.pt --data results/data/dataset.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import VAE, vae_loss


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_splits(npz_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load Member A's dataset.npz (`x`, `y`, `train_idx`, `val_idx`) into NCHW float32 [0, 1]."""
    data = np.load(npz_path)
    x = data["x"]              # uint8 (N, 64, 64, 3)
    y = data["y"]              # int64 (N,)
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]

    x_train = torch.from_numpy(x[train_idx]).permute(0, 3, 1, 2).float().div_(255.0)
    y_train = torch.from_numpy(y[train_idx]).long()
    x_val = torch.from_numpy(x[val_idx]).permute(0, 3, 1, 2).float().div_(255.0)
    y_val = torch.from_numpy(y[val_idx]).long()
    return x_train, y_train, x_val, y_val


@torch.no_grad()
def evaluate_vae(
    model: VAE,
    x_split: torch.Tensor,
    device: torch.device | str,
    batch_size: int = 256,
    sigma2: float = 0.5,
    eval_seed: int = 473,
) -> dict:
    """Deterministically evaluate a VAE on `x_split` with beta = 1 for negative ELBO."""
    model.eval()
    loader = DataLoader(TensorDataset(x_split), batch_size=batch_size, shuffle=False)
    gen = torch.Generator(device=device)
    gen.manual_seed(eval_seed)

    recon_sum = 0.0
    kl_sum = 0.0
    mse_pixel_sum = 0.0
    n_total = 0
    mus: list[torch.Tensor] = []

    for (xb,) in loader:
        xb = xb.to(device)
        mu, logvar = model.encoder(xb)
        z = model.reparameterise(mu, logvar, generator=gen)
        x_hat = model.decoder(z)
        _, recon_batch, kl_batch = vae_loss(xb, x_hat, mu, logvar, beta=1.0, sigma2=sigma2)

        # Also measure deterministic mean-decode per-pixel MSE for Milestone 2 Table 2.3
        x_hat_det = model.decoder(mu)
        mse_batch = ((xb - x_hat_det) ** 2).mean(dim=(1, 2, 3)).sum().item()

        bsz = xb.size(0)
        recon_sum += recon_batch.item() * bsz
        kl_sum += kl_batch.item() * bsz
        mse_pixel_sum += mse_batch
        n_total += bsz
        mus.append(mu.cpu())

    all_mu = torch.cat(mus, dim=0)  # (N_val, latent_dim)
    # Unbiased sample variance over VAL images; active when variance > 0.01 (brief Section M1.1)
    mu_var = all_mu.var(dim=0, unbiased=False)
    active_units = int((mu_var > 0.01).sum().item())

    val_recon = recon_sum / n_total
    val_kl = kl_sum / n_total
    return {
        "val_recon": float(val_recon),
        "val_kl": float(val_kl),
        "val_neg_elbo": float(val_recon + val_kl),
        "val_mse_per_pixel": float(mse_pixel_sum / n_total),
        "active_units": active_units,
        "latent_dim": int(model.latent_dim),
        "posterior_mean_var_min": float(mu_var.min().item()),
        "posterior_mean_var_median": float(mu_var.median().item()),
        "posterior_mean_var_max": float(mu_var.max().item()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate a saved VAE checkpoint on VAL.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to saved .pt weights/checkpoint")
    p.add_argument("--data", type=Path, default=Path("results/data/dataset.npz"))
    p.add_argument("--latent-dim", type=int, default=None, help="Inferred from checkpoint if omitted")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--sigma2", type=float, default=0.5)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw

    latent_dim = args.latent_dim
    if latent_dim is None:
        latent_dim = int(state_dict["encoder.fc_mu.weight"].shape[0])

    model = VAE(latent_dim=latent_dim).to(dev)
    model.load_state_dict(state_dict)

    _, _, x_val, _ = load_splits(args.data)
    metrics = evaluate_vae(model, x_val, device=dev, batch_size=args.batch_size, sigma2=args.sigma2)
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["sha256"] = sha256_file(args.checkpoint)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
