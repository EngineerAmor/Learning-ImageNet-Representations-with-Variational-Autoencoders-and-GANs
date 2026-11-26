"""Train one Convolutional VAE configuration with checkpoint/resume and compliant JSON logging.

Writes/updates `results/runs/vae_<name>.json` with:
  - member, commit_hash, status, config, seed, gpu, n_params
  - accumulated time_s across resumed sessions, peak_gpu_mb
  - final VAL metrics (val_recon, val_kl, val_neg_elbo at beta=1, active_units, val_mse_per_pixel)
  - per-epoch history for plotting training/validation curves
Also saves reconstruction grid across all 10 Team T12 classes on VAL.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .evaluate import evaluate_vae, load_splits, sha256_file
from .model import CLASS_NAMES, VAE, vae_loss


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return "unknown"


def get_gpu_name(dev: torch.device) -> str:
    if dev.type == "cuda":
        return torch.cuda.get_device_name(dev)
    return "CPU"


def ensure_dataset_npz(npz_path: Path, repo_root: Path) -> None:
    """If dataset.npz hasn't been built yet and T12_Splits exists, build it automatically."""
    if npz_path.is_file():
        return
    builder = repo_root / "scripts" / "build_dataset.py"
    splits_dir = repo_root / "T12_Splits"
    if builder.is_file() and splits_dir.is_dir():
        print(f"[setup] {npz_path} not found; building from {splits_dir} ...")
        subprocess.run(
            [sys.executable, str(builder), "--splits-dir", str(splits_dir), "--output-dir", str(npz_path.parent)],
            check=True,
        )
    else:
        raise FileNotFoundError(
            f"Missing {npz_path}. Provide --data path/to/dataset.npz or run scripts/build_dataset.py."
        )


def save_class_reconstructions(
    model: VAE,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    dev: torch.device,
    out_png: Path,
    run_name: str,
) -> None:
    """Plot original VAL image (top) vs deterministic posterior-mean reconstruction (bottom) for classes 0..9."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    picks: list[int] = []
    labels_present: list[int] = []
    for c in range(10):
        idx = (y_val == c).nonzero(as_tuple=False)
        if len(idx) > 0:
            picks.append(int(idx[0].item()))
            labels_present.append(c)

    with torch.no_grad():
        xs = x_val[picks].to(dev)
        mu, _ = model.encoder(xs)
        recons = model.decoder(mu).cpu()

    ncols = len(picks)
    fig, axes = plt.subplots(2, ncols, figsize=(1.7 * ncols, 3.8))
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i, c in enumerate(labels_present):
        axes[0, i].imshow(xs[i].cpu().permute(1, 2, 0).clamp(0, 1))
        axes[1, i].imshow(recons[i].permute(1, 2, 0).clamp(0, 1))
        short_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class {c}"
        axes[0, i].set_title(f"{c}: {short_name}", fontsize=7.5)
        axes[0, i].axis("off")
        axes[1, i].axis("off")

    axes[0, 0].text(-0.08, 0.5, "Original\n(VAL)", transform=axes[0, 0].transAxes, va="center", ha="right", fontsize=8.5, fontweight="bold")
    axes[1, 0].text(-0.08, 0.5, "Recon\n(VAE)", transform=axes[1, 0].transAxes, va="center", ha="right", fontsize=8.5, fontweight="bold")
    fig.suptitle(f"VAL Reconstructions Across All 10 Classes — {run_name}", fontsize=10, y=1.02)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def beta_schedule(epoch_idx: int, target_beta: float, warmup_epochs: int) -> float:
    """Linear KL annealing over `warmup_epochs` epochs (0 = warm-up switched off)."""
    if warmup_epochs <= 0:
        return float(target_beta)
    frac = min(1.0, float(epoch_idx + 1) / float(warmup_epochs))
    return float(target_beta * frac)


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Train one Milestone 1 Convolutional VAE run.")
    p.add_argument("--name", required=True, help="Run identifier, e.g. default, z32, z128, beta0.25, beta4, warmup_off")
    p.add_argument("--member", default="Bola", help="Name/GitHub username of the member running the experiment")
    p.add_argument("--data", type=Path, default=repo_root / "results" / "data" / "dataset.npz")
    p.add_argument("--dummy", action="store_true", help="Use synthetic random data for quick pipeline smoke testing")
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=int, default=10, help="0 disables KL warm-up; >0 ramps 0->beta linearly")
    p.add_argument("--sigma2", type=float, default=0.5, help="Fixed Gaussian observation variance (0.5 -> summed SE)")
    p.add_argument("--epochs", type=int, default=40, help="Identical epoch budget across all 6 runs")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=473)
    p.add_argument("--results-dir", type=Path, default=repo_root / "results")
    p.add_argument("--ckpt-dir", type=Path, default=None, help="Directory for checkpoints (point to Google Drive on Colab!)")
    args = p.parse_args(argv)

    runs_dir = args.results_dir / "runs"
    figs_dir = args.results_dir / "figures"
    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else (args.results_dir / "checkpoints")
    runs_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_json_path = runs_dir / f"vae_{args.name}.json"
    resume_ckpt_path = ckpt_dir / f"vae_{args.name}_last.pt"
    final_ckpt_path = ckpt_dir / f"vae_{args.name}_final.pt"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = get_gpu_name(dev)
    commit_hash = get_git_commit()

    config_dict = {
        "name": args.name,
        "latent_dim": args.latent_dim,
        "beta": args.beta,
        "warmup_epochs": args.warmup_epochs,
        "sigma2": args.sigma2,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimiser": "Adam",
        "dummy": args.dummy,
    }

    if args.dummy:
        rng = np.random.default_rng(args.seed)
        x_tr = torch.from_numpy(rng.integers(0, 256, (256, 3, 64, 64), dtype=np.uint8)).float().div_(255.0)
        x_va = torch.from_numpy(rng.integers(0, 256, (100, 3, 64, 64), dtype=np.uint8)).float().div_(255.0)
        y_va = torch.arange(100) % 10
    else:
        ensure_dataset_npz(args.data, repo_root)
        x_tr, _, x_va, y_va = load_splits(args.data)

    model = VAE(latent_dim=args.latent_dim).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict] = []
    start_epoch = 0
    prior_elapsed = 0.0
    peak_gpu_mb_prev = 0.0

    if resume_ckpt_path.is_file():
        ckpt = torch.load(resume_ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        history = ckpt.get("history", [])
        start_epoch = int(ckpt.get("epoch", 0))
        prior_elapsed = float(ckpt.get("elapsed_s", 0.0))
        peak_gpu_mb_prev = float(ckpt.get("peak_gpu_mb", 0.0))
        print(f"[{args.name}] Resumed from epoch {start_epoch}/{args.epochs} (prior time {prior_elapsed:.1f}s)")

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    t_session_start = time.perf_counter()

    def current_peak_mb() -> float:
        if dev.type == "cuda":
            return max(peak_gpu_mb_prev, torch.cuda.max_memory_allocated(dev) / (1024.0 ** 2))
        return peak_gpu_mb_prev

    try:
        for ep in range(start_epoch, args.epochs):
            # Per-epoch deterministic shuffle seed so resuming mid-run is reproducible
            g_loader = torch.Generator()
            g_loader.manual_seed(args.seed + ep)
            tr_loader = DataLoader(
                TensorDataset(x_tr),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                generator=g_loader,
            )

            model.train()
            b_now = beta_schedule(ep, args.beta, args.warmup_epochs)
            tr_loss_sum = tr_rec_sum = tr_kl_sum = 0.0
            n_batches = 0

            for (xb,) in tr_loader:
                xb = xb.to(dev)
                x_hat, mu, logvar = model(xb)
                loss, rec, kl = vae_loss(xb, x_hat, mu, logvar, beta=b_now, sigma2=args.sigma2)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss ({loss.item()}) at epoch {ep + 1}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                tr_loss_sum += loss.item()
                tr_rec_sum += rec.item()
                tr_kl_sum += kl.item()
                n_batches += 1

            val_metrics = evaluate_vae(model, x_va, device=dev, sigma2=args.sigma2)
            elapsed_total = prior_elapsed + (time.perf_counter() - t_session_start)
            peak_mb = current_peak_mb()

            ep_row = {
                "epoch": ep + 1,
                "beta_effective": b_now,
                "train_loss": tr_loss_sum / max(1, n_batches),
                "train_recon": tr_rec_sum / max(1, n_batches),
                "train_kl": tr_kl_sum / max(1, n_batches),
                **val_metrics,
            }
            history.append(ep_row)
            print(
                f"[{args.name}] ep {ep+1:3d}/{args.epochs} (beta={b_now:.3f}) | "
                f"train rec={ep_row['train_recon']:8.1f} kl={ep_row['train_kl']:6.2f} | "
                f"VAL rec={val_metrics['val_recon']:8.1f} kl={val_metrics['val_kl']:6.2f} "
                f"negELBO={val_metrics['val_neg_elbo']:8.1f} active={val_metrics['active_units']}/{args.latent_dim}"
            )

            # Save resumable checkpoint + update run record after EVERY epoch (safe for Colab disconnects)
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": ep + 1,
                    "elapsed_s": elapsed_total,
                    "peak_gpu_mb": peak_mb,
                    "config": config_dict,
                    "seed": args.seed,
                    "history": history,
                },
                resume_ckpt_path,
            )

            partial_record = {
                "run_id": f"vae_{args.name}",
                "name": args.name,
                "status": "completed" if (ep + 1 == args.epochs) else "running",
                "member": args.member,
                "commit_hash": commit_hash,
                "seed": args.seed,
                "gpu": gpu_name,
                "n_params": n_params,
                "config": config_dict,
                "epochs_completed": ep + 1,
                "time_s": elapsed_total,
                "peak_gpu_mb": peak_mb,
                "final_metrics": val_metrics,
                "history": history,
            }
            run_json_path.write_text(json.dumps(partial_record, indent=2) + "\n", encoding="utf-8")

        # Save lightweight final weights & SHA-256 hash
        torch.save(model.state_dict(), final_ckpt_path)
        ckpt_sha = sha256_file(final_ckpt_path)

        final_record = json.loads(run_json_path.read_text(encoding="utf-8"))
        final_record["status"] = "completed"
        final_record["checkpoint_path"] = str(final_ckpt_path)
        final_record["checkpoint_sha256"] = ckpt_sha
        run_json_path.write_text(json.dumps(final_record, indent=2) + "\n", encoding="utf-8")

        save_class_reconstructions(
            model=model,
            x_val=x_va,
            y_val=y_va,
            dev=dev,
            out_png=figs_dir / f"vae_recon_{args.name}.png",
            run_name=args.name,
        )
        print(f"[{args.name}] Completed! SHA-256={ckpt_sha[:16]}... Record -> {run_json_path}")
        return 0

    except Exception as exc:
        elapsed_total = prior_elapsed + (time.perf_counter() - t_session_start)
        failed_record = {
            "run_id": f"vae_{args.name}",
            "name": args.name,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "member": args.member,
            "commit_hash": commit_hash,
            "seed": args.seed,
            "gpu": gpu_name,
            "n_params": n_params,
            "config": config_dict,
            "epochs_completed": len(history),
            "time_s": elapsed_total,
            "peak_gpu_mb": current_peak_mb(),
            "final_metrics": history[-1] if history else {},
            "history": history,
        }
        run_json_path.write_text(json.dumps(failed_record, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
