"""Run all 6 controlled Milestone 1 VAE runs and regenerate Table 1.2 + figures + checkpoints.csv.

Notice how README.md in the repo states:
  "Baseline is latent 64, beta 1, warm-up on; alternatives change one setting."
Let's make our 6 controlled runs match BOTH the project brief AND Member A's README:
  1. default     : latent_dim=64,  beta=1.0,  warmup_epochs=10 (warm-up ON)
  2. z32         : latent_dim=32,  beta=1.0,  warmup_epochs=10 (latent size #2)
  3. z128        : latent_dim=128, beta=1.0,  warmup_epochs=10 (latent size #3)
  4. beta0.25    : latent_dim=64,  beta=0.25, warmup_epochs=10 (beta #2)
  5. beta4       : latent_dim=64,  beta=4.0,  warmup_epochs=10 (beta #3)
  6. warmup_off  : latent_dim=64,  beta=1.0,  warmup_epochs=0  (KL warm-up OFF)
Every run changes ONE setting from `default` and shares the exact same epoch budget!
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def load_study_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def choose_final_vae(records: list[dict], min_active_units: int = 16) -> dict:
    """Stated VAL selection criterion:
    Lowest VAL negative ELBO (computed at beta = 1: val_recon + val_kl) among non-collapsed runs
    that retain at least `min_active_units` active latent dimensions (Var_val(mu_k) > 0.01).
    """
    viable = [r for r in records if r["final_metrics"]["active_units"] >= min_active_units]
    pool = viable if viable else records
    return min(pool, key=lambda r: r["final_metrics"]["val_neg_elbo"])


def generate_tables_and_figures(results_dir: Path, run_order: list[str], min_active_units: int = 16) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs_dir = results_dir / "runs"
    figs_dir = results_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for name in run_order:
        p = runs_dir / f"vae_{name}.json"
        if p.is_file():
            rec = json.loads(p.read_text(encoding="utf-8"))
            if rec.get("status") == "completed":
                records.append(rec)

    if not records:
        raise RuntimeError(f"No completed VAE run records found in {runs_dir}")

    chosen = choose_final_vae(records, min_active_units=min_active_units)
    chosen_name = chosen["name"]

    # 1. Write Table 1.2 as CSV (required by brief Section 'Submission and Assessment')
    csv_path = results_dir / "table_1_2.csv"
    fieldnames = [
        "run",
        "latent_dim",
        "beta",
        "kl_warmup",
        "val_recon",
        "val_kl",
        "val_neg_elbo_beta1",
        "val_mse_per_pixel",
        "active_units",
        "time_s",
        "peak_gpu_mb",
        "n_params",
        "chosen_vae",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            c = r["config"]
            m = r["final_metrics"]
            writer.writerow(
                {
                    "run": r["name"],
                    "latent_dim": c["latent_dim"],
                    "beta": c["beta"],
                    "kl_warmup": f"on ({c['warmup_epochs']} ep)" if c["warmup_epochs"] > 0 else "off",
                    "val_recon": f"{m['val_recon']:.2f}",
                    "val_kl": f"{m['val_kl']:.2f}",
                    "val_neg_elbo_beta1": f"{m['val_neg_elbo']:.2f}",
                    "val_mse_per_pixel": f"{m['val_mse_per_pixel']:.6f}",
                    "active_units": f"{m['active_units']}/{c['latent_dim']}",
                    "time_s": f"{r['time_s']:.1f}",
                    "peak_gpu_mb": f"{r['peak_gpu_mb']:.1f}",
                    "n_params": r["n_params"],
                    "chosen_vae": "YES" if r["name"] == chosen_name else "",
                }
            )

    # 2. Write Table 1.2 as Markdown for easy inspection & report drafting
    md_lines = [
        "# Table 1.2 — Milestone 1 VAE Controlled Study (VAL Split)",
        "",
        "| Run | Latent dim | β | KL warm-up | VAL Recon | VAL KL | VAL Neg-ELBO (β=1) | Active Units | Time (s) | Peak GPU (MB) | Chosen |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in records:
        c = r["config"]
        m = r["final_metrics"]
        mark = "**★ YES**" if r["name"] == chosen_name else ""
        w_str = f"on ({c['warmup_epochs']} ep)" if c["warmup_epochs"] > 0 else "off"
        md_lines.append(
            f"| `{r['name']}` | {c['latent_dim']} | {c['beta']} | {w_str} | "
            f"{m['val_recon']:.1f} | {m['val_kl']:.2f} | {m['val_neg_elbo']:.1f} | "
            f"{m['active_units']} / {c['latent_dim']} | {r['time_s']:.1f} | {r['peak_gpu_mb']:.0f} | {mark} |"
        )
    md_lines.extend(
        [
            "",
            f"**Selected Final VAE:** `{chosen_name}` — lowest VAL negative ELBO (`{chosen['final_metrics']['val_neg_elbo']:.1f}`) "
            f"evaluated at β = 1 among runs with at least {min_active_units} active latent units.",
        ]
    )
    (results_dir / "table_1_2.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (results_dir / "selected_vae.json").write_text(
        json.dumps(
            {
                "selected_run": chosen_name,
                "criterion": f"Lowest VAL negative ELBO (beta=1) with active_units >= {min_active_units}",
                "checkpoint_path": chosen.get("checkpoint_path", ""),
                "checkpoint_sha256": chosen.get("checkpoint_sha256", ""),
                "latent_dim": chosen["config"]["latent_dim"],
                "val_metrics": chosen["final_metrics"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # 3. Update results/checkpoints.csv (preserving non-VAE rows if other teammates added classifiers)
    ckpt_csv = results_dir / "checkpoints.csv"
    ckpt_fields = ["model_id", "milestone", "member", "latent_dim", "checkpoint_file", "sha256", "cloud_url", "chosen"]
    existing_rows: list[dict] = []
    if ckpt_csv.is_file():
        with ckpt_csv.open(encoding="utf-8") as fh:
            existing_rows = [row for row in csv.DictReader(fh) if not row.get("model_id", "").startswith("vae_")]

    with ckpt_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ckpt_fields)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({k: row.get(k, "") for k in ckpt_fields})
        for r in records:
            writer.writerow(
                {
                    "model_id": f"vae_{r['name']}",
                    "milestone": "M1",
                    "member": r.get("member", "Member B"),
                    "latent_dim": r["config"]["latent_dim"],
                    "checkpoint_file": Path(r.get("checkpoint_path", "")).name,
                    "sha256": r.get("checkpoint_sha256", ""),
                    "cloud_url": "<PASTE_GOOGLE_DRIVE_SHARE_LINK>",
                    "chosen": "YES" if r["name"] == chosen_name else "NO",
                }
            )

    # 4. Plot training & validation curves of Reconstruction and KL terms across all 6 runs
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for r in records:
        ep = [h["epoch"] for h in r["history"]]
        lbl = f"{r['name']} (z={r['config']['latent_dim']}, β={r['config']['beta']})"
        lw = 2.2 if r["name"] == chosen_name else 1.4
        axes[0, 0].plot(ep, [h["train_recon"] for h in r["history"]], label=lbl, linewidth=lw)
        axes[0, 1].plot(ep, [h["val_recon"] for h in r["history"]], label=lbl, linewidth=lw)
        axes[1, 0].plot(ep, [h["train_kl"] for h in r["history"]], label=lbl, linewidth=lw)
        axes[1, 1].plot(ep, [h["val_kl"] for h in r["history"]], label=lbl, linewidth=lw)

    axes[0, 0].set_title("TRAIN Reconstruction Term (nats)")
    axes[0, 1].set_title("VAL Reconstruction Error (nats)")
    axes[1, 0].set_title("TRAIN KL Divergence (nats)")
    axes[1, 1].set_title("VAL KL Divergence (nats)")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    axes[0, 1].legend(fontsize=8, loc="best")
    fig.suptitle("Milestone 1 — VAE Reconstruction and KL Curves (6 Controlled Runs)", fontsize=12)
    fig.tight_layout()
    fig.savefig(figs_dir / "vae_training_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\n".join(md_lines))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Execute all 6 Milestone 1 VAE runs and build Table 1.2.")
    p.add_argument("--config", type=Path, default=repo_root / "configs" / "vae_study.json")
    p.add_argument("--member", default="Bola")
    p.add_argument("--data", type=Path, default=repo_root / "results" / "data" / "dataset.npz")
    p.add_argument("--dummy", action="store_true", help="Run fast smoke test on synthetic noise")
    p.add_argument("--epochs", type=int, default=None, help="Override epoch budget in config (e.g. --epochs 2 for smoke test)")
    p.add_argument("--results-dir", type=Path, default=repo_root / "results")
    p.add_argument("--ckpt-dir", type=Path, default=None, help="Save checkpoints directly to Google Drive folder on Colab")
    args = p.parse_args()

    study = load_study_config(args.config)
    defaults = study["defaults"]
    epochs = args.epochs if args.epochs is not None else defaults["epochs"]

    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_path

    run_order: list[str] = []
    for run_spec in study["runs"]:
        name = run_spec["name"]
        run_order.append(name)
        merged = {**defaults, **run_spec}

        existing_json = args.results_dir / "runs" / f"vae_{name}.json"
        if existing_json.is_file():
            existing = json.loads(existing_json.read_text(encoding="utf-8"))
            if existing.get("status") == "completed" and existing.get("epochs_completed") == epochs:
                print(f"[skip] vae_{name} already completed ({epochs} epochs).")
                continue

        cmd = [
            sys.executable,
            "-m",
            "vae.train",
            "--name", name,
            "--member", args.member,
            "--latent-dim", str(merged["latent_dim"]),
            "--beta", str(merged["beta"]),
            "--warmup-epochs", str(min(merged["warmup_epochs"], epochs) if merged["warmup_epochs"] > 0 else 0),
            "--sigma2", str(merged["sigma2"]),
            "--epochs", str(epochs),
            "--batch-size", str(merged["batch_size"]),
            "--lr", str(merged["lr"]),
            "--seed", str(merged["seed"]),
            "--results-dir", str(args.results_dir),
        ]
        if args.ckpt_dir is not None:
            cmd += ["--ckpt-dir", str(args.ckpt_dir)]
        if args.dummy:
            cmd += ["--dummy"]
        else:
            cmd += ["--data", str(args.data)]

        subprocess.run(cmd, check=True, env=env)

    min_active = 1 if args.dummy else int(study.get("selection", {}).get("min_active_units", 16))
    generate_tables_and_figures(args.results_dir, run_order=run_order, min_active_units=min_active)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
