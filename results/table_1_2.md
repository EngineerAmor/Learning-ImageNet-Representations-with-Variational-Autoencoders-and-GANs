# Table 1.2 — Milestone 1 VAE Controlled Study (VAL Split)

| Run | Latent dim | β | KL warm-up | VAL Recon | VAL KL | VAL Neg-ELBO (β=1) | Active Units | Time (s) | Peak GPU (MB) | Chosen |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `default` | 64 | 1.0 | on (10 ep) | 205.4 | 66.29 | 271.7 | 64 / 64 | 124.8 | 203 |  |
| `z32` | 32 | 1.0 | on (10 ep) | 245.6 | 48.56 | 294.2 | 32 / 32 | 126.0 | 199 |  |
| `z128` | 128 | 1.0 | on (10 ep) | 191.0 | 71.60 | 262.6 | 128 / 128 | 130.0 | 213 | **★ YES** |
| `beta0.25` | 64 | 0.25 | on (10 ep) | 186.1 | 118.84 | 304.9 | 64 / 64 | 128.0 | 203 |  |
| `beta4` | 64 | 4.0 | on (10 ep) | 285.3 | 24.64 | 309.9 | 53 / 64 | 127.5 | 203 |  |
| `warmup_off` | 64 | 1.0 | off | 203.5 | 66.01 | 269.5 | 64 / 64 | 127.4 | 203 |  |

**Selected Final VAE:** `z128` — lowest VAL negative ELBO (`262.6`) evaluated at β = 1 among runs with at least 16 active latent units.
