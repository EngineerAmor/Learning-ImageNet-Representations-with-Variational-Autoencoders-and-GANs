# Pre-Study Predictions — Team T12

## Study 1: Milestone 1 — VAE Controlled Settings Study (6 Runs)

- **Author:** Bola (Member B — VAE Lead)
- **Date written (before running the 6-run study):** 2026-11-26

### Predictions and Reasoning
1. **Latent size (`z32` vs `default` `z64` vs `z128`):**
   - `z32` will have higher VAL reconstruction error than `default` (`z64`) because 32 numbers are a tight bottleneck for 64×64 colour ImageNet scenes with diverse backgrounds, though almost all 32 units should stay active.
   - `z128` will achieve slightly lower reconstruction error than `z64`, but many of the extra dimensions will collapse to the prior (`Var_VAL(mu_k) <= 0.01`), so its active-unit fraction will drop and its total VAL negative ELBO (`beta = 1`) will be close to `z64`.
2. **KL weight (`beta0.25` vs `default` `beta=1.0` vs `beta4`):**
   - `beta0.25` relaxes the prior penalty, so it should give the lowest VAL reconstruction error and high active units, at the cost of a much larger KL divergence (worsening its true `beta=1` negative ELBO).
   - `beta4` (`beta`-VAE with strong regularisation) will aggressively push `q(z|x)` toward `N(0, I)`, causing partial posterior collapse (significantly fewer active units), tiny VAL KL, and noticeably blurrier reconstructions.
3. **KL warm-up (`warmup_off` vs `default` 10-epoch linear warm-up):**
   - Without KL warm-up (`warmup_off`), applying the full `beta = 1` penalty from epoch 1 lets the optimiser cheaply reduce loss early by driving `mu_k(x) -> 0` before the decoder learns useful filters, resulting in fewer active units than `default`.
4. **Expected winner under our VAL selection criterion:**
   - Selecting by lowest VAL negative ELBO (`beta = 1`) among non-collapsed models (`>= 16` active units), I expect **`default` (`z=64, beta=1.0, warm-up on`)** (or closely **`z128`**) to win.

##Study 2: Milestone 1, Latent Space Analysis (Member C)

**Author: Nourhan Alaa (Member C, Latent Analysis)**

Date written (before running t-SNE/UMAP, traversals, interpolations): 2026-11-26
Predictions and Reasoning
Flat-coated and curly-coated retriever will overlap the most in the latent space, because both are black dogs with a similar head shape.
Dhole will sit close to the retrievers (brown dog-like body, similar poses and backgrounds).
Terrapin and red-breasted merganser may partly overlap (water, mud and green backgrounds).
Chain saw, water bottle and butternut squash will be more separated from the animal classes.
Overall, classes will overlap a lot, because the VAE is trained only to reconstruct, so it groups images by colour, lighting and background, not by class.
The most active latent dimensions will mostly encode global properties like brightness and background colour, not object identity.
