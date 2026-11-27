# Latent Analysis (Milestone 1, Member C)

Model: final VAE (z128, beta = 1.0, 10-epoch KL warm-up). All analysis uses VAL latent means (n = 1265), fixed seed 0. TEST was not used.

## 1. Active units
All 128 / 128 dimensions pass the variance > 0.01 threshold (min var 0.0129, max 1.647), matching Member B's Table 1.2. Because the threshold does not discriminate, dimensions were ranked by var(mu) for the traversals (top 8: 80, 14, 54, 28, 112, 12, 33, 71).

## 2. t-SNE / UMAP
Figures: latent_tSNE.png, latent_UMAP.png (t-SNE perplexity 30, UMAP n_neighbors 15). Classes are heavily mixed with no clear clusters. Only butternut squash forms a visibly more compact region in UMAP. Global distances in these projections are distorted and were not interpreted.

## 3. Latent traversals
Base images 244 (flat-coated), 626 (chain saw), 1135 (butternut squash); each dimension varied from -3 to +3 sigma.
- dim 80 darkens the image in all three bases (appears brightness-related).
- dims 112, 12 and 71 mainly shift colour along a blue <-> brown/orange axis; dim 33 mainly changes darkness.
- dim 14 acts differently on each base image (entangled).
Little object-shape information is visible; most dimensions change colour and brightness together. Decoded images are blurry (pixel-wise loss at 64x64). Observations are visual, on three base images.

## 4. Interpolation
Pairs (VAL indices, results/interp_pairs.json, to be reused in M2 Table 2.3): close 244 -> 374 (flat-coated -> curly-coated), far 626 -> 1135 (chain saw -> squash), mid 496 -> 245 (dhole -> flat-coated).
Latent interpolation is smooth in colour and brightness, but intermediate frames are blurry colour blobs without a clear object. Pixel blending gives superimposed images (ghosting). Even the close pair changes mostly colour/background.

## 5. Class overlap
- kNN (k = 10, 5-fold on VAL) accuracy is 0.211 (chance about 0.10), so class information is weak.
- Closest centroid pairs: flat-coated/curly-coated 1.14, chain saw/water bottle 1.38, terrapin/chain saw 1.39, terrapin/water bottle 1.40, terrapin/dhole 1.48. Butternut squash is the most distant class (2.0 to 3.3).
- Largest confusions: chain saw -> merganser 0.54, water bottle -> merganser 0.48, butternut squash -> dhole 0.44. Merganser (recall 0.82) acts as an attractor; we hypothesise it occupies a dense central region (not verified).
- Flat-coated and curly-coated are the closest pair, but are rarely confused directly by kNN (3 and 5 images). Visually, both are dark dogs with very varied backgrounds, and the smooth/curly coat difference is hard to see at 64x64 (small visual sample).
- Predictions: retriever overlap supported; dhole near retrievers, terrapin/merganser overlap, and separation of chain saw/water bottle/squash not supported.

## 6. Why
The VAE is trained only to reconstruct, so its latent space is organised mainly by colour, brightness and background rather than class. Traversals, interpolations and kNN results all agree with this.

Limitations: one VAE, VAL only, visual inspection on few examples, centroid distance is not separability.
