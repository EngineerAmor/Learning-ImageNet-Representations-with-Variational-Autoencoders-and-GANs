

### Traversal findings, dims 112, 12, 33, 71, 28 (base idx 244, 626, 1135)
Based on the visible rows of the saved traversal figures. Tentative.

- dim 112: shifts colour along a blue <-> brown/orange axis (blue/grey at -3 sigma,
  brown/dark orange at +3 sigma) on bases 244, 626 and 1135. Appears colour-related.
- dim 12: shifts colour in the opposite direction (brown/orange -> blue/dark).
  Appears colour-related.
- dim 33: mainly changes darkness (grey -> black) on 244 and 626; adds a dark blob
  on 1135. Appears brightness-related.
- dim 71: changes colour/darkness (blue/black -> brown) on 244 and 626; small change on 1135.
- dim 28 (1135): beige/dark -> saturated orange. Colour/saturation.
- Summary: the most active dimensions mainly encode global colour (blue <-> brown/orange)
  and brightness, with little visible object-shape information. This supports prediction 6.
  Several dimensions change colour and brightness together (entangled).


### Interpolation findings
Pairs (VAL array indices, saved in results/interp_pairs.json; reuse in M2 Table 2.3):
- close_flat_curly: 244 -> 374
- far_chainsaw_squash: 626 -> 1135
- mid_dhole_flat: 496 -> 245

- In all three pairs the latent interpolation is smooth: colour and brightness change
  gradually from the first image to the second, with no abrupt jumps.
- Intermediate decoded images are blurry colour blobs. Object shape (dog, chain saw,
  squash) is not clearly visible in the middle frames.
- Pixel blending instead shows two superimposed images (ghosting), not a plausible
  intermediate image.
- Even the close pair (flat-coated -> curly-coated) is mostly a colour/background change
  (dark -> green/grey), not a change of dog type. Consistent with the VAE encoding colour
  and lighting more than class identity.
- Caveat: observed visually on 3 pairs only.


### Class overlap findings (VAL latent means, z128)
- kNN (k=10, 5-fold on VAL) accuracy = 0.211 (chance about 0.10): class information in the latent means is weak.
- Closest centroid pairs: flat-coated<->curly-coated 1.14; chain saw<->water bottle 1.38;
  terrapin<->chain saw 1.39; terrapin<->water bottle 1.40; terrapin<->dhole 1.48.
- Largest kNN confusions: chain saw->merganser 0.54; water bottle->merganser 0.48;
  butternut squash->dhole 0.44; crash helmet->merganser 0.43; flat-coated->merganser 0.42.
- Merganser is the only class recognised well (recall 0.82) and acts as an attractor for other classes.
  Hypothesis (not verified): it occupies a dense central region of the latent space.
- Predictions check: (1) retrievers overlap: supported (closest pair). (2) dhole near retrievers: NOT supported
  (centroid distances about 2.2-2.3). (3) terrapin/merganser overlap: NOT supported (1.9). (4) chain saw, water
  bottle, squash separated: NOT supported by kNN (squash is far in centroid distance but confused with dhole).
  (5) classes overlap a lot: supported. (6) top dims encode colour/brightness: supported by traversals.
- Caveats: single VAE, VAL only, centroid distance is not the same as separability.


### Examples of overlapping classes (flat-coated vs curly-coated retriever)
Figure: results/figures/overlap_flat_curly_examples.png (4 random images per class, seed 0;
misclassified indices: flat->curly [246 298 359], curly->flat [405 415 430 435]).

- Direct flat<->curly confusion by kNN is rare: 3 flat images predicted curly, 5 curly images predicted flat
  (about 130 images per class). The two classes are the closest centroid pair (1.14), yet kNN mostly
  confuses them with merganser, not with each other.
- Visually, images of both classes are dark dogs with very varied backgrounds (grass, water, indoor,
  close-up puppies). At 64x64 the difference between smooth and curly coat is hard to see, while background and
  lighting vary a lot within each class. This is a plausible explanation for the overlap (visual inspection of
  a small sample, not verified).
