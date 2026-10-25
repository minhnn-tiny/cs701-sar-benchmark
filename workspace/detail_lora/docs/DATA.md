# Train data: the numbers that drive the design

Train split only (9,392 images, 32,533 boxes); val and test images are not inspected. Reproduce with
`uv run python workspace/detail_lora/analyze_data.py`, which writes every number below to `docs/data_stats.json`.
"At 512" means after the model's resize to 512 × 512 (`sarbench/data.py`); box size = √(w·h).

## 1. Small objects dominate the metric

| | value |
|---|---|
| median box at 512 | 15.9 px (a 16 px ViT patch) |
| boxes < 8 px / < 6 px at 512 | 18.3 % / 4.5 % |
| COCO "small" (< 32² px, original pixels: what COCOeval buckets) | 69.7 % |

COCO mAP is the **mean over the 9 classes**, so the classes made of tiny boxes weigh 4/9 of mAP:

| class | boxes | boxes/img (max) | size p10 / p50 / p90 at 512 | < 8 px | COCO small | best anchor IoU < 0.5 |
|---|---:|---:|---|---:|---:|---:|
| wind_turbine | 8,625 | 9.4 (41) | 5.5 / 8.3 / 15.0 | 45 % | 100 % | 22 % |
| bridge | 3,984 | 3.2 (23) | 7.0 / 9.9 / 25.7 | 23 % | 95 % | 10 % |
| oil_tank | 5,580 | 4.5 (8) | 7.0 / 11.5 / 60.9 | 20 % | 92 % | 8 % |
| playground | 2,610 | 2.2 (20) | 10.7 / 14.8 / 19.5 | 0 % | 100 % | 1 % |
| harbor | 2,650 | 2.0 (15) | 23.2 / 42.9 / 216.8 | 0 % | 31 % | 8 % |
| car | 4,089 | 9.2 (28) | 31.0 / 41.4 / 56.4 | 0 % | 12 % | 1 % |
| ship | 1,544 | 1.3 (8) | 31.0 / 55.1 / 98.7 | 0 % | 59 % | 7 % |
| aircraft | 2,532 | 2.1 (21) | 49.0 / 76.4 / 106.8 | 0 % | 10 % | 1 % |
| airport | 919 | 1.4 (5) | 77.5 / 146.0 / 247.4 | 0 % | 0 % | 28 % (31 % have aspect > 3) |

**Decision.** The detector's fine levels (strides 4 and 8) are built by transposed convolutions from the
stride-16 ViT map, so a 6–10 px object is less than one token wide. A light convolutional branch on the
pixels feeds real stride-4 and stride-8 detail into those levels (design, see README). At most 41 boxes per image:
the cap of 100 detections per image is enough.

## 2. Anchor coverage

The best IoU of each box with any anchor (the RPN calls an anchor positive at ≥ 0.7, and matches a box below that
only through its single best anchor):

| box size at 512 | boxes | baseline anchors (8–181 px, 6 per location): median, share < 0.5 | proposed (5.7–181 px, 9 per location): median, share < 0.5 |
|---|---:|---|---|
| 4–6 px | 1,414 | 0.39, 92 % | 0.53, 39 % |
| 6–8 px | 4,512 | 0.57, 24 % | 0.58, 22 % |
| 8–12 px | 7,079 | 0.65, 3 % | 0.65, 3 % |
| all | 32,533 | 0.64, 10.5 % | 0.65, 7.9 % |

**Decision.** Add one √2 step below every level's sizes (the proposal: level sizes (5.7, 8, 11.3), (11.3, 16, 22.6),
… (90.5, 128, 181)). It mainly fixes the 4–6 px boxes. Below 8 px, the limit is the stride-4 grid (centres up to
2 px off), not the anchor size, so expect a modest gain. Elongated airports (aspect > 3) are 1/9 of mAP and large;
the RoI head's regression copes. Extra aspect ratios are left out.

## 3. Image size is a proxy for the source dataset

| class | 512² | 256² | 224² | other |
|---|---:|---:|---:|---:|
| aircraft | 1,119 | 4 | 59 | |
| airport, car, playground, wind_turbine | all | | | |
| bridge | 1,011 | 238 | | |
| harbor | 982 | 373 | | |
| oil_tank | 684 | 549 | | |
| ship | 13 | 1,157 | | 17 |

**Decision.** Stratify the holdout by (class, image size), so it keeps train's source mix. The baseline geometry
(every image resized to 512 × 512, non-square images stretched) is kept unchanged.

## 4. Intensity

| | p5 | p50 | p95 |
|---|---:|---:|---:|
| per-image mean (gray in [0, 1]) | 0.016 | 0.25 | 0.53 |
| per-image std | 0.037 | 0.128 | 0.236 |

- Class means differ: aircraft 0.095 and ship 0.138 (dark sea and runways) vs playground 0.413. Brightness is
  partly a class cue, so **no per-image standardisation**.
- Wind-turbine images have low contrast (mean std 0.048). 17.7 % of images have > 1 % saturated pixels (bright
  metal and corner reflectors). 9.9 % have > 10 % pure-black pixels (no-data borders or radar shadow).
- The global `MEAN, STD = 0.2526, 0.2133` (all train) stay valid and unchanged.

**Decision.** The frozen RGB patch embedding (timm sums the three channels) was never trained on these statistics.
It is cheap to train (0.2 M parameters, at a gentle learning rate). Mild random gamma augmentation covers the
display-stretch differences between sources.

## 5. Near-duplicate scenes inside train

Train holds the same scene several times, with other speckle or processing and a few pixels of shift. Method:
64 × 64 gray thumbnails, Pearson correlation > 0.85, connected components (`make_holdout.py`).

| | value |
|---|---|
| duplicate groups | 1,011 (2–12 images each) |
| images in a group | 2,594 (27.6 % of train) |
| most affected classes | harbor 838, oil_tank 612, wind_turbine 543, airport 336 |
| pairs with different labels, at correlation > 0.90 / 0.85 / 0.80 | 0 / 3 / 24 |

Zero cross-label pairs above 0.90 shows these are real copies, not hash collisions. A dHash check (also tried) was
not reliable on dark, low-contrast images, which collide.

**Decision.** A random holdout would put copies of training scenes in the holdout and reward memorisation. **The
holdout keeps every group whole on one side.** The Codabench val and test sets may also share scenes with train
(not checked: val/test images are not analysed), so holdout scores may run *below* the leaderboard. Compare designs
on the holdout, not holdout against leaderboard.

## 6. The holdout (`dataset/holdout/`, built by `make_holdout.py`)

- **Draw**: near-duplicate groups (§5) → strata (class, image size) → in each stratum, groups shuffled with
  `random.Random(0)` and taken while they fit, up to round(10 % × stratum size).
- **Result**: train 8,453 images / 29,331 boxes; holdout 939 images / 3,202 boxes; every class at
  9.9–10.0 %. All 1,011 groups are on one side.
- **Record**: `dataset/holdout/split.json` holds the seed, fraction, threshold, SHA-256 of the source label
  files, the groups and the holdout ids.
- **Layout**: `train/` and `val/` (labels.csv, instances.json, `images` → `../../train/images`), and `test → val`.
  The unchanged `train.py` scores val and test, so it scores the holdout twice.

## 7. SAR preprocessing and augmentation: proposals

| proposal | reason | decision |
|---|---|---|
| keep resize to 512² + global MEAN/STD | the rules and comparability with the baselines | **kept, unchanged** |
| random transpose (with the baseline's H/V flips: all 8 right-angle symmetries) | overhead images have no canonical orientation; boxes stay exactly axis-aligned | **use** (training only) |
| random gamma, γ ∈ [0.8, 1.25], p = 0.5, before normalisation | per-image brightness and contrast vary across sources (§4); monotone, keeps 0 and 1, keeps bright scatterers bright | **use** (training only) |
| trainable patch embedding | first layer sees SAR statistics and speckle, not RGB | **use** (model) |
| multiplicative speckle noise | images are 8-bit display products that already carry source-specific speckle; extra noise can erase 5–8 px point targets (45 % of wind turbines) | not used |
| despeckling (Lee / median) | smooths away the point scatterers that make up the small objects | not used |
| log / dB transform | display images are already compressed (median p99 = 0.78) | not used |
| random crops / scale jitter | zoom-in mismatches the fixed test geometry; zoom-out shrinks objects already < 8 px | not used |
| per-image standardisation | removes brightness, a class cue (§4) | not used |
