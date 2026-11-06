# DATA: the train split and the local holdout

Author: `data-eng`. Scope: **train only** (`dataset/train/`: 9,392 images, 32,533 boxes). No val/test pixels or
metadata were read for any number here. Code: `workspace/sarmtl/data/`. Every table below is printed by
`analyze_train.py` (full output: `dataset/holdout_sarmtl/analysis/train_analysis.md`). Box size = sqrt(w × h); "at
512" = after the baseline resize to 512 × 512, "original" = in the pixels of the file (what COCO scoring uses).

## 0. Facts that should change the design

1. **73.0 % of train images share pixels or objects with another train image.** 6,859 of 9,392 images sit in 964
   groups. The 40,787 links between them:
   - flipped or rotated pixel copies (4,238);
   - revisits of the same site, with another acquisition and the same boxes (3,640);
   - overlapping tiles found from boxes (2,582) and from pixels (13,832 + 14,024 under the normalised correlation);
   - flipped revisits (187), weaker relations (338) and audit-loop links (1,946).

   Train has only **3,497 independent units** (2.7 images per unit). Images per unit by class: wind turbine 102 (the
   whole class is 9 units), aircraft 20.4, oil tank 8.5, airport 3.7, harbor 3.7, ship 2.2, car 1.5, playground 1.4,
   bridge 1.1. **A naive random 10 % holdout would leak.** 72 % of its images would share pixels or objects with
   local train (wind turbine 100 %, oil tank 98 %, aircraft 97 %). Our holdout is leak-free with respect to every
   detector (§6).
2. **Train already contains flipped copies.** Left-right (1,070 links), up-down (1,095) and 180° (445) copies are
   in train, but 90° rotations occur in only 4 links. In the copy-heavy classes (wind turbine, oil tank, harbor,
   airport), the baseline's random H/V flips mostly re-create images that already exist. For bridge and ship, which
   have few copies, they are still new views.
3. **The boxes of copies are shifted.** In a flipped copy, every box is 1 px off (512² images) or 2 px off (256²)
   along each flipped axis, compared with the exact flip of the other copy's boxes. The two annotations of one object
   overlap at a median IoU of 0.67–0.80 (512²) and 0.50–0.69 (256²). This is label noise on the smallest objects, and
   it caps AP at high IoU thresholds. 292 of the 4,238 copy pairs also differ in box count; in the pair I inspected,
   boxes at the image border were missing from one copy.
4. **Tiny objects decide mAP.** COCO mAP is the mean over the 9 classes. Four of them are almost all COCO-small in
   original pixels: wind turbine (100 %, median 8.3 px at 512, 45.5 % below 8 px, max 21 px), playground (99.9 %,
   14.8 px), bridge (95.2 %, 9.9 px) and oil tank (92.3 %, 11.5 px). Over all boxes: 18.4 % below 8 px, 50.5 % below
   16 px and 62.4 % below 32 px at 512. In original pixels: 69.7 % small, 25.1 % medium, 5.1 % large.
5. **There are two resolutions.** 74.5 % of images are 512², 24.7 % are 256² (upsampled ×2, which makes them 11×
   blurrier at the input by Laplacian variance) and 0.6 % are 224² (×2.29). The 17 odd-sized images (all ships) are
   stretched by ×1.01–1.88 in x and ×1.34–2.39 in y. Ships are 97.5 % 256², so a ship of median 28 px in the file is
   55 px at the input. Harbor, oil tank and bridge come in both resolutions.
6. **Box shapes don't all fit the anchors (1:2, 1:1, 2:1).** 31 % of airport boxes are more elongated than 1:3 and
   20 % more than 1:4 (median 146 px, 32 % touch the border). 13.7 % of ship boxes are beyond 1:3. Every other class
   has ≤ 4.2 % beyond 1:3.
7. **One class per image.** Every box has the image's class, and every class's AP pays for false positives that
   other images produce. Rescoring detections by the classifier is a free inference-time lever (§8).
8. **Grey level tracks class and source.** The mean grey of a class ranges from 0.095 (aircraft) to 0.413
   (playground), and sources within a class differ (oil tank 256² PNG 0.58 vs 256² JPEG 0.12). Wind-turbine scenes
   are flat (std 0.048, p99 84/255) and the turbines are faint (1.69× their surroundings, 38 % below 1.5×). Airport,
   playground and bridge boxes are *darker* than their surroundings in 84 %, 83 % and 68 % of cases. Ships are the
   strongest targets (3.85×).
9. **The holdout rests on few scenes for several classes.** It has 336 independent units in its 940 images:
   airport 7, wind turbine 9, oil tank 12, car 14, aircraft 16, harbor 19. Per-class AP on these classes moves with a
   handful of scenes. Compare designs paired on the same holdout, and expect more spread than the seed std suggests.
   Local train is 8,057 images. A buffer of 395 (87 from the wind-turbine block, 303 from the final leak sweep, 5 from
   the critic's H3 list) is in neither side, and the full-train submission run uses all 9,392. By the lead's decision, holdout ships almost
   exclude the near-black 256² ship chips (2 of 538, §5–6), so holdout ship AP covers the brighter ship chips only.
10. **Sanity checks.** `MEAN, STD = 0.2526, 0.2133` reproduce exactly (0.2526 / 0.2133). Of the 8,701 three-channel
    files, 8,695 have R = G = B; the other 6 differ by ≤ 17 grey levels. Reading one grey channel loses nothing. There
    are no exact duplicate files and no pixel-identical images without a transform.

## 1. Classes and boxes per image

| class | images | img % | boxes | box % | mean boxes/img | median | max | single-box img % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aircraft | 1182 | 12.6 | 2532 | 7.8 | 2.1 | 1 | 21 | 52.1 |
| airport | 650 | 6.9 | 919 | 2.8 | 1.4 | 1 | 5 | 67.8 |
| bridge | 1249 | 13.3 | 3984 | 12.2 | 3.2 | 2 | 23 | 41.7 |
| car | 443 | 4.7 | 4089 | 12.6 | 9.2 | 8 | 28 | 4.5 |
| harbor | 1355 | 14.4 | 2650 | 8.1 | 2.0 | 1 | 15 | 54.2 |
| oil_tank | 1233 | 13.1 | 5580 | 17.2 | 4.5 | 5 | 8 | 10.4 |
| playground | 1175 | 12.5 | 2610 | 8.0 | 2.2 | 1 | 20 | 56.1 |
| ship | 1187 | 12.6 | 1544 | 4.7 | 1.3 | 1 | 8 | 82.4 |
| wind_turbine | 918 | 9.8 | 8625 | 26.5 | 9.4 | 8 | 41 | 10.8 |
| all | 9392 | 100.0 | 32533 | 100.0 | 3.5 | 2 | 41 | 44.7 |

- Images per class vary 3.1× (car 443 to harbor 1,355); boxes per class vary 9.4× (airport 919 to wind turbine
  8,625). Macro-F1 weights classes equally, and so does mAP. Car is the smallest class by images: 44 in the holdout,
  forming 14 independent units. By holdout units, airport (7), wind turbine (9) and oil tank (12) are smaller (§6).
- Oil tank images have at most 8 boxes (median 5, p90 8), which looks like a cap in the source. The maximum over all
  images is 41 ground-truth boxes (wind turbine; car 28). The cap `detections_per_img = 100` (and COCO's maxDets 100)
  limits *predictions*, not the ground truth. Faster R-CNN scores up to 1,000 proposals × 9 class-specific boxes per
  image before the cap. On dense images, low-score duplicates and other-class boxes can fill the 100 slots and push
  out true positives, so the cap can bind there.

## 2. Image sizes, file types and the resize to 512 × 512

| class | 224² | 256² | 512² | other | JPEG grey (L) | JPEG 3-ch | PNG 3-ch |
|---|---:|---:|---:|---:|---:|---:|---:|
| aircraft | 59 | 4 | 1119 | 0 | 0 | 1123 | 59 |
| airport | 0 | 0 | 650 | 0 | 0 | 650 | 0 |
| bridge | 0 | 238 | 1011 | 0 | 0 | 1042 | 207 |
| car | 0 | 0 | 443 | 0 | 443 | 0 | 0 |
| harbor | 0 | 373 | 982 | 0 | 0 | 982 | 373 |
| oil_tank | 0 | 549 | 684 | 0 | 0 | 1115 | 118 |
| playground | 0 | 0 | 1175 | 0 | 0 | 1175 | 0 |
| ship | 0 | 1157 | 13 | 17 | 248 | 939 | 0 |
| wind_turbine | 0 | 0 | 918 | 0 | 0 | 918 | 0 |
| all | 59 | 2321 | 6995 | 17 | 691 | 7944 | 757 |

What the resize does (`v2.Resize((512, 512))`, bilinear with antialias):

| size group | images | scale to 512 | sharpness at input (median Laplacian variance) | speckle CV (median, 8×8 blocks) |
|---|---:|---|---:|---:|
| 512² | 6995 | ×1 (unchanged) | 1283 | 0.19 |
| 256² | 2321 | ×2 upsampled | 117 | 0.59 |
| 224² | 59 | ×2.29 upsampled | 73 | 0.25 |
| other | 17 | ×1.01–1.88 in x, ×1.34–2.39 in y (anisotropy y/x 0.85–2.05) | 332 | 0.49 |

- The odd sizes are 504×328, 500×298, 502×360, 505×332, 507×354, 495×356, 273×322, 500×305, 336×345, 486×283,
  391×285, 506×347, 500×383, 439×296, 503×345, 438×214 and 501×324. All 17 are ships, and the resize squashes or
  stretches the hulls.
- At the input, a 256² image has 11× less high-frequency energy than a 512² one. Its objects are 2× larger in pixels
  but carry no more detail. The size group is also a source cue: ship ≈ 256², car is all 512² grey JPEG.

## 3. Box sizes, at the 512 input and in original pixels

| class | boxes | p10 @512 | median @512 | p90 @512 | max @512 | <8 % | <16 % | <32 % | median orig | orig <8 % | orig <16 % | COCO small % | medium % | large % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aircraft | 2532 | 49.0 | 76.4 | 106.8 | 270.8 | 0.2 | 1.9 | 3.2 | 74.4 | 2.0 | 3.6 | 10.5 | 71.6 | 18.0 |
| airport | 919 | 77.5 | 146.0 | 247.4 | 349.0 | 0.0 | 0.0 | 0.0 | 146.0 | 0.0 | 0.0 | 0.0 | 18.2 | 81.8 |
| bridge | 3984 | 7.0 | 9.9 | 25.7 | 510.0 | 22.8 | 82.7 | 92.6 | 9.8 | 23.1 | 85.9 | 95.2 | 4.0 | 0.8 |
| car | 4089 | 31.0 | 41.4 | 56.4 | 108.0 | 0.0 | 0.0 | 12.3 | 41.4 | 0.0 | 0.0 | 12.3 | 87.7 | 0.0 |
| harbor | 2650 | 23.2 | 42.9 | 216.8 | 447.9 | 0.0 | 1.7 | 29.2 | 41.1 | 0.0 | 2.2 | 31.2 | 52.9 | 15.9 |
| oil_tank | 5580 | 7.0 | 11.5 | 60.9 | 101.0 | 20.3 | 57.5 | 68.8 | 10.7 | 25.5 | 64.2 | 92.3 | 7.7 | 0.0 |
| playground | 2610 | 10.7 | 14.8 | 19.5 | 124.6 | 0.2 | 64.6 | 99.9 | 14.8 | 0.2 | 64.6 | 99.9 | 0.0 | 0.1 |
| ship | 1544 | 31.0 | 55.1 | 98.7 | 342.5 | 0.0 | 0.5 | 11.0 | 28.2 | 0.3 | 9.8 | 59.1 | 40.2 | 0.6 |
| wind_turbine | 8625 | 5.5 | 8.3 | 15.0 | 21.4 | 45.5 | 94.2 | 100.0 | 8.3 | 45.5 | 94.2 | 100.0 | 0.0 | 0.0 |
| all | 32533 | 6.9 | 15.9 | 79.0 | 510.0 | 18.4 | 50.5 | 62.4 | 15.0 | 19.5 | 52.6 | 69.7 | 25.1 | 5.1 |

By size group: 512² boxes have a median of 12.9 px (22.0 % below 8 px, 70.7 % COCO-small). 256² boxes are 55.7 px at
512 but 27.8 px in the file (7.2 % below 8 px and 64.9 % COCO-small in original pixels). COCO small/medium/large uses
the `area` field of the ground truth; it equals w × h for all 32,533 boxes, in original pixels. So a 256² object
that is 16 px at the input counts as an 8 px small object.

Box shapes at the 512 input (aspect = w / h):

| class | beyond 1:2 % | beyond 1:3 % | beyond 1:4 % | short side <4 px % | short side <8 px % | touches border % |
|---|---:|---:|---:|---:|---:|---:|
| aircraft | 1.7 | 0.2 | 0.0 | 0.0 | 0.3 | 15.9 |
| airport | 48.9 | 31.0 | 20.2 | 0.0 | 0.0 | 32.3 |
| bridge | 11.9 | 3.4 | 1.8 | 0.3 | 42.6 | 2.4 |
| car | 16.9 | 0.3 | 0.1 | 0.0 | 0.0 | 1.9 |
| harbor | 14.6 | 4.2 | 0.5 | 0.0 | 0.0 | 12.9 |
| oil_tank | 0.9 | 0.6 | 0.5 | 0.6 | 23.0 | 8.1 |
| playground | 0.2 | 0.1 | 0.0 | 0.0 | 5.4 | 1.9 |
| ship | 35.8 | 13.7 | 5.0 | 0.0 | 0.0 | 12.0 |
| wind_turbine | 0.0 | 0.0 | 0.0 | 0.9 | 49.4 | 2.0 |
| all | 8.2 | 2.4 | 1.2 | 0.4 | 22.7 | 6.4 |

- What the anchors cover: the smallest anchor is 8 px at stride 4. 22.7 % of boxes have a short side under 8 px
  (wind turbine 49 %, bridge 43 %, oil tank 23 %). At RPN `fg_iou_thresh = 0.7`, a 6 px box can't reach IoU 0.7 with
  an 8 px anchor. It reaches only 0.56, and only if perfectly centred. So the tiniest objects get positives only
  through torchvision's "best anchor per GT" rule.
- Label noise at high IoU thresholds: a 1 px shift of an 8 × 8 box gives IoU 0.78, and 2 px gives 0.60 (see §5 for
  the copies). Most of the AP@0.75–0.95 of wind turbines and bridges is out of reach.

## 4. Intensities

Grey levels are 0–255; mean and std are fractions of 255. The speckle CV is the median over 8 × 8 blocks of
std / mean, in original pixels (blocks with mean ≥ 8). In-box and out-box are mean grey inside and outside all boxes.

| class | images | mean | sd of image means | std within image | p1 | p50 | p99 | zero px % | 255 px % | speckle CV | in-box mean | out-box mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aircraft | 1182 | 0.095 | 0.058 | 0.099 | 5.3 | 16.6 | 139.9 | 0.58 | 0.34 | 0.43 | 0.112 | 0.094 |
| airport | 650 | 0.354 | 0.097 | 0.135 | 37.2 | 84.8 | 206.5 | 0.02 | 0.71 | 0.17 | 0.304 | 0.361 |
| bridge | 1249 | 0.370 | 0.115 | 0.147 | 31.8 | 89.2 | 213.2 | 1.23 | 0.80 | 0.19 | 0.337 | 0.370 |
| car | 443 | 0.220 | 0.099 | 0.196 | 1.4 | 41.5 | 237.3 | 1.04 | 1.68 | 0.54 | 0.387 | 0.209 |
| harbor | 1355 | 0.236 | 0.119 | 0.157 | 19.1 | 48.5 | 204.8 | 2.40 | 0.74 | 0.16 | 0.300 | 0.228 |
| oil_tank | 1233 | 0.265 | 0.181 | 0.141 | 20.3 | 59.7 | 193.6 | 11.87 | 0.58 | 0.30 | 0.437 | 0.262 |
| playground | 1175 | 0.413 | 0.112 | 0.159 | 42.6 | 97.7 | 234.2 | 0.02 | 1.41 | 0.20 | 0.380 | 0.413 |
| ship | 1187 | 0.138 | 0.159 | 0.106 | 3.4 | 30.9 | 111.2 | 19.08 | 0.35 | 0.68 | 0.363 | 0.132 |
| wind_turbine | 918 | 0.190 | 0.063 | 0.048 | 33.3 | 46.9 | 83.9 | 0.01 | 0.04 | 0.11 | 0.412 | 0.189 |
| all | 9392 | 0.252 | 0.161 | 0.130 | 21.9 | 57.3 | 178.0 | 4.61 | 0.69 | 0.22 | 0.333 | 0.250 |

By size group: 512² mean 0.272, speckle CV 0.19, 0.7 % zero pixels; 256² mean 0.197, speckle CV 0.59, **16.5 %
zero pixels**. By class × size × file type (likely one source each):

| class, size, file | images | mean | std within | p50 | p99 | zero px % | speckle CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| aircraft 224 PNG | 59 | 0.164 | 0.105 | 35.3 | 167.5 | 0.29 | 0.25 |
| aircraft 512 JPEG | 1119 | 0.091 | 0.099 | 15.6 | 138.1 | 0.58 | 0.43 |
| airport 512 JPEG | 650 | 0.354 | 0.135 | 84.8 | 206.5 | 0.02 | 0.17 |
| bridge 256 PNG | 207 | 0.498 | 0.208 | 126.6 | 246.2 | 4.26 | 0.23 |
| bridge 512 JPEG | 1011 | 0.345 | 0.135 | 81.9 | 206.8 | 0.29 | 0.18 |
| car 512 JPEG-L | 443 | 0.220 | 0.196 | 41.5 | 237.3 | 1.04 | 0.54 |
| harbor 256 PNG | 373 | 0.180 | 0.169 | 33.2 | 214.7 | 7.56 | 0.53 |
| harbor 512 JPEG | 982 | 0.257 | 0.152 | 54.3 | 201.1 | 0.44 | 0.15 |
| oil_tank 256 JPEG | 431 | 0.117 | 0.130 | 19.2 | 165.9 | 26.76 | 1.18 |
| oil_tank 256 PNG | 118 | 0.580 | 0.163 | 147.3 | 243.5 | 0.41 | 0.18 |
| oil_tank 512 JPEG | 684 | 0.304 | 0.144 | 70.1 | 202.4 | 4.46 | 0.19 |
| playground 512 JPEG | 1175 | 0.413 | 0.159 | 97.7 | 234.2 | 0.02 | 0.20 |
| ship 256 JPEG-L | 248 | 0.185 | 0.134 | 42.3 | 158.8 | 3.60 | 0.58 |
| ship 256 JPEG | 909 | 0.123 | 0.099 | 27.5 | 97.6 | 23.79 | 1.10 |
| wind_turbine 512 JPEG | 918 | 0.190 | 0.048 | 46.9 | 83.9 | 0.01 | 0.11 |

(The smaller strata are in the full output: aircraft 256 ×4, bridge 256 JPEG ×31, ship 512 ×13, ship odd ×17.)

Object vs local background: mean grey inside the box vs a ring one box-size wide around it, with other boxes
masked out. Grey levels are 0–255.

| class | median in-box | median ring | median ratio | box darker than ring % | ratio < 1.5 % |
|---|---:|---:|---:|---:|---:|
| aircraft | 29.2 | 20.8 | 1.43 | 19.2 | 54.4 |
| airport | 73.1 | 87.7 | 0.82 | 84.4 | 98.2 |
| bridge | 82.0 | 88.4 | 0.92 | 67.6 | 97.0 |
| car | 98.2 | 46.6 | 2.05 | 1.4 | 21.6 |
| harbor | 70.6 | 63.6 | 1.22 | 28.3 | 70.9 |
| oil_tank | 112.4 | 91.3 | 1.36 | 18.1 | 59.1 |
| playground | 93.6 | 113.9 | 0.81 | 82.9 | 99.7 |
| ship | 83.9 | 20.5 | 3.85 | 0.2 | 11.1 |
| wind_turbine | 102.5 | 62.3 | 1.69 | 14.9 | 37.6 |

What this means for the design:
- **Speckle is low in most of the data.** The 512² sources are smooth (CV 0.11–0.20, consistent with multi-looked
  products). Strong speckle is confined to the 256² ships and oil tanks (CV 0.6–1.2), which are dark scenes with
  24–27 % zero pixels. The zero pixels are spread through the image interior, so they are mostly clipped dark
  water. Some images (55 % of the oil-tank 512² images with > 5 % zeros) also have all-zero rows or columns
  (no-data padding).
- **Brightness alone cannot find airports, playgrounds and bridges.** They are textured regions darker than their
  surroundings, so a "bright blob" prior is wrong for 3 of 9 classes.
- **The classifier can use grey level.** Global grey statistics separate many classes and sources (aircraft 0.09,
  playground 0.41). This is a real property of the data. Since val/test presumably come from the same sources, it
  likely transfers. Avoid per-image normalisation that erases it (§7).

## 5. Near-duplicates and overlapping tiles

**Method** (`near_dups.py`; thresholds in `near_dup_groups.json`):

Below, "high-pass NCC" is the NCC of the image minus its Gaussian blur (σ = 3 px; it keeps speckle and edges). It is
≈ 1 only for the same acquisition. "Band-pass" means a 64 × 64 thumbnail minus its blur (σ = 2 thumbnail px): roads,
coasts and structures, without speckle. Every pair is compared under the 8 flips/rotations where noted. Six stages
of candidates are each confirmed at full resolution:

1. **Unshifted copies and revisits.** Candidates: 32 × 32 thumbnail NCC ≥ 0.6 under the best transform (109,333 of
   the 44 M pairs), plus 119 pairs with identical box lists. They are confirmed at full resolution under that
   transform. A pair is linked as
   - **copy** if high-pass NCC ≥ 0.5;
   - **revisit** if all boxes match at IoU ≥ 0.9 and NCC ≥ 0.3 (single-box pairs also need high-pass NCC ≥ 0.04), or
     if high-pass NCC ≥ 0.15 with box match ≥ 0.5;
   - **related** if the images have the same size and high-pass NCC ≥ 0.15, ≥ 0.1 with box match ≥ 0.3, or NCC ≥ 0.7
     with box match ≥ 0.3.
2. **Shifted tiles with the same boxes** (**overlap**). Boxes of the same class, image size and box size (to 0.5 px)
   vote for an offset. Offsets with ≥ 2 votes (or one box of ≥ 400 px²) are linked if the overlapping pixels agree:
   NCC ≥ 0.9 on ≥ 5 % of the image.
3. **Flipped revisits** (**georevisit**; the critic found these). Under one of the 8 transforms, ≥ 50 % of the boxes
   of both images (and at least 1) match in position within 2.6 px and in size within 0.6 px (512²; 4.6 / 1.1 px for
   smaller images). The band-pass thumbnail NCC under that transform must also be ≥ 0.15 (all pairs: p99 0.08).
4. **Shifted tiles from pixels alone** (**shift**). The same objects are often boxed with different sizes in
   neighbouring tiles, so box votes miss them. Stage 4 takes the cross-correlation of the band-pass thumbnails over
   every shift, for every pair of the same class and square size. Every pair not yet in one group with a peak ≥ 0.35
   gets a full-resolution cross-correlation of the high-passed images, which finds the shift again (best of its 5
   highest peaks). A pair is linked if the overlap is ≥ 5 % of the image and its high-pass NCC ≥ 0.3, or its NCC
   ≥ 0.9. Rules: `same` (high-pass ≥ 0.8, the same acquisition), `partial` (0.3–0.8, the same scene re-processed or
   re-acquired) and `ncc`.

   Evidence that `partial` links are real:
   - Null test: the same statistic on random same-class pairs, 120 per class, reaches ≥ 0.3 for 0 % of 256² ships
     (max 0.28), 0 % of airports and harbors, 1 % of aircraft and 3 % of wind turbines.
   - Visual check of aligned overlaps of disputed links: the same ship shape and sidelobes under other noise or
     banding, the same turbine constellations and static lines.

   I first tried a stricter, class-aware rule, to stop look-alike ship chips from chaining into giant groups. The
   critic's review and this evidence showed it dropped real links, so it was reverted. Giant groups are now handled
   by the split (§6), not by dropping links.
5. **Shifted and flipped copies** (**shift**, rules `*_transformed`; found by my audit). Circular cross-correlation
   of the band-pass thumbnails under the 7 non-identity transforms, for every pair of the same class and square size.
   Candidates need a peak ≥ 0.28 and a thumbnail overlap NCC ≥ 0.7. They are linked by the same full-resolution rule
   under that transform.
6. **Shifted overlaps under the normalised correlation** (**nshift**; the critic's method, `nshift.py`). Stages 4–5
   gate on the raw cross-correlation peak, which shrinks with the overlap fraction, so pixel-identical tiles that
   overlap by only 20–44 % (shifts of 288–380 px at 512²) slipped under the gate. Stage 6:
   - correlates 32 × 32 high-passed thumbnails (σ = 1, zero-padded to 64) and divides by √(ea·eb), the overlap
     energies from the squared images and the masks, so a partial overlap scores like a full one;
   - covers all pairs of the same class and size, under 8 transforms;
   - takes as candidates NCC ≥ 0.5 on ≥ 20 % of the image, or ≥ 0.85 on ≥ 10 %;
   - refines each candidate around its thumbnail shift: ±6 px at 128 px (stop below 0.2), then ±(f + 1) px at full
     resolution;
   - links by the stage-4 rule.

   All 8 pairs that the critic found leaking (e.g. aircraft 4766 ~ 1301 at (0, 288), 100 % equal pixels) are
   detected, at high-pass NCC 0.95–1.00.
7. **Groups** are the connected components of all links. The final groups mix no classes. The thumbnail searches and
   the full-resolution checks of stages 4–6 are cached in `analysis/cache_shift_search.json`; they depend only on the
   images and thresholds.

How the thresholds were set: high-pass NCC is ≤ 0.02 for 97 % of the unshifted candidate pairs (unrelated scenes).
Band-pass NCC over all 44 M pairs is p95 0.057, p99 0.082, p99.9 0.152. A false link costs only some split
flexibility, while a missed one leaks, so the rules lean towards linking. One cross-class "related" link (a
mostly-black 256² ship vs a resized aircraft image, box match 0) was the reason to restrict "related" to same-size
pairs.

**Findings** (links of `near_dups.py`, stages 1–6b, plus the 1,946 crossing links found by `audit_split.py` while the
split was drawn; groups are the connected components of all 40,787 links):

| class | copy | revisit | related | overlap | georevisit | shift | nshift | audit | images in groups | % | groups | independent units | images / unit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aircraft | 52 | 1 | 3 | 1502 | 0 | 1651 | 2629 | 247 | 1155 | 97.7 | 31 | 58 | 20.4 |
| airport | 545 | 577 | 4 | 0 | 12 | 664 | 46 | 3 | 608 | 93.5 | 133 | 175 | 3.7 |
| bridge | 32 | 34 | 6 | 6 | 0 | 73 | 50 | 6 | 254 | 20.3 | 108 | 1103 | 1.1 |
| car | 0 | 2 | 1 | 126 | 0 | 16 | 58 | 6 | 241 | 54.4 | 85 | 287 | 1.5 |
| harbor | 990 | 984 | 142 | 21 | 1 | 1056 | 542 | 129 | 1217 | 89.8 | 233 | 371 | 3.7 |
| oil_tank | 1293 | 1224 | 68 | 835 | 70 | 545 | 282 | 26 | 1211 | 98.2 | 123 | 145 | 8.5 |
| playground | 139 | 119 | 8 | 13 | 22 | 168 | 70 | 9 | 569 | 48.4 | 213 | 819 | 1.4 |
| ship | 0 | 16 | 80 | 12 | 7 | 6697 | 5489 | 145 | 686 | 57.8 | 29 | 530 | 2.2 |
| wind_turbine | 1187 | 683 | 26 | 67 | 75 | 2962 | 4858 | 1375 | 918 | 100.0 | 9 | 9 | 102.0 |
| all | 4238 | 3640 | 338 | 2582 | 187 | 13832 | 14024 | 1946 | 6859 | 73.0 | 964 | 3497 | 2.7 |

- **Copies.** The transforms are identity (1,624: the same picture saved twice, e.g. re-encoded, with ~1.7 % of
  pixels off by 1 level), left-right flip (1,070), up-down flip (1,095) and 180° rotation (445). 90° rotations occur
  in only 4 links. Copies are concentrated in six classes (bridge, harbor, oil_tank, playground, airport,
  wind_turbine), which, to my knowledge, are the categories of M4-SAR.
- **Revisits.** The same site appears with other grey levels (mean abs. difference p10–p90: 10–38 levels, median
  24) and partly different fine structure (high-pass NCC p10–p90: 0.20–0.46, median 0.36; down to 0.04 for
  identical-box pairs with independent speckle), but with the same boxes. That suggests another date, polarisation
  or processing of the same tile.
- **Shifted tiles are the biggest family (13,832 shift links).**
  - Sliding windows of large scenes: 512² tiles overlap at offsets of 312 px (a 200 px strip) and other systematic
    steps (68, 40, 46, 114 px); 256² oil-tank tiles at multiples of 96 px. Aircraft tiles of large airport scenes
    overlap by a median 50 %.
  - The same objects are often boxed with *different sizes* in two overlapping tiles. That is why box votes missed
    them and the pixel search was needed.
  - 6,639 shift links are flipped or rotated as well as shifted (rules `*_transformed`; 4,046 of them are
    partial matches between ship chips).
- **Wind turbine is one family of connected scenes.** All 918 images are in 9 groups (873 in the largest), so the
  whole class holds 9 independent units.
- **Group sizes.** The largest components hold 913 (aircraft), 873 (wind turbine), 627 (ship), 313 (oil tank) and
  132 (harbor) images. The 8,515 `*_inside` links of stage 6b don't change the groups; they complete the links inside
  the three components that could block a stratum (> 50 % of their class), which the buffer needs (§6).
- **The giant ship group is chaining of look-alikes.** It is mostly near-black 256² JPEG chips (median grey 0.010,
  speckle CV 1.23). Random pairs of these dark chips pass the link test (overlap high-pass NCC ≥ 0.3, best of 8 transforms
  and all shifts) in 15 % of cases, vs 0 % for brighter ships (60 random pairs each). For this source the statistic
  can't tell re-processed copies from look-alikes below the same-acquisition level (≥ 0.8). The consequence for the
  holdout is in §6.
- **The boxes of copies are offset.** Box positions after the exact inverse transform, measured on all copy pairs:

  | image size | transform | systematic box offset | box IoU between the two copies (median) |
  |---|---|---|---|
  | 512² | left-right flip | 1 px in x | 0.80 |
  | 512² | up-down flip | 1 px in y | 0.80 |
  | 512² | 180° rotation | 1 px in x and y | 0.67 |
  | 256² | left-right flip | 2 px in x | 0.69 |
  | 256² | up-down flip | 2 px in y | 0.69 |
  | 256² | 180° rotation | 2 px in x and y | 0.50 |

  One copy of each pair is mis-registered by 1 px (512²), or by 2 px (256², which is 4 px at the input). I tried to
  tell which copy is right from the object's intensity centroid, but it is inconclusive: the centroid noise (median
  0.6–2.3 px) is as large as the offset. So every copy pair hands the detector two conflicting regression targets for
  each object.
- **Residual risk of the split** (`audit_split.py`, run on the final split; output in `analysis/audit_split.txt`):
  - No crossing pair passes the link rule under either shifted-overlap detector. Every holdout image was compared
    with every image of its class and size, under 8 transforms and all shifts:
    - the raw-peak search: 5,474 crossing candidates; 96 reach 0.2–0.3, the band random pairs also reach;
    - the normalised search: 50,354 crossing candidates; 8,894 passed the 128 px check; 0 pass the link rule.
  - The critic's 8 leaking pairs all lie on one side.
  - No crossing pair has boxes matching under a flip or rotation together with a band-pass NCC ≥ 0.15.
  - Crossing pairs with a high whole-image band-pass NCC are the MSTAR-style car mosaics. The critic's chip-level
    check found no reused chip.
  - Not checked: crops at a different scale, and copies so re-processed that their high-pass NCC is < 0.3.
- **For the leaderboard (hypothesis, not checked).** If the TA drew val/test at random from the same pool, a large
  share of val/test images would have a copy, revisit or overlapping tile in train: about 0.73 × 0.73 ≈ 53 % (0.73 =
  train's share of the 12,883 released images), close to 100 % for wind turbine, oil tank and aircraft. If instead
  the TA split by source scene, there would be none. In the first case the leaderboard would reward fitting train
  more than our group-aware holdout does, and holdout scores would be pessimistic in those classes. **We did not and
  will not look at val/test pixels to check** (rules). The holdout stays group-aware because it measures
  generalisation, which is what the report needs.

## 6. The local holdout `dataset/holdout_sarmtl/`

**How it was drawn.** Commands, from the repo root:

```bash
uv run python workspace/sarmtl/data/extract_features.py   # per-image/per-box features + thumbnails (~20 s, 12 CPU workers)
uv run python workspace/sarmtl/data/near_dups.py          # links (stages 1-6b) + groups (~1.5 h of CPU the first time;
                                                          # the expensive searches and checks are cached)
uv run python workspace/sarmtl/data/build_holdout.py      # guarded draw + audit loop (<= 5 rounds) -> the draw
uv run python workspace/sarmtl/data/sweep_buffer.py       # FINAL: one sweep of every detector over holdout x local train
uv run python workspace/sarmtl/data/make_holdout.py --fraction 0.1 --seed 0   # applies the sweep, asserts, split.json
uv run python workspace/sarmtl/data/sweep_buffer.py --check                   # must find 0 crossings (it does)
uv run python workspace/sarmtl/data/verify_holdout.py     # SARMultiTask loads every split; COCO scoring check
```

With the tracked files `workspace/sarmtl/data/{near_dup_groups.json, audit_links.csv, frozen_holdout.json,
sweep_buffer.json, critic_h3_buffer.json}`, `make_holdout.py` alone reproduces this exact split.

**The draw** (frozen):
- **Units:** the connected components of all 40,787 links (38,841 from `near_dups.py`, 1,946 from the audit loop):
  964 groups plus 2,533 single images, 3,497 units in all.
- **Strata:** class × image-size group (512 / 256 / 224 / other); no group spans two size groups.
- **Targets:** each class gets round(10 % × its images), shared over its strata by largest remainder.
- **Drawing:** in each stratum, units are taken while they fit the target, in a stable random order. Every image has
  the key sha256(`0-<image_id>`), and a unit has the smallest key of its images.
- **Buffered block (the lead's option A).** A stratum that misses its target by more than 1 image because of a giant
  component takes whole units except the giant, then a BFS ball inside the giant (the seed, of up to 64, with the
  fewest outside neighbours). Those neighbours become a buffer.
  - Only wind_turbine|512 needed it. Its 873-image group gives a 47-image ball, next to 45 images from whole units,
    with a buffer of 87.
- **Frozen.** The lead froze this draw: the 940 holdout ids are in `frozen_holdout.json`, and `make_holdout.py`
  asserts that it draws exactly them. No more redraws.

**The final leak sweep** (`sweep_buffer.py`, the lead's procedure). Every detector runs once over holdout × local
train (same class and size, 8 transforms, all shifts), and every local-train image of a detected crossing pair moves
into the buffer. Removing local-train images can't create a crossing, so one pass terminates. The detectors:
- every near-duplicate and audit link (0 crossing);
- the raw-peak shift search (0 crossing);
- the critic's normalised search, 32 px, NCC ≥ 0.5 on ≥ 10 % overlap: 207,898 crossing candidates, 535 confirmed by
  the link rule;
- a 64 px normalised search for 5–10 % overlaps: 150,171 candidates, 152 confirmed. This detector needs high-pass
  NCC ≥ 0.5. On unrelated pairs (different groups, 80 per class) its best small-overlap match reaches the usual 0.3
  in 10 % of wind-turbine, 2.5 % of harbor and 1.2 % of aircraft cases (max 0.487, never 0.5). At 0.3 it would have
  swept 1,290 images, mostly on chance matches: 79 % of wind-turbine local train. The 32 px search never reaches 0.3
  on unrelated pairs (max 0.253), so it keeps the link rule.
- the critic's 8 verified pairs, as a regression test: all lie on one side, none crossing.

Result: 687 crossing pairs, involving **303 local-train images, now in the buffer**:

| | aircraft | airport | bridge | car | harbor | oil_tank | playground | ship | wind_turbine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| swept | 53 | 0 | 12 | 9 | 59 | 9 | 1 | 37 | 123 |
| share of that class's local train | 5.0 % | 0 % | 1.1 % | 2.3 % | 4.8 % | 0.8 % | 0.1 % | 3.5 % | **16.6 %** |

- Wind turbine exceeds the lead's ~10 % flag. Its local train is now 611 images in 9 units, after H3 below.
- **`sweep_buffer.py --check` on the swept split finds 0 crossing pairs under every detector.**
- The holdout and `halves.json` are unchanged by the sweep.

**The critic's final check (H3, REVIEW.md §5)** ran on the pre-sweep local train.
- It found 13 strong partners: 10 harbor 256² strips with ~20 % overlap at a 205 px shift (high-pass NCC 0.58–1.00),
  and car 7084 for holdout 7226. It also found 20 ambiguous wind-turbine partners (high-pass NCC 0.30–0.45, no
  identical pixels).
- **Why the strips were missed before the sweep:** their overlap is 19.9–20.3 %, right at the 20 % gate of my stage-6
  and audit searches, which below 20 % required thumbnail NCC ≥ 0.85. The sweep used the critic's 10 % gate and had
  already buffered all 13, and 15 of the 20 wind-turbine partners.
- By the lead's decision, both lists go to the buffer (`critic_h3_buffer.json`). That adds the 5 remaining
  wind-turbine images (430, 779, 1439, 5055, 7686).
- Two regression asserts: none of the 33 partners is in local train, and none of the critic's 8 leaking pairs
  crosses.

**Recorded in `split.json`:**
- the SHA-256 of all sources;
- the link and unit counts, and the counts per class and stratum;
- the units per side and per half, and the largest group per class;
- the representativeness note (dark ship chips) and the buffered block;
- `buffer_ball_ids` (87), `buffer_sweep_ids` (303), `buffer_h3_ids` (33, 5 of them new) and `buffer_ids` (395),
  with the reason per id (`buffer_reasons`) and the counts per class;
- the assertions and the 940 holdout ids.

**`halves.json` = {"A": ids, "B": ids}**, for cv-detect's 2-fold cross-fit. Per class, the holdout's units (the
components of the links among holdout images) are shuffled (seed 0) and given largest-first to the smaller half. No
unit straddles the halves, and no link joins them (asserted). The md5s are `val/labels.csv` `9915a68a…`,
`train/labels.csv` `0a6eb4d2…` and `halves.json` `d1b94180…`; a rerun is byte-identical.

Layout, as read by root `train.py --data dataset/holdout_sarmtl` (unchanged):
- `train/` and `val/` each hold `labels.csv`, `instances.json` (with the original train image ids) and
  `images -> ../../train/images`;
- `test -> val`;
- `split.json` and `halves.json`;
- `analysis/`: caches and outputs of these scripts, all train-only.

Output of `make_holdout.py` (verbatim):

```
OK  the drawn holdout equals the frozen holdout (frozen_holdout.json)
OK  the buffer sweep (303 images) removes local-train images only
OK  regression: none of the critic's 33 H3 partners is in local train or holdout (5 added by the H3 list itself)
OK  regression: none of the critic's 8 leaking pairs crosses holdout <-> local train
OK  train and holdout are disjoint (image ids and file names)
OK  the buffer (395 images) is in neither train nor holdout
OK  train + holdout + buffer = all 9,392 train images
OK  no image is listed twice
OK  train: labels.csv rows equal the source rows
OK  train: COCO images equal the source entries of its images
OK  train: boxes equal the boxes of dataset/train/instances.json for its images (27504)
OK  train: categories equal the source categories
OK  train: every class is present
OK  holdout: labels.csv rows equal the source rows
OK  holdout: COCO images equal the source entries of its images
OK  holdout: boxes equal the boxes of dataset/train/instances.json for its images (2715)
OK  holdout: categories equal the source categories
OK  holdout: every class is present
OK  no link of any kind crosses holdout <-> local train (40787 links checked)
OK  only the 1 buffered giant group(s) have images on both sides; the other 963 groups lie wholly on one side
OK  train + holdout + buffer boxes = all 32,533 boxes
OK  halves A (475) and B (465) cover the holdout, are disjoint and no link joins them
buffered block {'stratum': 'wind_turbine|512', 'group_size': 873, 'ball_seed': 4053, 'ball': 47, 'buffer': 87, 'whole_units_images': 45}
class         train img holdout img holdout % buffer train box holdout box  half A/B
aircraft           1011         118      10.0     53      2190         201     59/59
airport             585          65      10.0      0       836          83     36/29
bridge             1112         125      10.0     12      3538         387     63/62
car                 390          44       9.9      9      3694         348     22/22
harbor             1160         136      10.0     59      2300         236     68/68
oil_tank           1101         123      10.0      9      4997         551     61/62
playground         1056         118      10.0      1      2303         305     59/59
ship               1031         119      10.0     37      1322         171     60/59
wind_turbine        611          92      10.0    215      6324         433     47/45
all                8057         940      10.0    395     27504        2715   475/465
-> /home/nhatminh/workspace/cs701-sar-benchmark/dataset/holdout_sarmtl (split.json: 940 holdout ids, 395 buffer ids; halves.json)
```

Loader check (`verify_holdout.py`, CPU, unchanged `sarbench.data.SARMultiTask`, a full pass over every item):
`train: 8057 images, 27504 boxes`, `val: 940 images, 2715 boxes`, `test: 940 images, 2715 boxes`. All items load,
every box stays inside [0, 512] with positive size, and the box labels equal label + 1. The COCO path of root
`train.py` scores the ground truth, fed back as predictions, at `mAP 1.0000, AP50 1.0000`.

How the holdout compares with local train:

| | images | boxes | boxes / img | 512² % | 256² % | median box @512 | <8 px % | <16 px % | COCO small % | mean grey | in groups % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local train | 8057 | 27504 | 3.41 | 74.4 | 24.8 | 16.4 | 19.0 | 49.4 | 68.6 | 0.25 | 71.4 |
| holdout | 940 | 2715 | 2.89 | 74.4 | 24.8 | 22.8 | 13.0 | 44.2 | 68.3 | 0.28 | 77.6 |

The holdout has fewer and larger boxes than local train. Its wind turbines (the ball and a few whole groups) hold 4.7
boxes per image, against 10.4 in local train. mAP is a per-class mean, so this changes the noise more than the class
weights.

Independent units are the connected components of the links within each side:

| class | local-train images | local-train units | holdout images | holdout units | half A / B images | half A / B units | buffer (block + sweep + H3) |
|---|---:|---:|---:|---:|---:|---:|---:|
| aircraft | 1011 | 51 | 118 | 16 | 59 / 59 | 7 / 9 | 53 (0 + 53) |
| airport | 585 | 168 | 65 | 7 | 36 / 29 | 1 / 6 | 0 |
| bridge | 1112 | 1003 | 125 | 94 | 63 / 62 | 47 / 47 | 12 (0 + 12) |
| car | 390 | 271 | 44 | 14 | 22 / 22 | 7 / 7 | 9 (0 + 9) |
| harbor | 1160 | 352 | 136 | 19 | 68 / 68 | 9 / 10 | 59 (0 + 59) |
| oil_tank | 1101 | 134 | 123 | 12 | 61 / 62 | 5 / 7 | 9 (0 + 9) |
| playground | 1056 | 759 | 118 | 60 | 59 / 59 | 30 / 30 | 1 (0 + 1) |
| ship | 1031 | 429 | 119 | 105 | 60 / 59 | 53 / 52 | 37 (0 + 37) |
| wind_turbine | 611 | 9 | 92 | 9 | 47 / 45 | 1 / 8 | 215 (87 + 123 + 5) |
| all | 8057 | 3176 | 940 | 336 | 475 / 465 | | 395 (87 + 303 + 5) |

- **Few holdout scenes for several classes.** Airport (7 holdout units), wind turbine (9), oil tank (12), car (14),
  aircraft (16) and harbor (19) rest on few scenes. Per-class AP there moves with the scenes, beyond the seed std.
  Compare designs paired on the same holdout, and read per-class differences in these classes with care.
  - Half A of airport is a single unit of 36 images, and half A of wind turbine is the ball (one unit of 47).
- **Lead decision (a): the dark 256² ship source is not represented in the holdout.** There are 538 near-black 256²
  3-channel JPEG ship chips (mean grey < 0.05), 45.3 % of all ships. Most form one look-alike-chained group (627
  images, §5): 2 are in the holdout, 508 in local train and the rest in the buffer.
  - **Holdout ship AP covers the brighter ship chips only.**
  - This is leak-safe and recorded in `split.json` → `representativeness`.
- **The buffer only affects local experiments.** Local train loses 395 images (wind turbine 215). **The full-train
  submission run uses all 9,392 images.**
- **Root `train.py` scores val and test, which are the same holdout here.** Read either; they are identical.
- **Checkpoint selection.** Don't pick checkpoints on this holdout: the protocol uses the last epoch. Any
  hyper-parameter tuned on it biases its numbers a little. Keep the tuning budget small and report it.

## 7. SAR preprocessing and augmentation proposals (ranked)

Every proposal here is train-time, except P8, which changes nothing. Inference stays one deterministic pass over the
single image resized to 512 × 512, as in `sarbench/data.py`.

| # | proposal | reason (data fact) | expected effect | cost | rule check |
|---|---|---|---|---|---|
| P1 | **Scale jitter, zoom-in only**: random crop of 62–100 % of the side (zoom 1.0–1.6×), resized to 512. Clip the boxes, drop a box if < 50 % of it stays visible or its side would fall below 2 px. Probability 0.5. | Repeated scenes (1.8 images per unit; 5× for wind turbine and oil tank), so the model sees the same pixels again and again. 50.5 % of boxes are < 16 px and 22.7 % have a short side < 8 px, under the 8 px anchor, so enlarged views give more positives on tiny objects. | +0.5 to +1.5 mAP (ViTDet's large-scale jitter is the precedent; untested here). Classification neutral. | Free (CPU loader); no GPU change. | Train-time only; test input unchanged. OK |
| P2 | **Keep H/V flips; add 90° rotations as an ablation only** | Flips mostly reproduce copies already in train (2,610 flip/180° copy links). 90° rotations are absent (4 links), so they would be new views. Risk: they put SAR shadow and layover along the other axis, a geometry the data may never show. | −0.3 to +0.5 mAP; uncertain sign | Free | Train-time only. OK |
| P3 | **Mild photometric jitter**: gamma 0.8–1.25 and contrast ×0.9–1.1 (both keep 0 at 0), on the uint8 image before normalisation | Grey levels differ by source within a class (oil tank 256² PNG 0.58 vs JPEG 0.12; bridge 256² PNG 0.50 vs 512² 0.35). Jitter makes the features less brittle to scaling. Keep it mild: grey level is a real class cue that likely transfers. | −0.2 to +0.5 F1 and mAP | Free | Train-time only. OK |
| P4 | **Down-up blur**, p = 0.2: on a 512² image, resize to 256² and back | 24.7 % of images (harbor, oil tank and bridge in both resolutions) reach the model 11× blurrier; this teaches the detector both blur levels per class | 0 to +0.5 mAP on the mixed classes | Free | Train-time only. OK |
| P5 | **Ablation at most: harmonise the copy box offset** in local train. For copy pairs, give both copies the boxes of one of them, mapped by the exact transform, or average the two. | Copies disagree by 1 px (512²) or 2 px (256², 4 px at the input) per flipped axis, and the two annotations overlap at IoU 0.50–0.80. But val/test labels very likely follow the same annotation convention, offsets included. "Fixing" train can move predictions away from the evaluation labels, and we can't tell which copy is right. | Uncertain sign: −0.5 to +0.5 mAP (mainly AP75) | Small: a script over `near_dup_links.csv` that writes a corrected local-train `instances.json` | Train labels only. **Never edit the holdout labels** (they must stay as released, like val/test). State it in the report if used |
| P6 | **Dedupe-aware sampling** (weight each image by 1 / its group size), as an ablation | Wind turbine has 918 images but 181 units, oil tank 1,233 but 252. Uniform sampling spends 5× the steps on the same sites. | Uncertain. If val/test share sites with train (§5 hypothesis), down-weighting copies could cost leaderboard points. | Free | OK |
| P7 | Multiplicative speckle noise (Gamma, L = 4–8) on the smooth 512² sources | Speckle CV ranges from 0.11 to 1.18 by source | Likely ≤ 0, as it buries 8 px wind turbines (38 % below 1.5× local contrast) | Free | OK |
| P8 | **Not recommended: despeckling** (Lee, Frost, median) or per-image histogram equalisation as preprocessing | Most 512² sources already have low speckle (CV 0.11–0.20). A 5 × 5–7 × 7 filter smears 45 % of wind turbines (< 8 px) and 23 % of bridges. Equalisation erases the class-grey cue (0.095–0.413). | Negative for small-object mAP, likely negative for F1 | CPU per image at inference too | Would be allowed (deterministic single pass), but not worth it |

Normalisation: keep `MEAN, STD = 0.2526, 0.2133` (recomputed on train: 0.2526 / 0.2133) and one grey channel
(R = G = B in 8,695 of 8,701 three-channel files). Feeding 3 channels adds no information.

## 8. Other data facts for the model designers

- **Class-consistent detection scoring** (for `cv-detect`). Every image has one class and all its boxes share it.
  Multiply each detection's score by the classifier's softmax probability of that box's class, or keep only boxes
  of the predicted class. Both remove cross-class false positives, which cost every class's AP because COCO AP is
  per class over all images. It is one model and one pass, a deterministic post-processing of the model's own heads:
  not TTA, not an ensemble. Evaluate it on the holdout first. Classification accuracy is ~93–94 %, so the boxes of
  the 6–7 % mis-classified images get down-weighted.
- **Objects of other classes are visible but unannotated.** Source: a visual check of contact sheets (12 random 512²
  train images per class, boxes drawn; `contact_sheets.py` → `analysis/contact_<class>.png`). It is not quantified.
  - Harbor scenes: bright ships at sea, unannotated, in about 8 of the 12 (e.g. ids 3828, 8749, 7419, 19, 6605,
    4245).
  - Oil-tank scenes: piers and harbour structures next to the tanks (1347, 2454, 3027), and oval running-track shapes
    like playgrounds (8101).
  - Airport scenes: a ship-like target in the water (6937), and rivers that likely carry bridges (1349).
  - Aircraft tiles: high-resolution aprons, so the whole tile is inside an airport that isn't boxed. There are also
    gate-like bright crosses that are possibly unannotated aircraft (5802, 2222): missing positives of the same
    class, uncertain.
  - Cars: not visible in the ~10 m bridge and playground scenes, where a car is sub-pixel. Car images come from a
    separate high-resolution source.

  So the detector is trained to call real ships, harbours and playgrounds "background" in other classes' images. It
  will still fire on some of them, which makes cross-class false positives, and classifier-based rescoring targets
  exactly those.
- **Anchors.** Aspect ratios of 1:4 and 4:1 would cover airports (20 % beyond 1:4) and ships (5 % beyond 1:4).
  Sizes below 8 px or a lower RPN positive IoU would serve wind turbines, bridges and oil tanks (22.7 % of all
  boxes have a short side < 8 px).
- **Images that touch the border.** 32 % of airport boxes and 16 % of aircraft boxes touch the image border (cut
  tiles), so partial objects are normal.
- **Sources are confounded with classes.** Car is a single source (grey JPEG, 512², very sharp: Laplacian variance
  8,887). **88 of its 443 images are synthetic 4 × 4 mosaics**: 16 vehicle chips of 128 px, MSTAR-style, one box
  centred in each cell. That is a layout cue for the classifier, and it raises thumbnail similarity between unrelated
  mosaics. The critic's chip-level check found no chip reused across the split (max chip high-pass NCC 0.58; copies
  are ≥ 0.9). Ships are 97.5 % 256². A classifier can lean on such cues. They probably hold on val/test, which come
  from the same release.

## 9. Files

| path | what |
|---|---|
| `workspace/sarmtl/data/extract_features.py` | per-image and per-box features and thumbnails → `dataset/holdout_sarmtl/analysis/` |
| `workspace/sarmtl/data/near_dups.py` | near-duplicate links (stages 1–5) and groups → `workspace/sarmtl/data/near_dup_groups.json`, `analysis/near_dup_links.csv` |
| `workspace/sarmtl/data/make_holdout.py` | the split (whole groups + buffered blocks), its assertions, `split.json`, `halves.json` |
| `workspace/sarmtl/data/audit_split.py` | independent leak audit of the split; `--emit-links` appends crossing links to `workspace/sarmtl/data/audit_links.csv` (tracked, 123 links) |
| `workspace/sarmtl/data/build_holdout.py` | guarded draw + audit loop (≤ 5 rounds), then the loader check |
| `workspace/sarmtl/data/sweep_buffer.py` | the final one-pass leak sweep over holdout × local train, and its `--check` |
| `workspace/sarmtl/data/nshift.py` | the normalised shifted-overlap search (the critic's method) |
| `workspace/sarmtl/data/frozen_holdout.json`, `sweep_buffer.json` | the frozen holdout ids; the swept local-train ids with their crossing pairs |
| `workspace/sarmtl/data/critic_h3_buffer.json` | the 33 local-train partners of the critic's H3 strips, which `make_holdout.py` puts in the buffer |
| `workspace/sarmtl/data/verify_holdout.py` | the CPU loader check of all splits and the COCO scoring check |
| `workspace/sarmtl/data/analyze_train.py` | every table in this file |
| `workspace/sarmtl/data/contact_sheets.py` | 12 annotated train images per class → `analysis/contact_<class>.png` (§8) |
| `dataset/holdout_sarmtl/{train,val,test}` | the holdout dataset root (`--data dataset/holdout_sarmtl`) |
| `dataset/holdout_sarmtl/split.json`, `halves.json` | how the split was drawn (ids, buffer, counts, assertions); the group-aware halves of the holdout |
| `dataset/holdout_sarmtl/analysis/` | `train_image_stats.csv`, `train_box_stats.csv`, `train_thumbs.npz`, `near_dup_links.csv`, `cache_*.json` (thumbnail searches and full-resolution checks), `train_analysis.md`, `make_holdout_output.txt`, `audit_iter*.txt`, `audit_split.txt`, `verify_holdout.txt` |
