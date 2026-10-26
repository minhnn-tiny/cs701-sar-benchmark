# MTDet experiment log

Every experiment: hypothesis, change, cost, result, decision (KEEP / REJECT / pending). Scores in %. Local Δm is on ND
(`dataset/holdout_fitdet/val`, 805 images) against the leaderboard's val reference (93.31 / 26.13). LF = leak-free
holdout (`holdout_sarmtl/val`, 940 images), FIT = 373 local-train images. One NVIDIA L40S (46 GB) per run; local train
7,252 images; batch 16, 512 px, bf16, flips; per-architecture recipe in README.md. Single-seed screens resolve effects
above about 2 Δm (FitDet's unit bootstrap on ND).

## Before training (CPU, or short GPU benchmarks)

| id | question | method | result | decision |
|---|---|---|---|---|
| M0.1 | Faster steps without changing results? | images/s on 2,000 local-train images, batch 16 (`runs/smoke/speed.sh`) | channels_last: YOLO26-L 148 → 160, -L-P2 104 → 127, -X 103 → 120, RT-DETR-L 69 → 74. `torch.compile` (YOLO26-L, per layer): 109 vs 97 ms per step, recompiles on dynamic shapes, 160 s compile | channels_last KEEP (default for Ultralytics models); compile REJECT; DDP REJECT (batch 16 comparability, shared GPUs) |
| M0.2 | Is the harness sound? | Codex adversarial review #1 | 4 issues: in-train eval sets reported as holdout, resume with changed args, rerun overwriting a finished run, worker-dependent data order | all fixed: overlap check + `held_out`, args/source stored and validated, completed-run guard, (seed, epoch)-seeded sampler and flips; 9 tests pass |
| M0.3 | Would no flips, or added 90° rotations, help val? | near-duplicate links of `holdout_sarmtl` (copy / revisit / shift, with the dihedral transform k) between ND and local train | 535 of 805 ND images have a local-train partner; 491 match it through k ∈ {identity, flips, 180°}; 7 need a 90° rotation. **Corrected 26 Oct:** 372 of the 535 have a partner in the same orientation (k = 0), which flips show 1 epoch in 4; about 110 have only a flipped or 180°-turned one | rot90 REJECT (7 images; r50fpn's rot90 cost 8.3 FIT mAP). No flips was rejected on a misreading of the 491 as flip-only; it is now measured (`YNF_s0`) |
| M0.5 | Is train.py's step CPU-bound (GPU 60–70 % busy)? | `runs/smoke/step_bench.py`: YOLO26-L, 100 preloaded batches, steps 21–100; and the train loader alone on CPU | train.py's step 86.5 ms; without the per-step `isfinite` sync 87.1; plus foreach EMA (identical weights, max diff 0) 86.4; plus fused AdamW 86.3 (and a different loss after 100 steps, 9.95 vs 9.67); train.py again 86.3. The loader gives 1,150 images/s with 8 workers (16: 1,312). In training: 154 images/s, against 185 with preloaded batches | REJECT all three (no gain). The loader isn't the limit; the remaining 17 % is the main process's batch handoff |
| M0.4 | Why are bridge and playground near zero? | same links, per class | partner share on ND: bridge 13 %, playground 36 %; airport 97 %, oil tank 98 %, wind turbine 100 %, harbor 83 % | bridge and playground need small-object generalisation (stride 4: `YP2_s0`), not more memorisation |
| M0.6 | Do two runs per GPU finish more work than one? (26 Oct, user's question) | a short YOLO26-L run (6 epochs on 2,000 local-train images, `runs/smoke/coloc_g{1,2}`) next to YA150 on GPU 1 and next to FitDet's C72_s1 (ConvNeXt Cascade R-CNN) on GPU 2, through `gpu_run.sh --slots 2`; images/s per 50 steps, alone vs together | GPU 1: YA150 154 → 100 images/s (0.65×), the test run 95 (0.62× of 154): **+27 %** work per hour; busy 65 → 80 %. GPU 2: C72_s1 37.4 → 18.9 (0.51×), the test run 106 (0.69×): **+20 %**; busy ~80 → 94 %. No MPS, so the runs take turns on the GPU | KEEP: the user's rule is now up to 2 of our runs per GPU (`gpu_run.sh` default `--slots 2`, highest slot first; two queue loops on `jobs/plan.txt`, 2 jobs on GPU 1 and 1 job ≤ 12 GB on GPU 2 next to FitDet's). A third run doesn't fit safely on GPU 1 next to hanliu's 17.7 GB app and has little idle time left on GPU 2. Each shared run takes ~1.5× (FitDet's ~2×) longer |

## GPU runs

| id | change vs Y_s0 | epochs | seed | GPU h | ND F1 | ND mAP | ND Δm | best decoding: ND Δm | LF mAP | FIT mAP | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| Y_s0 | YOLO26-L (26.2 M, all trainable) | 100 | 0 | 1.37 | 94.34 | 43.90 | +34.57 | `o2m_nms0.7`: **+36.48** (mAP 44.90) | 18.24 | 66.53 | **KEEP, the base.** FitDet's best on ND: CAS_s0 +16.70. Gains: airport +23, harbor +18, oil tank +18, wind turbine +20 AP over CAS; bridge 4.5, playground 3.9. LF peaks at epoch 48 (20.6) and falls to 18.2 while FIT (66.5) and ND keep rising. Training CE ≈ 0 from epoch 60; the trunk's classification gradient vanishes. JD and 100 vs 300 boxes change nothing |
| YP2_s0 | stride-4 level (YOLO26-L-P2, 25.7 M) | 100 | 0 | 1.72 | 94.64 | 43.85 | +34.62 | `o2m_nms0.7`: +37.26 (mAP 45.23) | 18.17 | 69.73 | **not resolved**: +0.78 over Y_s0 at the best decoding; the paired unit bootstrap over ND (667 near-duplicate units, 200 draws) gives +0.76, 95 % interval [−1.54, +2.60], from ND's sampling alone. Small objects gain on FIT (playground 29.9 → 52.4, bridge 13.1 → 18.3 AP) but barely on ND (bridge 4.5 → 4.9, playground 3.9 → 5.8, wind turbine 35.0 → 34.7); ship −4.6. +26 % time, 10.5 GB peak. A candidate for the final seeds, not carried into the screens |
| YA_s0 | Y_s0 on all of train (`dataset/repro`), for upload | 100 | 0 | 1.73 | — | — | — | — | — | — | all of train, so no holdout score. **Val (leaderboard, `o2m_nms0.7`): F1 92.06, mAP 45.19, Δm +35.81**, posted (`submissions/SUBMISSIONS.md`). Val − ND (Y_s0): mAP +0.29, F1 −2.28 |
| Y150_s0 | 150 epochs | 150 | 0 | 2.08 | 94.27 | 45.71 | +37.97 | `o2m_nms0.7`: **+40.04** (mAP 46.79) | 18.15 | 75.92 | **KEEP the longer schedule**: +3.56 over Y_s0 at the submitted decoding (mAP +1.89, F1 −0.07); the paired unit bootstrap over ND gives +3.54, 95 % interval [+1.38, +5.39], so ND's sampling doesn't explain it (seeds still to come). The gain is fitting: FIT +9.4, LF −0.1. ND per class (o2o): wind turbine +8.9, oil tank +3.1, playground +2.7, harbor +2.6, ship −2.4, car −1.2. At its epoch 96, LR still high, ND was already +35.85 (Y_s0's end: +34.57); the last 30 epochs add +0.5. LF F1 ends at 83.2 (Y_s0: 81.2), so Y_s0's LF F1 decline after epoch 48 doesn't repeat. 7.0 GB peak. Next: its all-train model (YA150) and a third point (Y250_s0) |
| YA150 | Y150_s0 on all of train, for upload | 150 | 0 | 2.56 | — | — | — | — | — | — | all of train, so no holdout score. **Val (leaderboard, `o2m_nms0.7`): F1 92.38, mAP 46.24, Δm +37.98**, posted (`submissions/SUBMISSIONS.md`). Val − ND (Y150_s0): mAP −0.55, F1 −1.89, Δm −2.06. Against YA_s0 on val: Δm +2.17 (mAP +1.05, F1 +0.32), so val confirms about 60 % of ND's +3.56 schedule gain. Its last ~4 min shared GPU 1 with the M0.6 test |
| Y250_s0 | 250 epochs | 250 | 0 | | | | | | | | does ND keep rising with FIT past 150? |
| YNF_s0 | no flips | 100 | 0 | 1.39 | 90.88 | 42.60 | +30.22 | `o2m_nms0.7`: +33.11 (mAP 44.11) | 14.21 | 93.98 | **DISCARD, keep both flips**: −3.37 at the submitted decoding (F1 −3.46, mAP −0.79); paired unit bootstrap −3.41, 95 % interval [−7.06, −0.23]. Without flips the model memorises local train (FIT 66.5 → 94.0) and generalises less (LF 18.2 → 14.2; ND peaks at epoch 72, +31.47 o2o). Partners in the same orientation gain only +1.3 mAP, turned ones lose 19.2, and accuracy falls on every partner kind (same 97.7 → 92.4): the flips also regularise the classifier (failure analysis) |
| YLS_s0 | label smoothing 0.1 on the classification cross-entropy | 100 | 0 | | | | | | | | YA's val F1 is 2.3 below ND's; LF F1 falls after epoch 48; ND's errors are confident (failure analysis) |
| YCP_s0 | classifier on the mean and max of P3, P4, P5 (`--cls-head pyramid`, +22.5 k parameters) | 100 | 0 | | | | | | | | the default head averages one stride-32 map, where a 10 px bridge covers a fraction of a cell; ND's 270 unpartnered images are 85.9 % accurate |
| YX_s0 | YOLO26-X, backbone layers 6–10 frozen (58.8 M / 38.7 M) | 100 | 0 | | | | | | | | |
| D_s0 | RT-DETR-L (32.8 M) | 72 | 0 | | | | | | | | |
| C_s0 | ConvNeXt-B + Cascade R-CNN (117.0 M / 37.9 M) | 72 | 0 | | | | | | | | 72, not 24 epochs: FitDet's R50 cascade scores ND +34.80 at 72 epochs and +16.70 at 24 (26 Oct), so 24 would measure the schedule |

ND classification (Y_s0): 51 errors in 805; 23 are bridge ↔ playground and 4 are bridge → airport.

## Failure analysis (`analyze.py`, ND, `o2m_nms0.7`)

By each ND image's strongest near-duplicate partner in local train. Accuracy, then mAP:

| partner | images | Y_s0 accuracy | YP2_s0 accuracy | Y_s0 mAP | YP2_s0 mAP |
|---|---:|---:|---:|---:|---:|
| copy (the same image, maybe flipped or turned) | 223 | 99.1 | 98.7 | 59.6 | 61.7 |
| crop (shifted or overlapping tile) | 258 | 97.7 | 96.5 | 39.7 | 40.4 |
| revisit (another acquisition) | 52 | 90.4 | 94.2 | 28.2 | 30.5 |
| none | 270 | 85.9 | 87.4 | 16.0 | 16.3 |

- P2's +0.33 ND mAP comes from the partnered images (copy +2.1, revisit +2.3), not from the 270 without one (+0.3).
- Classification errors (YP2_s0: 49) are mostly on images with no partner: 11 playground → bridge, 7 bridge →
  playground, 3 playground → harbor.
  - The none images are 97 bridge, 68 playground, 55 ship, 22 car and 20 harbor.
  - The wrong predictions are confident, with a median top probability of 0.99.
- Val's F1 falls 2.3 below ND's while its mAP holds, so val seems to have more of these unpartnered scenes.

By orientation (any partner kind, `analyze.py`): same = a partner with k = 0, the only orientation that training without
flips sees; turned = partners only flipped or turned; unknown = only overlap links, which have no k. Accuracy / mAP:

| orientation | images | Y_s0 | YNF_s0 (no flips) | Y150_s0 (150 epochs) |
|---|---:|---:|---:|---:|
| same | 434 | 97.7 / 50.6 | 92.4 / 51.9 | 97.0 / 55.6 |
| turned | 64 | 96.9 / 43.0 | 93.8 / 23.8 | 100.0 / 46.5 |
| unknown | 37 | 97.3 / 46.2 | 97.3 / 55.0 | 97.3 / 44.8 |
| none | 270 | 85.9 / 16.0 | 83.3 / 14.7 | 85.6 / 15.5 |

- No flips buys little where it should help: +1.3 mAP on the 434 same-orientation images, for −19.2 on the 64 turned
  ones and lower accuracy everywhere.
- 150 epochs gains where partners exist (copy 59.6 → 66.8, crop 39.7 → 42.3, revisit 28.2 → 29.8 mAP) and nothing on
  the 270 without one (16.0 → 15.5), which matches FIT +9.4 with LF flat.

Per class and IoU (Y_s0, ND, `o2m_nms0.7`; GT sizes from train's boxes at the 512 input):

| class | median box side (px) | AP50 | AP75 | AP | recall at IoU 0.5 |
|---|---:|---:|---:|---:|---:|
| bridge | 9.9 | 11.3 | 2.8 | 4.5 | 40.7 |
| playground | 14.8 | 8.9 | 4.4 | 4.5 | 42.0 |
| wind turbine | 8.3 | 85.5 | 20.3 | 36.2 | 94.1 |
| oil tank | 11.5 | 85.8 | 60.9 | 56.0 | 89.5 |
| ship | 55.1 | 78.3 | 37.8 | 40.6 | 89.2 |

- Bridge and playground are missed, not mislocated: under half are found even at IoU 0.5.
  - Their boxes are as small as wind turbines' and oil tanks', so size alone doesn't explain it.
  - In 12 train images drawn at random, each box sits in a wide, cluttered scene where similar-looking structures (other
    river crossings, other bright blocks) carry no box.
  - Even on training images (FIT) they reach only 25 / 59 AP after 150 epochs (Y150_s0).
  - So their AP comes from fitting near-duplicate partners: copy 36.8 / 20.2, none 2.5 / 0.4 (Y_s0).
- Wind turbine is found but mislocated: AP75 is 20.3 at 8 px. P2 lifts it to 24.4 (YP2_s0), and 150 epochs add +8.9 AP
  at o2o.

## Decodings (evaluate.py, no training)

| id | question | result (ND Δm) | decision |
|---|---|---|---|
| M1.1 | YOLO26: one-to-one head (native, NMS-free) or one-to-many head + per-class NMS? | Y_s0: o2o +34.57, o2m NMS 0.6 / 0.7 +36.22 / **+36.48**; YP2_s0: +34.57 vs +36.95 / **+37.26** | o2m + NMS 0.7 KEEP (`o2m_nms0.7_a0_k100`, the submitted decoding) |
| M1.2 | JD (box score × image-class probability, α = 1) and 300 vs 100 boxes per image? | within ±0.1 on both runs | neither: α 0, 100 boxes |
| M1.4 | Can detection evidence fix the image class? Class = argmax log p + β log(m + ε), with m the detector's best score per class | β ∈ 0.1–3, ε ∈ {0.001, 0.01, 0.05}, on `outputs_nd.json`: Y_s0 F1 94.34 → at best 95.21 (β 2); YP2_s0 94.64 → 94.89. Split-half cross-fit over units, 50 halvings: Y_s0 +0.48 F1 (positive 93/100), YP2_s0 −0.10 (30/100) | REJECT: ≤ +0.3 Δm and not on both runs. The classifier is confidently wrong, so a box score rarely flips it |
| M1.3 | NMS IoU 0.8, and merge NMS (YOLOv5: each kept box becomes the score-weighted mean of the same-class boxes above the IoU)? | YP2_s0: NMS 0.8 +37.25 (= 0.7). Merge at 0.6 / 0.7 / 0.8: +31.00 / +31.57 / +31.42, i.e. −5.7 at each IoU; AP50 also falls (70.3 → 68.5). Y_s0 (`Ydec_s0`): NMS 0.8 +36.35, merge +30.71 / +31.35 / +31.08 | NMS 0.8 adds nothing; merge REJECT: STAL ranks the best-localised box first, and averaging in its neighbours moves it, which costs most on 8–15 px boxes |
