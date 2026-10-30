# MTDet: three detector families as multi-task models on one harness (`workspace/mtdet/`)

Team `cs701-team05`'s fifth design. It asks which detector family gives the best Δm in the budget: YOLO26, a DETR
(RT-DETR) or a ConvNeXt-B two-stage detector. Each candidate is one model: one shared trunk feeds a 9-class image
classifier and a detector, trained jointly. All three share the training loop, the holdout, the decoding sweep and the
submission writer, so their numbers compare.

**Why these three.** Our posted entry (R50-FPN Faster R-CNN, `workspace/r50fpn`) reaches val mAP 35.6; the leaders reach
~47. On the holdout, four classes have median boxes of 8–15 px (wind turbine 8, bridge 10, oil tank 11, playground 15),
and bridge and playground AP stay at ~1–3 even on training images. The candidates differ exactly where that matters:
- **YOLO26** (anchor-free, small-target-aware assignment STAL, NMS-free one-to-one head): a stride-8 detector with a
  mature fine-tuning recipe; `yolo26l-p2` adds a stride-4 level.
- **RT-DETR-L** (set prediction, no anchors or NMS): the DETR family's real-time member. Plain DETR needs ~500 epochs and
  is weak on small objects, so it is not a candidate.
- **ConvNeXt-B + Cascade R-CNN**: the strongest backbone in the budget on FitDet's best detector (`workspace/fitdet`:
  the cascade gave +7.5 Δm on the same holdout).

## The models

| arch | pretrained (source) | total | trainable | classification head reads |
|---|---|---:|---:|---|
| `yolo26l` | YOLO26-L, COCO (`yolo26l.pt`, github.com/ultralytics/assets v8.4.0) | 26,195,863 | 26,195,863 | backbone end (layer 10, C2PSA, P5) |
| `yolo26l-p2` | same weights; the stride-4 path and its head are new | 25,747,889 | 25,747,889 | layer 10 |
| `yolo26x` | YOLO26-X, COCO (`yolo26x.pt`) | 58,837,815 | 38,740,791 | the neck's P5 (layer 22): backbone layers 6–10 are frozen for the budget |
| `rtdetr-l` | RT-DETR-L, COCO (`rtdetr-l.pt`) | 32,847,108 | 32,847,108 | HGNetv2's last stage (layer 9) |
| `convnext-b` | ConvNeXt-B `convnext_base.fb_in22k_ft_in1k` (timm / Hugging Face; ImageNet-22k → 1k) + torchvision's COCO Faster R-CNN R50-FPN v2 heads | 116,953,333 | 37,855,477 | C5 |

All five pass the budget assertion at startup (≤ 130 M total, ≤ 40 M trainable, heads included). Every pretrained model
is trained on natural images (COCO, ImageNet): none on SAR data.
- **Ultralytics models** (`ultra_model.py`): Ultralytics' own detection head and loss with 9 classes. YOLO26 trains its
  one-to-many and one-to-one heads with ProgLoss; RT-DETR uses Hungarian matching with denoising queries. The input is
  sarbench's 512 × 512 gray mapped back to [0, 1] and repeated as R, G, B, as COCO-trained Ultralytics weights expect.
- **ConvNeXt-B** (`convnext_model.py`): FitDet's model (`frcnn_base.py`, a pinned copy of `workspace/fitdet/fitdet_model.py`)
  with its ResNet-50 replaced: new FPN laterals for ConvNeXt's widths, COCO's FPN output convs, RPN with sarbench's
  small anchors, Cascade R-CNN stages 2–3, bf16 heads. For the budget, the stem, stages 0–1 and the last 9 of stage 2's
  27 blocks train; stage 2's first 18 blocks are frozen; stage 3 is frozen with LoRA (rank 32) on its MLPs.
- **Classification head** everywhere: global average pool → LayerNorm → Linear(9), as the reference head, on the layer
  in the table. `--cls-head pyramid` (Ultralytics models) feeds it instead the mean and max of every map the detection
  head reads (YOLO26-L: P3–P5, 2,560 features, +22.5 k parameters).
- **Precision**: bf16 autocast for the trunks; box decoding, box losses and anchors in fp32 (in bf16, coordinates near
  512 snap to a 2 px grid); RT-DETR's decoder and loss in fp32.

## Training (`train.py`)

- Loss = cross-entropy + the architecture's detection loss (YOLO26's divided by the batch size, so it is a per-batch mean
  like the others).
- AdamW, batch 16, 512 px, sarbench's flips, last partial batch dropped. Linear warmup, cosine decay to 1 % of the lr.
  EMA of the weights (decay 0.9998); the EMA model is the one scored and saved.
- Per-architecture defaults (deliberate changes from the TA protocol, stated): YOLO26 lr 5e-4, weight decay 5e-4, 3
  warmup epochs, gradient clip 10 (Ultralytics' fine-tuning recipe at our batch size); RT-DETR lr 1e-4 with the backbone
  at 1e-5, weight decay 1e-4, clip 0.1 (RT-DETR's recipe); ConvNeXt-B lr 1e-4, weight decay 0.05, one warmup epoch (the
  R50-FPN-MT protocol).
- Flags the screens vary (`train.py --help`): `--epochs` (24 by default; the YOLO26 screens use 100–250), `--no-flip`,
  `--zoom` (scale jitter: `common.Zoom`), `--mosaic` (same-class 2 × 2 mosaic: `common.Mosaic`), `--sar-aug` (extra
  speckle and a random gamma: `common.SARAugment`), `--label-smoothing`, `--cls-weight`, `--cls-head`. ND, LF and FIT are scored every 12 epochs (`--eval-every`) and after the last.
- A checkpoint is written atomically after every epoch, and a run directory holding one resumes; the data order depends
  only on (seed, epoch).
- Up to 2 runs per GPU (the user's rule since 26 Oct, `EXPERIMENTS.md` M0.6), no DDP: comparable runs keep batch 16
  (CLAUDE.md).

## Holdout and selection

- **Holdout: `dataset/holdout_fitdet` (ND)**, reused from FitDet: a stratified random 10 % (class × image size, seed 0)
  of `holdout_sarmtl/train`, 805 images; local train 7,252. Not group-aware on purpose: 66.5 % of ND images have a
  near-duplicate in local train, as val images do in train, so ND rewards what val rewards.
- **Diagnostics**, scored during training: LF (`holdout_sarmtl/val`, 940 leak-free images) and FIT (`dataset/fitdet/fit`,
  373 local-train images).
- **Local Δm** on ND against the leaderboard's val reference (93.31 / 26.13).
- **Decoding** (`evaluate.py`): each run is scored on ND under its architecture's grid (YOLO26: one-to-one top-k, or
  one-to-many + per-class NMS 0.6 / 0.7; RT-DETR: queries with or without NMS; ConvNeXt-B: r50fpn's 19 decodings), each
  with or without JD (scores × classifier probability) and 100 or 300 boxes per image. A configuration's decoding is the
  one with the best seed-mean ND Δm.
- **Upload rule** (CLAUDE.md): per architecture, the configuration with the best 3-seed ND Δm, decoded as submitted, is
  retrained on all of train (`dataset/repro`) and uploaded; it is posted only if its val Δm beats both our posted entry
  and the TA's +4.4.

## Speed (one L40S, batch 16, 512 px, bf16; images/s with data loading, first 2,000 local-train images)

| arch | default | channels_last | GPU h per 24 epochs of local train |
|---|---:|---:|---:|
| `yolo26l` | 148 | 160 | 0.30 |
| `yolo26l-p2` | 104 | 127 | 0.38 |
| `yolo26x` | 103 | 120 | 0.40 |
| `rtdetr-l` | 69 | 74 | 0.66 |
| `convnext-b` (bf16 heads, channels_last) | 30 | — | 1.59 |

channels_last is the default for the Ultralytics models (same math, another memory layout). ConvNeXt-B inherits FitDet's
fast mode (1.63× per step in FitDet's measurement).

## Results and run plan (27 Oct, 11:40 SGT)

The full log, with each hypothesis and the per-class numbers, is `EXPERIMENTS.md`; the uploads are in
`submissions/SUBMISSIONS.md`. Seed 0 throughout; ND Δm with the decoding submitted until 27 Oct 11:30
(`o2m_nms0.7_a0_k100`; the current one adds +0.55 to +1.01, M1.6). A run that shared its GPU counts half its wall hours.

| run | change vs `Y_s0` | GPU h | ND Δm | LF mAP | FIT mAP | decision |
|---|---|---:|---:|---:|---:|---|
| `Y_s0` | YOLO26-L, 100 epochs | 1.37 | +36.48 | 18.2 | 66.5 | the base |
| `YP2_s0` | stride-4 level (`yolo26l-p2`) | 1.72 | +37.26 | 18.2 | 69.7 | not resolved: paired unit bootstrap +0.76, 95 % interval [−1.54, +2.60] |
| `Y150_s0` | 150 epochs | 2.08 | **+40.04** | 18.2 | 75.9 | keep: bootstrap +3.54, [+1.38, +5.39] |
| `YNF_s0` | no flips | 1.39 | +33.11 | 14.2 | 94.0 | discard: bootstrap −3.41, [−7.06, −0.23] |
| `Y250_s0` | 250 epochs | 2.42 | **+43.23** | 16.8 | 85.0 | keep: +3.19 over `Y150_s0`; round 2 trains 250 epochs |
| `YLS_s0` | label smoothing 0.1 | 1.06 | +35.49 | 17.9 | 68.3 | discard: −0.99 (F1 −0.98) |
| `YCP_s0` | pyramid classification head | 1.06 | +33.03 | 17.4 | 60.5 | discard: −3.45 (F1 +1.09, mAP −2.10) |
| `YZ150_s0` | zoom 0.5–1.5, 150 epochs | 1.59 | +35.29 | 20.9 | 58.9 | discard: −4.75 against `Y150_s0` (F1 +1.15, mAP −2.81) |
| `YM150_s0` | same-class mosaic, 150 epochs | 1.58 | +36.62 | 19.9 | 62.1 | −3.42 against `Y150_s0`; retested at 250 epochs |
| `YSA150_s0` | SAR speckle and gamma, 150 epochs | 1.53 | +41.18 | 17.3 | 73.1 | +1.14 against `Y150_s0` (F1 +0.27, mAP +0.52), one seed; retested at 250 epochs |
| `YX_s0` | YOLO26-X, layers 6–10 frozen (58.8 M / 38.7 M) | 1.38 | +36.34 | 18.2 | 65.3 | discard: −0.14 (F1 +1.37, mAP −0.45) at 2.2× the parameters |
| `D_s0` | RT-DETR-L, 72 epochs | 2.07 | +31.72 (its best decoding) | 20.4 | 66.6 | discard: −4.76; stalls after epoch 48, ~3× slower per image |
| `C_s0` | ConvNeXt-B + Cascade R-CNN, 72 epochs (117.0 M / 37.9 M) | ≈ 5.0 | +38.73 (its best decoding) | 17.8 | 84.1 | discard: the best classifier (F1 95.0) but mAP 45.9; YOLO26-L with SAR reaches +45.63 (3 seeds) in half the GPU hours |
| `YSA250_s0` | SAR speckle and gamma, 250 epochs | 2.61 | **+44.18** | 17.3 | 82.2 | keep, pending its seeds: +0.95 over `Y250_s0` (F1 +0.21, mAP +0.44), its second single-seed win |

- **Decoding**: YOLO26's one-to-many head with per-class NMS 0.7 and 100 boxes per image beats its native one-to-one
  head by 1.9–2.7 Δm. With the 0.001 score floor, JD, 300 boxes, NMS 0.8 and merge NMS add nothing or lose
  (M1.1–M1.3). Correcting the image class with detection evidence: rejected (M1.4); so is refitting the classification
  head on the frozen trunk (M1.5). Dropping the floor (every image keeps its top 100 boxes) adds +0.55 to +0.83 Δm on
  ND, and NMS 0.8 then beats 0.7 on all three 250-epoch runs: `o2m_nms0.8_a0_k100_t0` is the submitted decoding since
  27 Oct 11:30 (3-run ND mean +44.12 against +43.27; M1.6).
- **Speed**: channels_last kept. `torch.compile`, fused AdamW, a foreach EMA and dropping the per-step sync gain nothing
  (M0.1, M0.5).
- **Val**: `YA_s0`, `Y_s0`'s configuration trained on all of train, scored +35.81 (F1 92.06, mAP 45.19), ND − 0.67.
  `YA150`, `Y150_s0`'s on all of train, scored **+37.98** (F1 92.38, mAP 46.24), ND − 2.06, and replaced it as the
  posted entry: 5th of 6 on 26 Oct, with the third-best mAP and the lowest macro-F1 of the top five. The leader on
  27 Oct: +41.66 (F1 95.29, mAP 47.35).
  `YA250`, `Y250_s0`'s on all of train, scored **+39.93** (F1 92.43, mAP 47.25), ND − 3.30, and replaced it: 3rd of
  6, its mAP 0.10 below the leader's, so the gap is macro-F1 (2.86 points, ~1.5 Δm). The same model without the score
  floor scored **+41.11** (mAP 47.86, the board's best): 2nd of 6, 0.55 behind team09, all of it F1.
  `YASA250`, `YSA250_s0`'s configuration (SAR augmentation) on all of train, scored **+43.50** (F1 93.40, mAP 48.84)
  at the same decoding: 1st of 6, 1.84 ahead of team09. Its +2.39 over `YA250` is three times what
  ND showed (+0.79, one seed), so the seeds still decide.
  `YASA350` (SAR, 350 epochs, the 3-seed rule's pick) scored **+43.56** (F1 92.46, mAP 49.14) and is posted: **1st
  of 6, 1.27 ahead of team09 (+42.29) at the time. It kept only ~4 % of its holdout gain on val (+0.06 over `YASA250`).
  `YASA450` (450 epochs, the 3-seed rule's pick since 28 Oct 13:04) scored **+44.11** (F1 92.97, mAP 49.28) and is
  posted: **2nd of 6**, 0.33 behind team09 (+44.44, F1 95.81, mAP 48.65). Our mAP is the board's best; the gap is
  macro-F1 (2.84 points, ~1.5 Δm).
  `YASP350` (SAR at probability 0.8, 350 epochs; uploaded at the user's request although the holdout saw no gain from
  0.8) scored **+44.95** (F1 93.59, mAP 49.54): +1.39 over `YASA350` on val. Posted at the user's call: **1st of 6**.
- **Where the score comes from** (`analyze.py`):
  - ND images with a near-duplicate partner in local train score 28–60 mAP; the 270 without one score 16.
  - The 150-epoch gain is all on partnered images.
  - Bridge and playground stay near 4 AP. They are missed, not mislocated, in scenes where look-alike structures carry no
    box.

**Next** (`jobs/plan.txt`): the server rebooted at 09:43 on 27 Oct (`REBOOT.md`); the cut-off runs resumed as `_r2` jobs
from their last checkpoints. Since 27 Oct 13:40 the user's rule is to use the GPUs fully: up to 4 of our jobs per GPU as
memory allows (`gpu_run.sh --slots 4`). GPU 2 runs 3–4 of our jobs (FitDet's queue is down after the reboot); GPUs 0
and 1 held tom's jobs until ~17:10; since then every queued job runs, three per GPU:
1. Done: `YASA250` (SAR, all of train) scored val +43.50 and is posted (1st of 6).
2. Done: round 2's 3-seed means at the submitted decoding: SAR augmentation **+45.63 ± 0.71** against plain
   +43.42 ± 0.72, with SAR ahead on every seed (+0.79, +3.34, +2.50). SAR at 250 epochs is the design's best
   configuration so far, and `YASA250` (posted) is its all-train model.
3. Done: round 3's 3-seed means at the submitted decoding: 350 epochs **+47.23 ± 0.79** against 250 epochs
   +45.63 ± 0.71, with 350 ahead on every seed (+1.88, +0.22, +2.70). `YASA350` (SAR, 350 epochs, all of train) is
   the next upload (~05:30 on 28 Oct). Mosaic at 250 epochs (`YM250_s0`, +42.10 against +44.24): discarded.
4. Done: `C_s0` (ConvNeXt-B + Cascade R-CNN, 72 epochs) scored ND +38.73 (F1 95.03, mAP 45.89): the best classifier,
   but far behind YOLO26-L on mAP. YOLO26-L stays the design's architecture.
5. Done: round 4, SAR augmentation on more images (`--sar-p 0.8`) at 250 epochs: +46.19 ± 0.34 against +45.63 ± 0.71,
   ahead on every seed (+0.56 on average). Round 6 retested it at 350 epochs: +47.44 ± 0.07 against +47.23 ± 0.79,
   paired +0.56, +0.72, −0.64 (mean +0.21): no gain at the longer schedule, so not adopted, and `YASP350` (all of
   train) is not uploaded.
6. Done: round 5's 3-seed means at the submitted decoding: 450 epochs **+48.19 ± 0.95** against 350 epochs
   +47.23 ± 0.79, with 450 ahead on every seed (+0.99, +0.78, +1.11). `YASA450` (SAR, 450 epochs, all of train) is
   the next upload (~16:00 on 28 Oct), posted if it beats +43.56 on val.
7. Round 7 (since 28 Oct 13:06): SAR at 0.8 with 450 epochs at batch 32: `YASPB450` (all of train) runs; its seeds are
   held at the user's request (`YSPB450_s0` ran out of memory at epoch 21 as a 4th job on GPU 1). Round 6 removed its
   premise (0.8 adds nothing at 350 epochs).
8. Round 8 (since 28 Oct 19:33, GPU 2 alone): `YSA450B96_s0`, the best configuration at batch 96, the largest batch a free L40S fits
   (38.8 GB, 188 images/s, ~1.5× one batch-16 run), seed 0; due ~00:45 on 29 Oct.
   `YSA450B80_s1` (since 19:40, GPU 0 beside `YASP350`): seed 1 at batch 80, the largest that fitted there (32.4 GB);
   due ~02:15 on 29 Oct.
   `YSA450B96_s2` (since 23:58, GPU 1 alone): seed 2 at batch 96; with the two above, a 3-seed check of large batches
   against the batch-16 seeds; due ~05:00 on 29 Oct.
   `YSP450B64_s1` (since 29 Oct 01:30, GPU 2 beside another student's job): SAR at 0.8, 450 epochs, seed 1, at batch 64,
   the largest that leaves their job a 4 GB margin; paired with `YSA450B80_s1`; due ~09:00–10:00 on 29 Oct.
   `YSP450B96_s0` (since 29 Oct 01:40, GPU 0 alone): SAR at 0.8, 450 epochs, seed 0, at batch 96; paired with
   `YSA450B96_s0` (0.5, same batch and seed): **+0.69** (+47.93 against +47.24), all from F1.
   `YSP450B96_s1` (GPU 0) and `YSP450B96_s2` (GPU 1), since 29 Oct 08:12 at the user's choice of batch 96: seeds 1 and 2
   of 0.8 at large batch; due ~14:00 on 29 Oct. `YSA450B96_s1` (GPU 2, since 13:05): 0.5 at seed 1 and batch 96, the exact
   pair of `YSP450B96_s1`: +47.20. With exact batch-96 pairs on all three seeds, SAR at 0.8 beats 0.5 by **+1.65**
   (+0.69, +2.20, +2.05).
   **Best configuration since 29 Oct 13:46:** SAR at 0.8, 450 epochs, batch 96, 3-seed holdout **+48.55 ± 0.76** (F1 94.98,
   mAP 51.04), ahead of 0.5 at batch 16 (+48.19) on F1. Its all-train model `YASP450B96` trains on GPU 0 since 13:50,
   due ~21:00; its val zip is the next upload. `YASP450B96_s1` (GPU 1, since 13:54): the same on seed 1, a second
   candidate for Phase 2. Both done: val +44.60 (seed 0) and +43.99 (seed 1), not posted (posting paused).
   `YSF450B96_s0`/`s1` (GPUs 0 and 1, since 29 Oct 22:26): SAR on every image (probability 1.0), 450 epochs, batch 96,
   paired with the 0.8 seeds, and `YSF450B96_s2`: SAR on every image loses to 0.8 on every seed (−0.43, −2.69, −0.61;
   mean −1.24; +47.31 ± 0.54). The strength peaks near 0.8, which stays the best configuration.
9. The Phase 1 upload of the best configuration's all-train model by its 3-seed mean, before 31 Oct 12:00 SGT.

**Batch size for the next runs (the user's rule, 28 Oct 11:00):** each new run uses the largest batch, a multiple of 16,
whose memory fits what its GPU has free, keeping ~3 GB spare. YOLO26-L needs ~0.39 GB per image plus ~0.8 GB: batch
16 = 7.0 GB and 32 = 13.3 GB (measured, M0.8); 48 ≈ 19.5 GB, 64 ≈ 26 GB and 96 ≈ 38.5 GB (extrapolated, checked
at the first launch). The run's `MIN_FREE=` prefix is set to that need, so the queue starts it only on a GPU with the
room. The recipe scales with the batch B: `--lr` and `--backbone-lr` 5e-4 × √(B/16) (the square-root rule for AdamW)
and `--ema` 0.9998^(B/16), the same averaging window in epochs; warmup is already counted in epochs. A run at a new
batch is a new configuration: it is compared only with runs at the same batch. An all-train model of an existing
batch-16 configuration stays at batch 16, so that it matches the seeds that selected it. Measured since (EXPERIMENTS.md round 8, 29 Oct):
the best configuration at batch 80–96 scored −1.10 Δm on the holdout against batch 16, mean of three paired seeds
(−0.66, +0.33, −2.95). One run alone at batch 96 trains ~1.5× faster than one batch-16 run, but three batch-16 runs
sharing a GPU do more work per GPU hour. So a large batch buys a sooner single result at the cost of accuracy.

**Submission checkpoint**: the EMA weights of the chosen configuration's all-train run, `runs/YA<config>/model.pt`, decoded
with `o2m_nms0.8_a0_k100_t0`. Posted now (the user's call, 28 Oct 21:18; re-posted 29 Oct 08:13 after the entry was
changed outside this session, `submissions/SUBMISSIONS.md`): `runs/YASP350` (`s_oK1S2mxVie`, SAR at
probability 0.8, 350 epochs, +44.95, 1st of 6). The holdout's pick, `runs/YASA450` (+44.11), stays uploaded.

## Commands (from the repo root; every GPU command through `workspace/gpu_run.sh`)

```bash
uv run python -m pytest workspace/mtdet/tests -q            # CPU, ~2 min
mkdir -p workspace/mtdet/runs/queue          # the queue (from r50fpn's run.sh): two loops share the plan and its claims
GPUS=1 WORKERS=2 MIN_FREE=10000 nohup setsid workspace/mtdet/run.sh workspace/mtdet/jobs/plan.txt >> workspace/mtdet/runs/queue/run.log 2>&1 &
GPUS=2 WORKERS=1 MIN_FREE=10000 MAX_NEED=12000 nohup setsid workspace/mtdet/run.sh workspace/mtdet/jobs/plan.txt >> workspace/mtdet/runs/queue/run.log 2>&1 &
workspace/mtdet/run.sh --status workspace/mtdet/jobs/plan.txt
uv run python workspace/mtdet/compare.py                     # the table
HF_HUB_OFFLINE=1 workspace/gpu_run.sh --min-free 8000 -- uv run python workspace/mtdet/analyze.py --dump \
    workspace/mtdet/runs/<run>                               # a run's ND outputs (GPU, ~1 min)
uv run python workspace/mtdet/analyze.py workspace/mtdet/runs/Y_s0 workspace/mtdet/runs/<run> --bootstrap 200
                                                             # partner splits and the paired bootstrap (CPU, ~12 min)
HF_HUB_OFFLINE=1 workspace/gpu_run.sh --min-free 8000 -- uv run python workspace/mtdet/predict.py \
    workspace/mtdet/runs/<all-train run> --split val --decoding <chosen> --out workspace/mtdet/submissions/<name>.zip
```

Weights (not in git, `weights/`): `yolo26l.pt`, `yolo26x.pt`, `rtdetr-l.pt` from
`https://github.com/ultralytics/assets/releases/download/v8.4.0/`; ConvNeXt-B through timm's Hugging Face cache.
Dependencies added: `ultralytics==8.4.189` (installed with `--no-deps`, so `opencv-python-headless` keeps providing `cv2`),
plus `ultralytics-thop 2.2.3`, `polars 2.1.0`, `polars-runtime-32 2.1.0`, `psutil 7.2.2`, `nvidia-ml-py 13.615.71`.

## Files

| file | what |
|---|---|
| `common.py` | data, evaluation sets, `dense_decode` (top-k / NMS, JD), Δm, the budget check |
| `ultra_model.py` | YOLO26 and RT-DETR multi-task models, COCO weight transfer, their decodings |
| `convnext_model.py`, `frcnn_base.py` | ConvNeXt-B + Cascade R-CNN on FitDet's pinned model |
| `train.py` | a run: loop, EMA, schedule, checkpoint/resume, ND/LF/FIT scoring, `metrics.json`, `model.pt` |
| `evaluate.py` | the decoding sweep on ND → `decode_eval.json` |
| `predict.py` | the leaderboard zip, compacted and checked with `workspace/sarmtl/check_submission.py` |
| `compare.py` | the results table, per run and per configuration |
| `analyze.py` | failure analysis on ND: per class, by near-duplicate partner and orientation; paired unit bootstrap |
| `EXPERIMENTS.md`, `submissions/SUBMISSIONS.md` | the experiment log; the uploads, their ids and scores |
| `run.sh`, `jobs/` | the queue and its plan |
| `tests/` | budgets, joint forward/backward, every decoding, data order, the zip, the holdout |
