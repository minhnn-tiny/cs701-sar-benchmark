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
  `--label-smoothing`, `--cls-weight`, `--cls-head`. ND, LF and FIT are scored every 12 epochs (`--eval-every`) and after the last.
- A checkpoint is written atomically after every epoch, and a run directory holding one resumes; the data order depends
  only on (seed, epoch).
- One run per GPU, no DDP: comparable runs keep batch 16 (CLAUDE.md), and the GPUs are shared.

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

## Results and run plan (26 Oct, 13:00 SGT)

The full log, with each hypothesis and the per-class numbers, is `EXPERIMENTS.md`; the uploads are in
`submissions/SUBMISSIONS.md`. Seed 0 throughout; ND Δm as submitted (`o2m_nms0.7_a0_k100`).

| run | change vs `Y_s0` | GPU h | ND Δm | LF mAP | FIT mAP | decision |
|---|---|---:|---:|---:|---:|---|
| `Y_s0` | YOLO26-L, 100 epochs | 1.37 | +36.48 | 18.2 | 66.5 | the base |
| `YP2_s0` | stride-4 level (`yolo26l-p2`) | 1.72 | +37.26 | 18.2 | 69.7 | not resolved: paired unit bootstrap +0.76, 95 % interval [−1.54, +2.60] |
| `Y150_s0` | 150 epochs | 2.08 | **+40.04** | 18.2 | 75.9 | keep: bootstrap +3.54, [+1.38, +5.39] |
| `YNF_s0` | no flips | 1.39 | +33.11 | 14.2 | 94.0 | discard: bootstrap −3.41, [−7.06, −0.23] |

- **Decoding**: YOLO26's one-to-many head with per-class NMS 0.7 and 100 boxes per image beats its native one-to-one
  head by 1.9–2.7 Δm. JD, 300 boxes, NMS 0.8 and merge NMS add nothing or lose (M1.1–M1.3). Correcting the image class
  with detection evidence: rejected (M1.4).
- **Speed**: channels_last kept. `torch.compile`, fused AdamW, a foreach EMA and dropping the per-step sync gain nothing
  (M0.1, M0.5).
- **Val**: `YA_s0`, `Y_s0`'s configuration trained on all of train, scored +35.81 (F1 92.06, mAP 45.19), ND − 0.67.
  `YA150`, `Y150_s0`'s on all of train, scored **+37.98** (F1 92.38, mAP 46.24), ND − 2.06, and replaced it as the
  posted entry: 5th of 6 on 26 Oct, with the third-best mAP and the lowest macro-F1 of the top five.
- **Where the score comes from** (`analyze.py`):
  - ND images with a near-duplicate partner in local train score 28–60 mAP; the 270 without one score 16.
  - The 150-epoch gain is all on partnered images.
  - Bridge and playground stay near 4 AP. They are missed, not mislocated, in scenes where look-alike structures carry no
    box.

**Next** (`jobs/plan.txt`): from 26 Oct ~15:30, two jobs at once on GPU 1 and one job of at most 12 GB on GPU 2 next to
FitDet's queue (the user's rule of up to 2 runs per GPU, `EXPERIMENTS.md` M0.6):
1. Screens: `Y250_s0` (the schedule's third point), `YLS_s0` (label smoothing 0.1), `YCP_s0` (the pyramid classification
   head), `YX_s0` (YOLO26-X), `D_s0` (RT-DETR-L, 72 epochs), `C_s0` (ConvNeXt-B + Cascade R-CNN, 72 epochs).
2. The combined winner with seeds 1 and 2, its all-train model and the Phase 1 upload, before 31 Oct 12:00 SGT.

**Submission checkpoint**: the EMA weights of the chosen configuration's all-train run, `runs/YA<config>/model.pt`, decoded
with `o2m_nms0.7_a0_k100`. Posted now: `runs/YA150` (`s_6bBQH7vFS2`, +37.98).

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
