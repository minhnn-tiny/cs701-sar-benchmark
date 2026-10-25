# R50-FPN-MT: a COCO-pretrained Faster R-CNN as the multi-task model (`workspace/r50fpn/`)

Team `cs701-team05`'s second design. The ViT designs (`workspace/sarmtl/`) stop near val mAP 30: a plain ViT gives one
stride-16 map, and half the boxes are smaller than one patch. Even the TA's full fine-tuning (over budget) reaches only
mAP 32.0. The teams ahead of us on 25 Oct sit at mAP 45.5–45.9. This design replaces the backbone with a CNN feature
pyramid that starts at stride 4.

## The model (`r50fpn_model.py`)

- **Pretrained model:** torchvision's `fasterrcnn_resnet50_fpn_v2`, COCO weights `COCO_V1`
  (`fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`, from download.pytorch.org; COCO box mAP 46.7). COCO is natural
  images, so this is a generic pretrained model, not a SAR one.
- **One shared backbone:** ResNet-50 + FPN (with v2's BatchNorm). It feeds both heads:
  - classification: C5 global average pool → LayerNorm → Linear (as the reference head on the ViT tokens);
  - detection: v2's Faster R-CNN, trained jointly. The loss is cross-entropy + the four Faster R-CNN losses.
- **Small objects:**
  - FPN levels P2–P6 (strides 4–64);
  - sarbench's anchors (8 to 181 px, two scales per octave × 3 ratios = 6 per location), with new RPN output convs;
  - COCO's two RPN convs and the four-conv + fc box head are kept.
- **Budget:** **46,796,953 total, 33,725,529 trainable** (asserted at startup; limits 130 M / 40 M).
  - conv1 and layer1 are frozen, torchvision's fine-tuning default;
  - the box head's 12.8 M fc6 is frozen and adapted with sarbench's LoRA (rank 256, scale 1);
  - everything else trains: ResNet layers 2–4, FPN, RPN, box head convs, predictor, classification head.
- **Input:** sarbench/data.py, unchanged:
  - 512 × 512 gray, normalised with the SAR mean/std, as the baseline;
  - the gray channel is repeated for COCO's RGB conv1;
  - no GeneralizedRCNNTransform, so nothing is resized beyond the baseline's geometry;
  - each image is fed once: no TTA, no tiling.
- **Training protocol:** the baseline's, through root `train.py`'s loop:
  - AdamW, lr 1e-4, weight decay 0.05;
  - batch 16, 24 epochs, one warmup epoch, cosine schedule;
  - horizontal and vertical flips.

## Decoding and the upload rule (fixed before any result)

- **The sweep (`evaluate.py`):** each holdout run is scored under 19 decodings (`DECODINGS`):
  - torchvision's reference decoding;
  - torchvision's, JD product and JD posterior, each with score threshold 0.001, 100 or 300 boxes per image, and
    per-class NMS IoU 0.5, 0.6 or 0.7.

  It refuses a run that trained on any evaluated image (resolved image paths), and reference runs scored on another
  split.
- **The choice (`crossfit.py`):** a 2-fold cross-fit over the holdout halves.
  - In each fold, the decoding with the best seed-mean mAP on one half is scored against the reference decoding on the
    other half.
  - If all 6 out-of-fold gains (2 folds × 3 seeds) are positive, the decoding with the best seed-mean mAP on the whole
    holdout is adopted. Otherwise the reference decoding is kept.
  - This validates the selection procedure, not the adopted decoding on its own: the whole-holdout winner can differ
    from both fold winners (Codex review, 25 Oct).
  - Only seeds 0, 1 and 2 of one configuration (same training args) can adopt a decoding. A single-seed screen is
    compared with `H_s0` on the reference decoding.
  - The grid stays as fixed before `H_s0`. Decodings added after seeing it (NMS 0.8, soft-NMS, a higher JD alpha)
    could not be adopted: the only images they did not inform are val's, which are never used for tuning.
- **Upload** (CLAUDE.md, "Submission rule", 25 Oct): once the runs are done, the configuration with the best 3-seed
  holdout Δm, decoded as submitted, is uploaded from its all-train checkpoint (`F_s0` for the base configuration),
  without first comparing it with the TA's baselines. `crossfit.py` still prints the TA's in-budget configurations on
  the same holdout as context: TerraMind + MoE-LoRA +9.9 (JD posterior), ViT + LoRA +0.6.
  - **The user's exception (25 Oct, after `E_s0`): upload 36 epochs.**
    - About 2/3 of val behaves like training images: val mAP ≈ 0.65 × fit + 0.35 × leak-free holdout mAP, fitted on
      4 earlier uploads by another session. Here, fit means mAP on a model's own training images.
    - On 407 such images (`dataset/repro/val` ∩ `holdout_sarmtl/train`), mAP is 36.2 ± 0.3 for H (3 seeds), 42.8 ± 0.6
      for E (36 epochs, 3 seeds) and 27.9 for R (rotations, seeds 0–1). On the holdout, seed 0 alone had R +1.5 Δm
      above H and E 1.1 below.
    - So `R_s1`/`R_s2` were dropped, and the all-train 36-epoch model `FE_s0` is uploaded. By the time of the
      choice (19:20), `R_s1` had run and `R_s2` was 2/3 through, so both complete the record.
    - The rule above agrees: E's 3 seeds score +32.6 ± 1.7 as submitted, against H's +29.3 ± 2.6. `crossfit.py`
      adopted `off_d300_nms0.6` for E (all 6 out-of-fold gains positive), so `FE_s0` is decoded with it.
- **Posting:** only a val score above both the posted entry and the TA's best in-budget val baseline (+4.4).

## Commands (from the repo root; every GPU command through `workspace/gpu_run.sh`)

- **Holdout:** `dataset/holdout_sarmtl/`, as `workspace/sarmtl/README.md` §1 builds it: 8,057 local train / 940
  holdout.
- **Δm reference:** `workspace/sarmtl/runs/ref_s{0,1,2}`, ViT + LoRA on the same holdout (macro-F1 76.92, mAP
  12.52).
- **Tests:** `uv run python -m pytest workspace/r50fpn/tests -q` (CPU, ~1 min).
- **The runs:** `jobs/plan.txt`.
  - `H_s0`–`H_s2`: the holdout seeds, each followed by its sweep.
  - `F_s0`: the submission model, on all 9,392 train images (`dataset/repro`).

  The queue (a copy of `workspace/sarmtl/run.sh` with its own `runs/queue/`):

  ```bash
  mkdir -p workspace/r50fpn/runs/queue
  nohup setsid workspace/r50fpn/run.sh workspace/r50fpn/jobs/plan.txt >> workspace/r50fpn/runs/queue/run.log 2>&1 &
  workspace/r50fpn/run.sh --status workspace/r50fpn/jobs/plan.txt
  ```

- **One run by hand:**

  ```bash
  HF_HUB_OFFLINE=1 workspace/gpu_run.sh -- uv run python workspace/r50fpn/train.py --data dataset/holdout_sarmtl \
      --seed 0 --out workspace/r50fpn/runs/H_s0
  HF_HUB_OFFLINE=1 workspace/gpu_run.sh --min-free 8000 -- uv run python workspace/r50fpn/evaluate.py \
      workspace/r50fpn/runs/H_s0/r50fpn
  uv run python workspace/r50fpn/crossfit.py     # the decoding choice and the bar
  ```

- **The val zip:**

  ```bash
  HF_HUB_OFFLINE=1 workspace/gpu_run.sh --min-free 8000 -- uv run python workspace/r50fpn/predict.py \
      workspace/r50fpn/runs/F_s0/r50fpn --split val --decoding <adopted> --out workspace/r50fpn/submissions/<name>.zip
  ```

  `predict.py`:
  - rounds the boxes (0.01 px, clamped inside the image) and scores (6 digits);
  - compresses the zip, so 300 boxes per image stay under 25 MB;
  - runs `workspace/sarmtl/check_submission.py` at the end.

## Files

| file | what |
|---|---|
| `r50fpn_model.py` | the model, `DECODINGS`, `set_decoding`, the parameter count |
| `train.py` | a run: root `train.py`'s loop, evaluation and schedule; budget asserted; `metrics.json`, `model.pt` |
| `evaluate.py` | the post-processing sweep on the holdout and its halves → `decode_eval.json` |
| `crossfit.py` | the decoding choice and the upload bar |
| `predict.py` | the leaderboard zip, checked |
| `run.sh`, `jobs/plan.txt` | the queue |
| `tests/` | budget, frozen parts, COCO weights, forward, reload, the zip rounding, the cross-fit rule, the guard |
