# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CS701 (Deep Learning and Vision, SMU, AY2026-27 T1) team course project: **multi-task learning on SAR images**.
One model, one shared backbone, two outputs per image: a **class** (9 classes) and **object boxes** (detection).
This repo started as the TA's reference code (ViT-B/16 + Faster R-CNN); our job is to beat the reference on the
Codabench leaderboard within the rules below. The briefing slides and video are in `introduction/`;
`README.md` and `docs/DETAILS.md` hold the full reference results and design notes.

## Competition rules (hard constraints, check every change against them)

- **≤ 130 M total parameters** (frozen ones count) and **≤ 40 M trainable** (heads included). The heads alone are
  20.1 M; ViT + LoRA is 21.0 M trainable. `--adapt full` and `--init scratch` (106 M trainable) are **over budget**:
  references only, never a valid submission. `train.py` logs both counts; the report must state them.
- One model: no ensembles, no test-time augmentation.
- Input ≤ 512 × 512 px, each image fed once: no upscaling, no tiling.
- Train only on the train split. No other SAR datasets (or models trained on them); generic pretrained models
  (ImageNet, CLIP, TerraMind, …) are fine. Never train on or hand-label val/test images.
- Efficiency counts in grading: a similar Δm with fewer trainable params / less compute earns credit.

## Metric

`Δm = 100% × ½ [(F1 − F1_ref)/F1_ref + (mAP − mAP_ref)/mAP_ref]`, macro-F1 for classification, COCO mAP
(IoU .50:.95) for detection. Reference = ViT + LoRA, mean of 3 seeds: **val 93.31 / 26.13, test 93.99 / 25.41**.
Best in-budget reference: TerraMind + MoE-LoRA, Δm ≈ +3.8 (test); full fine-tuning (over budget) ≈ +13.5 is the
upper bound. Seed-to-seed noise is a few tenths of a point; gains below ~2 should be reported as mean ± std.

## Data (`dataset/`, downloaded and unpacked)

The course data (HF `doem1997/cs701-sar-course-data`) is in `<repo>/dataset/`, about 1.9 GB including the
archives. **It is neither tracked by git nor listed in `.gitignore`: never commit it.** Licence CC BY-NC 4.0. The
images come from SARDet-100K, RSAR, M4-SAR and HRSID; the report should cite them (`dataset/SOURCES_AND_LICENSES.md`).

```
dataset/
├── classes.json                label 0–8 ↔ category_id 1–9 ↔ name
├── train/labels.csv            image_id,file_name,label   (9,392 rows)
├── train/instances.json        COCO: images {id, file_name, width, height}, annotations {bbox [x,y,w,h] in original px}
├── train/images/               8,635 .jpg + 757 .png
├── val/images.json             {"images": [...], "categories": [...]}: 1,426 images, no annotations
├── val/images/
├── sample_submission_val.zip   classification.csv (every label 0) + empty detection.json: upload once to test
└── train.tar.gz, val.tar.gz, .cache/   the downloaded archives and the HF download cache
```

- **Test split is not here yet.** It arrives when Phase 2 opens (2 Nov 2026, 00:01 SGT) as `test.tar.gz` +
  `sample_submission_test.zip`. Run `hf download doem1997/cs701-sar-course-data --repo-type dataset --local-dir dataset`
  again, then `tar -xzf test.tar.gz` inside `dataset/`.
- `file_name` is relative to the split folder (`images/00001.jpg`). **`image_id` is numbered per split** (train
  1–9,392, val 1–1,426), so ids overlap across splits. Submissions use the val/test `id`, not the file name.
- Verified on train: CSV ids and file names match `instances.json` exactly. 32,533 boxes, 1–41 per image (45 % of
  images have one). Every box has `category_id == label + 1`. No out-of-bounds or crowd boxes; the smallest box is
  1 × 1 px. Half the boxes are smaller than one 16 px ViT patch at 512 px input (median 15.9 px).
- Image sizes: train 6,995 at 512², 2,321 at 256², 59 at 224², 17 odd non-square ones. Val 1,067 / 349 / 6 / 4 others.
  Images are 8-bit grey, but many store it in three channels: read with `Image.open(p).convert('L')`.
- Images per class (train): aircraft 1,182 · airport 650 · bridge 1,249 · car 443 · harbor 1,355 · oil_tank 1,233 ·
  playground 1,175 · ship 1,187 · wind_turbine 918. Boxes per class vary 9× (airport 919 to wind_turbine 8,625).
- **The code does not read this layout yet.** `sarbench/data.py` expects TA Zichen's layout:
  `classification/{split}.csv` (column `image`) + `detection/instances_{split}.json`, with labelled val/test, under
  default `--data ../dataset/SARFact-Course-20K` (outside the repo). The tests use the same path. To adapt it:
  - point it at `dataset/`;
  - train from `train/labels.csv` + `train/instances.json`;
  - hold out a fixed, saved part of train for local validation (`train.py`'s val/test scoring moves there; the
    report must say how train was split);
  - run inference only (no metrics) on `val/images.json` / `test/images.json` to build submissions.

  The `MEAN, STD` constants were computed on all 9,392 train images and stay valid.

## Commands

```bash
conda create -n cs701bench python=3.12 -y && conda activate cs701bench
pip install -r requirements.txt          # pinned CUDA 12.8 torch 2.9.0 / torchvision 0.24.0

python train.py --backbone vit --init pretrained --adapt lora    # the Δm reference (~50 min on the TA's RTX PRO 6000)
python train.py -h                                               # all options
bash run_all.sh                                                  # all 8 reference configs in sequence on one GPU (~7 h there)
bash run_all.sh --limit 64 --epochs 1 --out runs/smoke           # quick end-to-end smoke test of all configs

python -m pytest tests                                           # needs a GPU
python -m pytest tests/test_adapters.py::<test_name>             # single test
```

- `--limit N` uses the first N images of every split; use it with `--epochs 1` for fast checks.
- `test_backbones.py` / `test_adapters.py` need no data. `test_data.py`, `test_metrics.py`, `test_model.py` read
  TA Zichen's layout at `../dataset/SARFact-Course-20K` (which doesn't exist here), so they need the same
  adaptation to `dataset/` as the code.
- Pretrained weights download from Hugging Face on first use (ViT 0.4 GB, TerraMind 1.5 GB).
- Changing `--batch-size` changes the protocol.

## Compute (our machine)

- **3× NVIDIA L40S, 46 GB each** (Ada, sm_89, bf16 supported), 128 CPU cores, 251 GB RAM. Driver 550.163
  (CUDA 12.4 driver). A cu126 torch build is known to work on this machine. The pinned cu128 wheels in
  `requirements.txt` (meant for Blackwell) have not been tested here yet; if they fail, install torch 2.9.0 /
  torchvision 0.24.0 from the cu126 index instead.
- The time and memory figures in README/DETAILS are for one RTX PRO 6000 Blackwell. Expect different (likely longer)
  times on an L40S and measure them. Every reference config needs 11–17 GiB, so each fits on one L40S with plenty of
  room, TerraMind + MoE-LoRA included.
- `train.py` uses one GPU (plain `.cuda()`, no DDP). Use the three GPUs for **three independent runs at once**, one
  per GPU, e.g. the 3 seeds the report needs. Don't use DDP for comparable runs: it changes the effective batch size.
- **Run directories don't include the seed** (`runs/<backbone>_<init>_<adapt>/`), so parallel runs of the same
  config must get different `--out` values or they overwrite each other:

```bash
mkdir -p runs
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s nohup python train.py --backbone vit --init pretrained --adapt lora \
      --seed $s --out runs/seed$s > runs/seed$s.out 2>&1 &
done
```

- Check `nvidia-smi` before launching, because other processes may already be using some memory. Default
  `--workers 8` per run is fine with 3 runs on 128 cores.
- For the report: GPU type/count = 3× L40S 46 GB. Keep track of GPU hours for every run, tuning included
  (`train_hours` in each `metrics.json`), for the whole-project total.

## Architecture

```
image [B,1,512,512] -> backbone -> tokens [B,1024,768]   (32×32 patches)
  classification: mean over tokens -> LayerNorm -> Linear(768, 9)
  detection:      tokens reshaped to [B,768,32,32] -> SimpleFeaturePyramid (ViTDet neck, strides 4–64, 256 ch)
                  -> torchvision RPN + RoIHeads (Faster R-CNN, 9 classes + background)
loss = cross-entropy + the four Faster R-CNN losses (unweighted sum)
```

- `train.py` builds the backbone (`sarbench/backbones.py`), optionally wraps it with adapters
  (`sarbench/adapters.py`, which also freezes it) **before** `.cuda()`, then wraps it in `MultiTaskModel`
  (`sarbench/model.py`). Two AdamW param groups: trainable backbone params at `--backbone-lr`, everything else at
  `--lr`. 1 epoch linear warmup then per-iteration cosine decay; 24 epochs, batch 16, bf16 autocast. Validates
  every `--eval-every` epochs; the reported model is the **last epoch** (no checkpoint selection).
- Backbones must expose `.blocks` (each with `attn.qkv` / `attn.proj` linears, which `add_adapters` wraps),
  `.embed_dim`, and return patch tokens without CLS. `vit` = timm `vit_base_patch16_224.augreg_in21k` with
  `img_size=512, in_chans=1`; `terramind` = TerraMind-1.0-base via terratorch, grey image fed as both VV and VH.
- MoE-LoRA has a router per task; `set_task(module, 0|1)` switches them, and with `task_routing=True` the model
  runs the backbone **twice** per forward (once per task).
- **Precision rule:** RPN and RoIHeads run in fp32 inside `autocast(enabled=False)`. In bf16 the box coder snaps
  coordinates near 512 to a 2 px grid, which destroys tiny boxes. Keep this when changing the detector.
- Small-object anchors: 2 scales per octave from 8 px (stride 4) to 181 px (stride 64), 3 aspect ratios.
- Data: images read as one grey channel via PIL (.bmp), resized (stretched) to 512×512, normalized with SAR
  stats `MEAN, STD = 0.2526, 0.2133`; train-time aug is only h/v flips. Boxes are XYXY on the 512 input;
  `to_original_xywh` maps predictions back to original-pixel COCO `[x, y, w, h]` before scoring/submission.
- Run outputs go to `runs/<backbone>_<init>_<adapt>/` (gitignored): `log.txt`, `metrics.json` (args, param counts,
  `train_hours`, `peak_gpu_memory_gb`, val history, final metrics), `predictions_{val,test}.json`, `model.pt`.

## Submission (Codabench, competition 18271)

`make_submission.py:write_submission(images, preds)` writes `submission.zip` containing:
- `classification.csv`: `image_id,label` (label 0–8), one row per image;
- `detection.json`: list of `{image_id, category_id (1–9), bbox [x, y, w, h] in original pixels, score}`.

`image_id` is the integer id from `images.json`, not the file name. `preds[file_name]` takes boxes as XYXY on the
512 input with classes 0–8; the function rescales and adds 1. One team account (`cs701-teamNN`).
Phase 1 (val): 12 Oct – 1 Nov 2026, 5 submissions/day. Phase 2 (test): 2 – 8 Nov 2026, **2 submissions total**.
Talk 11 Nov 2026; report (≤ 6 pages, CVPR template) due 22 Nov 2026. All deadlines 23:59 SGT.

## Report must state

Total and trainable params; compute (GPU type/count, final training time, total project GPU hours incl. tuning,
peak GPU memory, inference time per image); input size and preprocessing; every pretrained model with source;
external libraries vs. own code; how train was split for validation and how the Phase 2 submission was chosen;
number of seeds; ablations on the local validation split.
