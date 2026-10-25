# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CS701 (Deep Learning and Vision, SMU SCIS, AY2026-27 Term 1) team assignment, run by the CVML Lab: **multi-task
learning on SAR images**, ranked on the course leaderboard at https://web-production-c241e.up.railway.app/ (it replaced
Codabench on 24 Oct 2026; see Leaderboard phases). Our goal is the best Δm within the rules below. `README.md` and
`docs/DETAILS.md` hold the reference results and design notes.

## The competition

### Task

One model with one shared backbone, trained jointly on both tasks, gives two outputs per SAR image:
- **classification**: the image's class, 1 of 9 (aircraft, airport, bridge, car, harbor, oil_tank, playground,
  ship, wind_turbine);
- **detection**: every object as a box, class and confidence score. All boxes in an image have the image's class.

### SAR images are not photos

SAR (synthetic aperture radar) is an active radar on a satellite or aircraft. It sends its own microwave pulses
sideways and records the echoes, so it works day and night and through clouds. Combining the echoes along the
flight path acts as one long antenna, which gives fine resolution. A pixel is echo strength: 8-bit grey, no colour.
- **Speckle**: grain on uniform surfaces, caused by coherent imaging.
- **Brightness follows geometry and material**: metal and corners are very bright (a ship on calm sea); smooth
  surfaces are dark (water and runways reflect the pulse away); slopes facing the radar are bright, slopes facing
  away are dark.
- **For RGB-pretrained models**: one grey channel (repeat it or adapt the first layer); different statistics
  (normalise with SAR mean/std, not ImageNet's); new patterns (speckle, bright points, shadow) that need fine-tuning.
- **Objects are small and many**: half the boxes are smaller than one 16 px ViT patch at 512 px input.

### Reference model and baselines

- **Model**: a ViT-B/16 backbone (1 × 512 × 512 input → 1,024 tokens × 768) with two heads. Classification:
  mean-pool + LayerNorm + linear → 9 class scores. Detection: ViTDet feature pyramid + Faster R-CNN → boxes.
  Loss = cross-entropy + the Faster R-CNN losses.
- **Backbones**: ViT-B/16 pretrained on ImageNet-21k, and TerraMind, pretrained on Earth-observation data.
- **Four ways to train the backbone**:
  - from scratch and full fine-tuning: all 86 M backbone parameters train, which is over the budget;
  - LoRA rank 16 on the frozen backbone: 0.88 M trainable; this is the **Δm reference**;
  - MoE-LoRA: 4 LoRA experts with a router per task, 1.03 M trainable.

  The heads (20.1 M) always train.
- **Protocol** (the same for every run): 512 px input, AdamW, batch 16, 24 epochs.
- **TA's takeaways**:
  - pretraining matters most (from scratch: Δm −19 to −23);
  - LoRA and MoE-LoRA train ~1 % of the parameters and lose 10–20 % mAP but only 2–3 % macro-F1;
  - the best in-budget run reaches +3.8 and full fine-tuning +13.5, so there is room to grow.

### Leaderboard phases

After a security issue on Codabench, the TA moved the leaderboard to their own site on 24 Oct 2026:
**https://web-production-c241e.up.railway.app/**, with its docs at `/docs` (`#browser` and `#api` cover the two ways
to submit). The address the TA first announced, https://web-staging-a810.up.railway.app/, serves the same board.
Codabench no longer counts, even if it comes back. Its submissions were not carried over, so Phase 1 starts again
here.

| phase | opens (SGT) | closes (SGT) | released at opening | leaderboard | limit |
|---|---|---|---|---|---|
| 1 · validation | 12 Oct 2026, 00:00 | 1 Nov 2026, 23:59 | val images | val, 1,426 images | 5 scored submissions per team per SGT day (refills 00:00 SGT) |
| 2 · test | 2 Nov 2026, 00:01 | 8 Nov 2026, 23:59 | test images | test, 2,065 images | **2 scored submissions per team** |

- Val and test labels are never released: hold out part of train to validate.

- There are no accounts. Each team signs in with its **team token** from the TA. Ours (team `cs701-team05`) is
  `CS701_TOKEN` in `<repo>/.env`, which is gitignored. Never print the token, commit it, or put it in code or logs.
  If it may have leaked, ask the TA to replace it.
- Only scored submissions use quota. A format error (HTTP 422) never does. Phase 1 has **reset cards**: 2 per team,
  plus any the TA issues; `/api/me` lists them. One card restores the day's 5 submissions. Phase 2 has no reset
  cards, so check every zip with the design's `check_submission.py` before uploading it.
- A submission is **private until it is posted**. "Post to leaderboard" (on the Team page, or
  `POST /api/submissions/{id}/post`) makes it the team's single entry for that phase and replaces the previous one.
  An entry can be withdrawn until the deadline. The board locks at the deadline, and the posted entry counts. If
  nothing is posted, the phase's last successful submission counts and is marked "auto".
- The leaderboard ranks by Δm (ties go to the higher macro-F1, then to the earlier submission). It also shows
  accuracy and AP50.

### Grading, deliverables and timeline

- **Leaderboard 25 %**: Phase 1 (val) 10 %, Phase 2 (test) 15 %.
- **Report + talk 10 %**: a report of at most 6 pages in the CVPR template (what it must state: see the end of this
  file), and a 10-minute talk + 5 minutes of Q&A in class.
- **Bonus +5 %**: interesting extensions beyond the reference methods.
- **Efficiency counts**: within the budget, a similar Δm with fewer trainable parameters or less compute earns credit.
- **Timeline** (every deadline 23:59 SGT, on a Sunday): 30 Sep train data and code released · 5–11 Oct recess ·
  12 Oct Phase 1 opens · 1 Nov Phase 1 closes · 2 Nov Phase 2 opens · 8 Nov Phase 2 closes ·
  Wed 11 Nov talk · 22 Nov report due.

### Where to start (the TA's suggestions)

- **Loss balancing**: the two losses differ in scale and speed, so weight or schedule them (Kendall et al.
  CVPR 2018; GradNorm, ICML 2018; PCGrad, NeurIPS 2020).
- **Efficient tuning**: LoRA and other PEFT; mixtures of LoRA experts with task-specific routing (LoRA, ICLR 2022;
  MTLoRA, CVPR 2024; MLoRE, CVPR 2024).
- **SAR preprocessing**: intensity transforms, despeckling, SAR-aware augmentation (Argenti et al., IEEE GRSM 2013,
  a speckle tutorial).
- **Small objects**: finer feature maps (e.g. a light conv branch), small anchors (ViTDet, ECCV 2022; ViT-Adapter,
  ICLR 2023).


## Competition rules (hard constraints, check every change against them)

The rules exist because the competition ranks methods, not compute, and because other SAR data may contain our
val and test images.
- **Allowed**: any method trained on the train split; efficient adaptation (LoRA, adapters, prompts, partial
  fine-tuning, new necks and heads); generic pretrained models (ImageNet models, CLIP, TerraMind, …), each named in
  the report.
- **≤ 130 M total parameters** (frozen ones count) and **≤ 40 M trainable** (heads included). The heads alone are
  20.1 M; ViT + LoRA is 21.0 M trainable. `--adapt full` and `--init scratch` (106 M trainable) are **over budget**:
  references only, never a valid submission. `train.py` logs both counts; the report must state them.
- One model: no ensembles, no test-time augmentation.
- Input ≤ 512 × 512 px, each image fed once: no upscaling, no tiling.
- **Not allowed**: other SAR datasets, or models trained on them (including detectors pretrained on SAR data: risk
  of leakage); training on or hand-labelling val/test images; sharing work outside the team; submitting with any
  token other than our own team token.
- **Distillation** is not recommended. If used, the teacher must be a frozen, generic pretrained model (not trained
  on SAR data) with ≤ 130 M parameters.
- **The TA reproduces results** and inspects our open-sourced code for fairness and rule compliance, so every
  submitted result must be reproducible from the code (seeds, holdout script, exact commands).

## Metric

`Δm = 100% × ½ [(F1 − F1_ref)/F1_ref + (mAP − mAP_ref)/mAP_ref]` (Maninis et al., CVPR 2019; Vandenhende et al.,
TPAMI 2021): the mean relative change of macro-F1 (the mean F1 of the 9 classes) and COCO mAP (AP averaged over
IoU 0.50–0.95). 0 means as good as the reference; above 0 is better. Reference = ViT + LoRA, mean of 3 seeds:
**val 93.31 / 26.13, test 93.99 / 25.41**; the leaderboard uses these values.
Best in-budget reference: TerraMind + MoE-LoRA, Δm ≈ +3.8 (test); full fine-tuning (over budget) ≈ +13.5 is the
upper bound. Seed-to-seed noise is a few tenths of a point; gains below ~2 should be reported as mean ± std.

## Data (`dataset/`)

The course data (HF `doem1997/cs701-sar-course-data`; its card is `dataset/README.md`) is in `<repo>/dataset/`,
about 1.9 GB including the archives.

| split | images | labels and boxes | files | released (SGT) |
|---|---:|---|---|---|
| train | 9,392 | included (32,533 boxes) | `train.tar.gz` | 30 Sep 2026 |
| val | 1,426 | **withheld** | `val.tar.gz`, `sample_submission_val.zip` | 12 Oct 2026, 00:00 (Phase 1 opens) |
| test | 2,065 | **withheld** | `test.tar.gz`, `sample_submission_test.zip` | 2 Nov 2026, 00:01 (Phase 2 opens) |

- **Val and test labels are never released: hold out part of train to validate.** Each design either reuses an
  existing holdout split or creates a new one, as `dataset/holdout_<design>/`, with a script that rebuilds it and a
  `split.json` recording how it was drawn.
  - Within a design, every run uses that one holdout.
  - Baseline runs are not required. A design picks its best model on its own holdout and doesn't compare it with
    the TA's baselines before uploading (see Submission rule). So a new holdout doesn't need the TA's
    configurations retrained on it. The baseline runs on the two existing holdouts stay available as context.
  - Numbers from different holdouts don't compare. ViT + LoRA scores macro-F1 ≈ 91 / mAP ≈ 22 on
    `dataset/holdout/`, but ≈ 77 / 12.5 on the leak-free `dataset/holdout_sarmtl/`.
  - The design's README names its holdout. The report must say how train was split.

  `train.py` as shipped scores labelled val and test splits, so on `dataset/` it stops at startup
  (`dataset/val/labels.csv` doesn't exist).
- Available holdouts. Each already has 3-seed ViT + LoRA and TerraMind + MoE-LoRA runs (24 epochs):

  | holdout | holdout / local train images | how it was drawn | built by | baseline runs |
  |---|---|---|---|---|
  | `dataset/holdout/` | 939 / 8,453 | 10 %, seed 0, class × image size stratified, near-duplicate groups kept whole | `workspace/detail_lora/make_holdout.py` | `workspace/detail_lora/runs/ref_seed{0,1,2}/` |
  | `dataset/holdout_sarmtl/` | 940 / 8,057 (+ 395 buffer, unused) | 10 %, seed 0, leak-free: near-duplicate components kept whole, a buffer of their neighbours left out of local train | `workspace/sarmtl/data/` (its README §1) | `workspace/sarmtl/runs/{ref,tm}_s{0,1,2}/` |

  `dataset/repro/` is not a holdout. Its train split is all of train, and its val/test are a sample of train.
- `dataset/test/` isn't here yet; the user downloads it when Phase 2 opens.
- Files:
  - `classes.json`: the 9 class names. `label` 0–8 is the class of an image; `category_id = label + 1` is the
    class of a box.
  - `train/labels.csv`: `image_id,file_name,label`, one row per image.
  - `train/instances.json`: the boxes in COCO format, `bbox = [x, y, w, h]` in pixels of the original image.
  - `val/images.json` (and later `test/images.json`): the images to predict (ids, file names, sizes).
  - `{split}/images/`: the images, JPEG or PNG.
- `file_name` is relative to the split folder (`images/00001.jpg`). **`image_id` is numbered per split** (train
  1–9,392, val 1–1,426), so ids overlap across splits. Submissions use the val/test `id`, not the file name.
- Verified on train: CSV ids and file names match `instances.json` exactly. 32,533 boxes, 1–41 per image (45 % of
  images have one). Every box has `category_id == label + 1`. No out-of-bounds or crowd boxes; the smallest box is
  1 × 1 px. Half the boxes are smaller than one 16 px ViT patch at 512 px input (median 15.9 px).
- Image sizes: train 6,995 at 512², 2,321 at 256², 59 at 224², 17 odd non-square ones. Val 1,067 / 349 / 6 / 4 others.
  Images are 8-bit grey, but many store it in three channels: read with `Image.open(p).convert('L')`.
- Images per class (train): aircraft 1,182 · airport 650 · bridge 1,249 · car 443 · harbor 1,355 · oil_tank 1,233 ·
  playground 1,175 · ship 1,187 · wind_turbine 918. Boxes per class vary 9× (airport 919 to wind_turbine 8,625).

  The `MEAN, STD` constants were computed on all 9,392 train images and stay valid.
- **Licence**: CC BY-NC 4.0, for non-commercial research and teaching only. The images and boxes come from
  SARDet-100K, RSAR, M4-SAR and HRSID (reorganised by SARFact) and keep their original terms. The report should
  cite all four (citations in `dataset/SOURCES_AND_LICENSES.md`).

## Workspace (where new work goes)

- Keep the baseline (`sarbench/`, `train.py`, `run_all.sh`, `make_submission.py`) unchanged as the reference.
- Do new work in `workspace/`, **one directory per model design**. `workspace/<design>/` holds:
  - that design's code: new modules, or its own train script that imports the unchanged parts of `sarbench/`;
  - its holdout: the script for a new `dataset/holdout_<design>/`, or the name of the existing holdout it reuses
    (see Data);
  - its runs in `workspace/<design>/runs/` (gitignored, like every `runs/` directory);
  - its submission zips.
- Run scripts from the repo root (`uv run python workspace/<design>/<script>.py`). A script there must put the
  repo root on `sys.path` to import `sarbench` and `make_submission`; see `workspace/baseline/predict.py`.
- `workspace/baseline/` reproduces the reference:
  - `predict.py <run_dir> --split val|test` loads a run's `model.pt` and writes `<run_dir>/submission_{split}.zip`;
  - `make_repro_data.py` builds `dataset/repro/` so the unchanged `train.py` runs: all of train, plus a sample of
    train standing in for val/test;
  - `run_repro.sh` queues the 3 ViT + LoRA seeds; each starts only on a GPU that is healthy, idle and has
    ≥ 25 GB free, with the same checks and shared locks as `workspace/gpu_run.sh`.
- `workspace/gpu_run.sh [--min-free MiB] [--gpus "0 1 2"] [--max-wait s] -- <command>` runs one command on one GPU:
  it waits until a GPU is healthy, has the free memory (default 25,000 MiB; ~8,000 for predict), is idle (< 20 %)
  and isn't locked by our runs (`workspace/runs/gpu_locks/gpu<N>/`), then runs the command with
  `CUDA_VISIBLE_DEVICES=<N>` and releases the lock when it ends. `--status` prints every GPU's state. **Every GPU
  command in `workspace/` (smoke tests, training, predict) goes through it**; a design's queue script calls it per run.
- `workspace/wandb_log.py`: optional W&B tracking for the design train scripts (detail_lora, r50fpn, sarmtl).
  - It is on for runs started while `WANDB_API_KEY` is set in `.env` (with `WANDB_ENTITY` and `WANDB_PROJECT`), and
    off otherwise.
  - It only mirrors root train.py's log lines (the losses and lr every 50 iterations, each validation) and
    metrics.json, and it turns itself off if wandb fails.
  - The W&B key is a secret like the team token: never print it.
  - `uv run python workspace/wandb_log.py` checks the set-up.
  - A new design's train script calls `wandb_log.init(design, run_dir, vars(args), root)` once its log is open, and
    `wandb_log.finish(summary)` after writing metrics.json.


## Python environment (uv)

- **Always run Python through the uv environment** in `<repo>/.venv`: `uv run python train.py ...`,
  `uv run python -m pytest`. Never use the system `python` / `pip`, which don't have the dependencies.
  For `nohup` / background jobs, `.venv/bin/python` works too.
- Installed: torch 2.9.0+cu126 / torchvision 0.24.0+cu126, plus the other pins from `requirements.txt`, and
  wandb 0.31.0 for `workspace/wandb_log.py`. wandb is optional and isn't in `requirements.txt`; timm imports it
  whenever it is installed.


## Compute (our machine)

- **3× NVIDIA L40S, 46 GB each** (Ada, sm_89, bf16 supported), 128 CPU cores, 251 GB RAM. Driver 550.163
  (CUDA 12.4 driver). The cu126 torch build in `.venv` works on this machine (verified: CUDA, bf16, torchvision NMS).
  The pinned cu128 wheels in `requirements.txt` (meant for Blackwell) are not used here.
- The time and memory figures in README/DETAILS are for one RTX PRO 6000 Blackwell. Expect different (likely longer)
  times on an L40S and measure them. Every reference config needs 11–17 GiB, so each fits on one L40S with plenty of
  room, TerraMind + MoE-LoRA included.
- `train.py` uses one GPU (plain `.cuda()`, no DDP). Use the three GPUs for **three independent runs at once**, one
  per GPU, e.g. the 3 seeds the report needs. Don't use DDP for comparable runs: it changes the effective batch size.
- **Run directories don't include the seed** (`train.py` writes to `<out>/<backbone>_<init>_<adapt>/`), so
  parallel runs of the same config must get different `--out` values or they overwrite each other:

```bash
D=workspace/<design>; mkdir -p $D/runs
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s nohup uv run python $D/train.py <args> --seed $s --out $D/runs/seed$s \
      > $D/runs/seed$s.out 2>&1 &
done
```

- **The GPU server is shared by the team.** For example, `tom` runs ~31–35 GB jobs on GPUs 1 and 2, and `hanliu`
  keeps a small ~2–4 GB app on every GPU. **Check availability before every launch**:
  - see memory, utilisation and processes with `nvidia-smi`, and whose jobs they are with
    `ps -o user,etime,args -p <pid>`;
  - only use a GPU nobody else is using;
  - if none is free, wait (a queue that polls `nvidia-smi` is fine) rather than squeezing onto a busy GPU.

  Default `--workers 8` per run is fine with 3 runs on 128 cores.
- **GPU 0 has a hardware memory fault**: uncorrectable ECC errors and a failed row remap, seen 23 Oct 2026. A run
  on it crashed at epoch 18. Don't use GPU 0 until it's repaired.
- **Check GPU health before every launch**, on all GPUs: skip a GPU if `nvidia-smi -i N -q -d ROW_REMAPPER,ECC`
  shows `Remapping Failure Occurred: Yes`, a pending remap, or a volatile `DRAM Uncorrectable` count above 0
  (`gpu_ok` in `workspace/gpu_run.sh`, which does this for you). Free memory and utilisation don't reveal a faulty GPU.
- For the report: GPU type/count = 3× L40S 46 GB. Keep track of GPU hours for every run, tuning included
  (`train_hours` in each `metrics.json`), for the whole-project total.

## Baselines

The TA's results, from [docs/DETAILS.md](docs/DETAILS.md#results-in-full). Scores in %, mean ± sample std over
seeds 0–2, last-epoch model, trained on all of train. Δm is computed per seed against the reference (the mean of
the three ViT + LoRA runs), then averaged, so the reference shows 0. The trainable backbone params exclude the
20.1 M of heads, which always train. Full fine-tuning and from scratch (106 M trainable with the heads) are over
the 40 M budget: references only, with full fine-tuning the upper bound.

Test (2,065 images):

| backbone | init | adapt | trainable backbone params (M) | accuracy | macro-F1 | mAP | AP50 | Δm |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vit | pretrained | full | 86.0 | 95.9 ± 0.3 | 95.8 ± 0.3 | 31.8 ± 0.2 | 62.4 ± 0.1 | +13.5 ± 0.4 |
| vit | pretrained | lora | 0.88 | 94.2 ± 0.2 | 94.0 ± 0.2 | 25.4 ± 0.5 | 53.9 ± 0.4 | 0.0 ± 0.9 |
| vit | pretrained | moelora | 1.03 | 93.7 ± 0.5 | 93.4 ± 0.5 | 26.8 ± 0.3 | 55.9 ± 0.5 | +2.5 ± 0.8 |
| terramind | pretrained | full | 85.3 | 95.8 ± 0.2 | 95.6 ± 0.2 | 30.8 ± 0.4 | 61.8 ± 0.4 | +11.5 ± 0.7 |
| terramind | pretrained | lora | 0.88 | 93.1 ± 0.3 | 92.6 ± 0.4 | 26.5 ± 0.4 | 55.4 ± 1.2 | +1.4 ± 1.0 |
| terramind | pretrained | moelora | 1.03 | 92.9 ± 0.1 | 92.4 ± 0.1 | 27.8 ± 0.3 | 57.7 ± 0.4 | +3.8 ± 0.7 |
| vit | scratch | full | 86.0 | 83.9 ± 0.6 | 82.8 ± 0.7 | 17.0 ± 0.3 | 36.7 ± 0.4 | −22.5 ± 0.9 |
| terramind | scratch | full | 85.3 | 86.7 ± 0.2 | 86.1 ± 0.2 | 17.8 ± 0.5 | 38.9 ± 0.4 | −19.3 ± 0.8 |

Val (1,426 images; the Phase 1 leaderboard set):

| backbone | init | adapt | trainable backbone params (M) | accuracy | macro-F1 | mAP | AP50 | Δm |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vit | pretrained | full | 86.0 | 94.7 ± 0.3 | 94.4 ± 0.3 | 32.0 ± 0.4 | 62.7 ± 0.4 | +11.8 ± 0.7 |
| vit | pretrained | lora | 0.88 | 93.6 ± 0.5 | 93.3 ± 0.4 | 26.1 ± 0.4 | 54.7 ± 0.6 | 0.0 ± 0.9 |
| vit | pretrained | moelora | 1.03 | 92.6 ± 0.8 | 92.2 ± 0.8 | 27.1 ± 0.2 | 56.4 ± 0.3 | +1.3 ± 0.9 |
| terramind | pretrained | full | 85.3 | 94.5 ± 0.2 | 94.2 ± 0.3 | 31.5 ± 0.1 | 62.4 ± 0.0 | +10.7 ± 0.3 |
| terramind | pretrained | lora | 0.88 | 92.6 ± 0.4 | 92.2 ± 0.4 | 27.0 ± 0.5 | 55.9 ± 0.8 | +1.0 ± 1.3 |
| terramind | pretrained | moelora | 1.03 | 92.2 ± 0.5 | 91.9 ± 0.4 | 28.8 ± 0.1 | 58.7 ± 0.1 | +4.4 ± 0.3 |
| vit | scratch | full | 86.0 | 83.8 ± 0.8 | 83.5 ± 0.9 | 18.2 ± 0.5 | 39.0 ± 0.7 | −20.3 ± 1.3 |
| terramind | scratch | full | 85.3 | 86.0 ± 1.0 | 86.0 ± 0.9 | 19.2 ± 0.4 | 41.4 ± 0.4 | −17.3 ± 1.2 |

## Submission format

One zip with two files; `make_submission.py:write_submission(images, preds)` writes it:
- `classification.csv`: `image_id,label` (label 0–8), one row per image;
- `detection.json`: a list of `{image_id, category_id (1–9), bbox [x, y, w, h] in original pixels, score}`, where
  score is the confidence.

`image_id` is the integer id from `images.json`, not the file name. `preds[file_name]` takes boxes as XYXY on the
512 input with classes 0–8; the function rescales them and adds 1. The samples are `dataset/sample_submission_val.zip`
(every label 0, no detections) and, from Phase 2, `sample_submission_test.zip`. The authoritative format is in
section 02 of the leaderboard docs (`/docs`). The format is unchanged from Codabench, and the scorer gives the same
numbers.
- The zip must be **≤ 25 MB** (HTTP 413 otherwise). The two files can sit in any folder of the zip; other files are
  ignored. Our val zips are about 19–22 MB, so a design that outputs more boxes per image can exceed the limit.
- As in COCO, only the 100 highest-scoring detections per image and class are scored.

### How to submit

- **From the browser**: sign in on the site with the team token, drop the zip on the Submit page, then post it from
  the Submit result or the Team page.
- **From the API**: every call sends `Authorization: Bearer $CS701_TOKEN`. Load the token with
  `set -a; . ./.env; set +a`, then pass the header as a file (`-H @<(auth)`). Never write the token into curl's
  arguments: other users on this shared server can read every process's arguments with `ps`.

  ```bash
  LB=https://web-production-c241e.up.railway.app
  auth() { printf 'Authorization: Bearer %s\n' "$CS701_TOKEN"; }   # a shell builtin: no process gets the token
  curl -sS --fail-with-body -X POST "$LB/api/submissions" -H @<(auth) \
      -H "Idempotency-Key: $(uuidgen)" -F "file=@<zip>"    # 201: scores + quota left; 422: format error
  curl -sS --fail-with-body "$LB/api/me" -H @<(auth)                            # quota, reset cards, posted entry
  curl -sS --fail-with-body "$LB/api/submissions" -H @<(auth)                   # ours, newest first
  curl -sS --fail-with-body -X POST "$LB/api/submissions/<id>/post" -H @<(auth)
  ```

  Reuse an `Idempotency-Key` only to retry the same upload. A retry with the same key returns the original
  submission (HTTP 200) and doesn't count twice. Other endpoints: `GET /api/status`, `GET /api/leaderboard?phase=N`
  (both public), `POST /api/phases/{n}/withdraw`, `POST /api/resets/{id}/use`.
- **Who does what**: Claude uploads through the API with the token from `.env`, and only zips that pass
  `check_submission.py`. Claude records every submission's id and scores next to the zip. It withdraws an entry or
  uses a reset card only when the user asks.

### Submission rule (the user's, 25 Oct 2026)

Upload each design's best model, **without first comparing it with the TA's baselines**. The local holdout doesn't
predict the leaderboard well enough to justify that step, and it would mean rerunning the baselines for every
holdout.
- **Phase 1 (val)**: once a design's runs are done, Claude uploads its best model.
  - The best model is the design's own configuration with the highest local Δm on its holdout: the 3-seed mean,
    decoded as it will be submitted. TA configurations are not candidates.
  - Local Δm is as the design's comparison script computes it: against the holdout's ViT + LoRA runs if it has
    them, otherwise against the leaderboard's val reference (93.31 / 26.13).
  - The zip comes from the checkpoint the design's README names for submission, e.g. that configuration retrained
    on all of train.
- **Phase 2 (test, 2 uploads in total)**: the 2 best models by their val leaderboard Δm. Claude confirms the pair
  with the user before uploading, because those 2 uploads can't be redone.
- **Posting**: the score comes back with the upload, so the TA's baselines are compared only then. Claude posts a
  scored submission if it beats both the team's current entry and the TA's best in-budget baseline (TerraMind +
  MoE-LoRA: Δm **+4.4 on val, +3.8 on test**; full fine-tuning is over budget, so it isn't the bar). Otherwise
  Claude reports the score and leaves the board as it is.

## Report must state

Total and trainable params; compute (GPU type/count, final training time, total project GPU hours incl. tuning,
peak GPU memory, inference time per image); input size and preprocessing; every pretrained model with source;
external libraries vs. own code; how train was split for validation and how the Phase 2 submission was chosen;
number of seeds; ablations on the local validation split. Cite the four source datasets (SARDet-100K, RSAR, M4-SAR,
HRSID).
