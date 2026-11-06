# DetailLoRA

An in-budget multi-task design for the CS701 SAR benchmark. It keeps the reference model (ViT-B/16 ImageNet-21k,
mean-pool classifier, ViTDet pyramid + Faster R-CNN) and adds:

1. **dense LoRA**: rank 32 on every linear layer, plus the patch and position embeddings and LayerNorms unfrozen
   at 2e-5;
2. **a detail branch**: a 0.34 M-parameter conv stem feeding real pixel detail into the stride-4/8 pyramid levels;
3. **small anchors**: from 5.7 px;
4. **class-guided detection scoring**: every box of an image has the image's class;
5. **SAR augmentation**: transpose and gamma.

**111.2 M total, 26.2 M trainable** (budget 130 M / 40 M; asserted at startup).

- Why each piece: [docs/DESIGN.md](docs/DESIGN.md).
- The data facts behind it, and how the holdout was drawn: [docs/DATA.md](docs/DATA.md).

Every command below runs **from the repo root** (`~/workspace/cs701-sar-benchmark`). Only the seed changes between
runs. Every GPU command goes through the team's runner `workspace/gpu_run.sh --gpus "1 2"`. It waits for a GPU that
is healthy, has the free memory, is idle and isn't locked by our other runs (shared locks in
`workspace/runs/gpu_locks/`), then sets `CUDA_VISIBLE_DEVICES`. **GPU 0** (memory fault) fails its health check
and is excluded anyway.

## Files

| file | what |
|---|---|
| `make_holdout.py` | builds `dataset/holdout/` (10 %, seed 0, class × image-size stratified, near-duplicate groups kept whole) |
| `analyze_data.py` | the train statistics of `docs/DATA.md` → `docs/data_stats.json` |
| `detail_model.py` | dense LoRA, detail branch, small anchors, class-guided RoI heads, parameter count and budget check |
| `detail_data.py` | training dataset: the reference pipeline + transpose + gamma |
| `common.py` | reuses the unchanged top-level `train.py` (training loop, evaluation, schedule) and `workspace/baseline/predict.py` (image preprocessing); rebuilds any run's model |
| `train.py` | trains DetailLoRA; checkpoints every epoch (`--resume`); writes `metrics.json` with params, `train_hours`, peak memory, inference ms/image, all three scorings |
| `evaluate.py` | ablation A2 on any run, reference runs included: holdout scores under scoring `none` / `hard` / `soft` → `fusion_val.json` |
| `predict.py` | Codabench zip for val or test via `make_submission.write_submission`, then checks its format |
| `compare.py` | markdown tables: metrics, Δm per seed vs the local ViT + LoRA (mean ± std), A2, per-class AP, total GPU hours |
| `queue.sh`, `jobs/*.txt` | runs a job file in order, each job through `workspace/gpu_run.sh` on GPU 1 or 2 |
| `tests/` | `uv run python -m pytest workspace/detail_lora/tests` (CPU, about 1 min) |
| `runs/` | all runs (gitignored) |

## 1. Build the holdout (once, about 15 s)

```bash
uv run python workspace/detail_lora/make_holdout.py
uv run python -m pytest workspace/detail_lora/tests -q   # includes the holdout checks: disjoint, stratified, groups whole
```

- **Writes** `dataset/holdout/`: `train/` (8,453 images), `val/` (the 939-image holdout), and `test → val`.
  Images are symlinks to `dataset/train/images`.
- **Record**: `dataset/holdout/split.json` holds the seed, threshold, source checksums, the duplicate groups and the
  holdout ids.
- **Rebuilding**: the same command rebuilds the same split.

## 2. Check that GPUs 1 and 2 are free (before every launch)

```bash
workspace/gpu_run.sh --gpus "1 2" --status          # health, free memory, utilisation, our locks
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
  | while IFS=, read -r pid mem; do echo "$mem $(ps -o user=,etime=,args= -p "$pid" | cut -c1-100)"; done
```

- **Free**: a GPU is free when it is healthy, has ≥ 25 GB free and is < 20 % utilised, with no lock of ours.
  `hanliu`'s small `app.py` is not a training job.
- **Busy**: you don't need to wait by hand. Every command below goes through `gpu_run.sh`, which waits for a free
  GPU (`--max-wait <s>` gives up after that long).
- **Memory**: each run needs about 10–20 GB (DetailLoRA peaked at 9.9 GB).

## 3. Local baselines on the holdout (ViT + LoRA = Δm reference, TerraMind + MoE-LoRA)

These run the unchanged top-level `train.py` on the holdout layout, with one `--out` per seed. Set `S` to the seed
(0, 1, 2) before each launch. `gpu_run.sh` picks the GPU and logs its choice at the top of the `.out` file.

ViT + LoRA (the Δm reference):

```bash
S=0
nohup workspace/gpu_run.sh --gpus "1 2" -- uv run python train.py --backbone vit --init pretrained --adapt lora \
    --data dataset/holdout --seed $S --out workspace/detail_lora/runs/ref_seed$S \
    > workspace/detail_lora/runs/ref_seed$S.out 2>&1 &
```

TerraMind + MoE-LoRA (the best in-budget baseline):

```bash
S=0
nohup workspace/gpu_run.sh --gpus "1 2" -- uv run python train.py --backbone terramind --init pretrained --adapt moelora \
    --data dataset/holdout --seed $S --out workspace/detail_lora/runs/ref_seed$S \
    > workspace/detail_lora/runs/tm_seed$S.out 2>&1 &
```

- **Output**: runs land in `runs/ref_seedS/vit_pretrained_lora/` and `runs/ref_seedS/terramind_pretrained_moelora/`.
- **Holdout scored twice**: train.py scores its `test` split too, which here is the holdout again.

## 4. DetailLoRA, 3 seeds (+ ablation A1)

**Option A, one run by hand** (set `S` to the seed before each launch):

```bash
S=0
nohup workspace/gpu_run.sh --gpus "1 2" -- uv run python workspace/detail_lora/train.py --seed $S \
    --out workspace/detail_lora/runs/seed$S > workspace/detail_lora/runs/seed$S.out 2>&1 &
```

Ablation A1 (no detail branch, reference anchors):

```bash
S=0
nohup workspace/gpu_run.sh --gpus "1 2" -- uv run python workspace/detail_lora/train.py --no-detail --reference-anchors \
    --seed $S --out workspace/detail_lora/runs/seed$S > workspace/detail_lora/runs/a1_seed$S.out 2>&1 &
```

- **Output**: the primary goes to `runs/seedS/detaillora/`, and A1 to `runs/seedS/detaillora_nodetail_refanchors/`.
  The names differ, so nothing is overwritten.

**Option B, the whole plan, queued** (steps 3 + 4: 12 runs, primary and its reference first, each through
`gpu_run.sh` on GPU 1 or 2):

```bash
mkdir -p workspace/detail_lora/runs
nohup setsid workspace/detail_lora/queue.sh workspace/detail_lora/jobs/holdout.txt \
    >> workspace/detail_lora/runs/queue.log 2>&1 &
```

- **Order** (`jobs/holdout.txt`): DetailLoRA s0, ViT + LoRA s0, then s1 and s2 of both, then A1 and TerraMind +
  MoE-LoRA for s0–s2.
- **Trimming**: delete lines of `jobs/holdout.txt` *before* starting the queue to skip runs (e.g. A1 seeds 1–2 if
  short of time).
- **Shared GPUs**: each job takes a shared lock through `gpu_run.sh`, so the queue never lands on a GPU that
  `run_repro.sh` or another design's runs hold.
- **History**: on 24 Oct the plan was dispatched by an earlier version of `queue.sh` (its own markers) up to
  DetailLoRA s1. The rest went through `jobs/holdout_rest.txt` with this version.

## 4b. Follow-up: the detail branch on TerraMind + MoE-LoRA

The TerraMind + MoE-LoRA baseline, unchanged (MoE-LoRA with task routers, flips-only augmentation, torchvision
scoring), plus the detail branch and small anchors. It is a single-factor comparison against the baseline runs of
step 3. **106,836,680 total, 21,488,840 trainable** (the baseline: 106,493,209 / 21,145,369).

```bash
S=0
nohup workspace/gpu_run.sh --gpus "1 2" -- uv run python workspace/detail_lora/train.py --backbone terramind \
    --adapt moelora --no-augment --fusion none --seed $S --out workspace/detail_lora/runs/seed$S \
    > workspace/detail_lora/runs/tmd_seed$S.out 2>&1 &
```

- **Output**: `runs/seedS/detail_terramind_moelora_noaug/`. `compare.py` labels it "TerraMind + MoE-LoRA + detail +
  anchors".
- **Queued form**: `jobs/terramind_detail_smoke.txt` (smoke test), then `jobs/terramind_detail.txt` (3 seeds),
  each with `queue.sh`.

## 5. Compare

After the runs finish, add ablation A2 (scorings `none` / `hard` / `soft`) for the reference runs. It needs one
free GPU for about 3 min per run; DetailLoRA runs already contain it. Then print the tables:

```bash
workspace/gpu_run.sh --gpus "1 2" --min-free 8000 -- \
    uv run python workspace/detail_lora/evaluate.py workspace/detail_lora/runs/ref_seed*/*/
uv run python workspace/detail_lora/compare.py workspace/detail_lora/runs
```

`compare.py` prints:
- **the main table**: accuracy, macro-F1, mAP, AP50, Δm per seed against the mean of the local ViT + LoRA seeds
  (mean ± sample std), parameters, train hours, peak memory and ms/image;
- **the A2 table**;
- **per-class AP**;
- **the total GPU hours**.

Groups are kept apart by training data, epochs and `--limit`, so smoke and full-train runs never mix with holdout
runs. A difference below ~2 Δm is within seed noise: report mean ± std.

**Choosing the scoring for submissions:** use the scoring with the best mean holdout Δm across the 3 DetailLoRA
seeds in the A2 table (the default is `soft`). If it isn't `soft`, pass `--fusion <best>` to `predict.py`.

## 6. Retrain on all of train for the submissions

**Submission checkpoint**: the best holdout configuration, TerraMind + MoE-LoRA + detail + anchors (step 4b), seed 0,
queued with `jobs/terramind_detail_final.txt` → `runs/full_seed0/detail_terramind_moelora_noaug/` (25 Oct, 1.85 h on
GPU 1). Its val zip, decoded with `--fusion soft` (the best A2 mean for that configuration):
`submissions/submission_val_tmd_full_seed0_soft.zip`. The DetailLoRA (ViT) commands below are kept for reference.

```bash
nohup workspace/gpu_run.sh --gpus "1 2" -- uv run python workspace/detail_lora/train.py --data dataset/repro --seed 0 \
    --out workspace/detail_lora/runs/full_seed0 > workspace/detail_lora/runs/full_seed0.out 2>&1 &
# or queued: nohup setsid workspace/detail_lora/queue.sh workspace/detail_lora/jobs/final.txt >> workspace/detail_lora/runs/queue.log 2>&1 &
```

**Why retrain.** It adds 11 % more data, 939 images that include whole duplicate groups (new scenes). Nothing in
the recipe was tuned on the holdout: fixed 24 epochs, last-epoch model, no checkpoint selection. The holdout only
chose the design and the scoring, so retraining loses nothing. The TA's reference runs, which the leaderboard's Δm
is relative to, were also trained on all of train.

- **Data**: `dataset/repro/` is `workspace/baseline/make_repro_data.py`'s layout, with train = all 9,392 images. If
  it is missing, build it with `uv run python workspace/baseline/make_repro_data.py`.
- **Its val scores**: they are on a 5 % sample of train, a fit check, not a holdout.
- **Fallback**: if the full-train run can't finish in time, submit the best holdout-trained seed instead.

## 7. Val submission zip (Codabench Phase 1, until 1 Nov 23:59 SGT, 5/day)

```bash
workspace/gpu_run.sh --gpus "1 2" --min-free 8000 -- \
    uv run python workspace/detail_lora/predict.py workspace/detail_lora/runs/full_seed0/detaillora --split val
unzip -l workspace/detail_lora/runs/full_seed0/detaillora/submission_val.zip
```

`predict.py` checks the zip before printing `format OK`:
- **files**: exactly `classification.csv` + `detection.json`, with the same names and CSV header as
  `dataset/sample_submission_val.zip`;
- **classification**: one row per val image, labels 0–8;
- **detection**: every entry has `image_id`, `category_id` 1–9, `bbox [x, y, w, h]` and `score` in [0, 1].

Then **you** upload `submission_val.zip` on Codabench as `cs701-team05` (Participate → Submit). Any trained run
works the same way, and so do reference runs. Submitting `runs/seed0/detaillora` as soon as it finishes gives an
early leaderboard reading, before the full-train run:
`predict.py workspace/detail_lora/runs/ref_seed0/vit_pretrained_lora --split val`. If the format is ever
rejected, upload `dataset/sample_submission_val.zip` once to check that the pipeline accepts the sample.

## Phase 2 (test, 2–8 Nov, **only 2 uploads**)

1. **Get the test images.** When Phase 2 opens (2 Nov 00:01 SGT), download and unpack `test.tar.gz` and
   `sample_submission_test.zip` into `dataset/` (`hf download doem1997/cs701-sar-course-data --repo-type dataset
   --local-dir dataset`, then `tar -xzf dataset/test.tar.gz -C dataset`). Check that
   `dataset/test/images.json` lists 2,065 images.
2. **Fix the model before Phase 2.** Use the full-train DetailLoRA run with the scoring chosen in step 5,
   confirmed on the Phase 1 val leaderboard. **Do not choose between models by their test score**: with 2 uploads,
   that is selection on the test set.
3. **Build the test zip.**
   `workspace/gpu_run.sh --gpus "1 2" --min-free 8000 -- uv run python workspace/detail_lora/predict.py workspace/detail_lora/runs/full_seed0/detaillora --split test`.
   It must print `format OK` with 2,065 images.
4. **Sanity-check the zip against the val zip from the same model.** Run
   `unzip -p <zip> classification.csv | cut -d, -f2 | sort | uniq -c` on both: the class counts should be roughly
   proportional. The detections per image should be similar.
5. **Upload 1** = that zip. Then click **"Add to Leaderboard"** next to it.
6. **Upload 2 is a reserve.** Use it only if upload 1 fails technically (rejected, wrong file). Without an "Add to
   Leaderboard" click, the *last* Phase 2 upload counts, so never use upload 2 for an experiment.

## Expected time per run (L40S)

| run | per run | note |
|---|---:|---|
| ViT + LoRA, holdout | ~1.05 h | measured 57 images/s on an L40S (23 Oct) |
| TerraMind + MoE-LoRA, holdout | ~1.5 h | estimate (backbone runs twice) |
| DetailLoRA, holdout | 1.30 h | measured 24 Oct, seed 0: 44.6 images/s, peak 9.9 GB, inference 9.2 ms/image (batch 16) and 18.9 ms (batch 1) |
| A1 (no detail branch), holdout | 1.19 h | measured 24 Oct |
| TerraMind + MoE-LoRA (+ detail + anchors), holdout | 1.51 h (1.67 h) | measured 24 Oct, on a GPU not shared during the run |
| DetailLoRA, all of train | ~1.45 h | from the holdout run, × 9,392 / 8,453 images |
| smoke test (`--limit 64 --epochs 1`) | 2 min | measured 23 Oct: peak 9.9 GB; inference 18.8 ms/image (batch 16, forward), 41.4 ms (batch 1), 22.5 ms with data loading |

The whole plan (12 holdout runs + 1 full-train run) is about **16 GPU-h ≈ 8 h on GPUs 1 + 2 when both are free**.

- **Shared GPUs run slower.** On 23 Oct a ViT + LoRA run on GPU 1, next to another user's app, trained at 30–36
  images/s, against 57 alone: up to ~1.6× the times above.
- **Start early**: the queue may wait hours for a free GPU.

## Monitor, resume, stop

```bash
workspace/gpu_run.sh --gpus "1 2" --status                     # GPU health, memory, utilisation, our locks
cat workspace/detail_lora/runs/queue.log                        # what the queue started / released, on which GPU
tail -f workspace/detail_lora/runs/seed0.out                    # live progress of one run (loss, images/s)
grep -h "val:" workspace/detail_lora/runs/*/*/log.txt           # holdout scores every 4 epochs, all runs
ls workspace/detail_lora/runs/*/*/metrics.json                  # finished runs
nvidia-smi                                                       # memory and utilisation
```

- **Resume a DetailLoRA run** after a crash or kill: re-run the *same* command with `--resume`. It continues from
  `checkpoint.pt` (written after every epoch) and keeps counting `train_hours`. The checkpoint is deleted when the
  run finishes (`--keep-checkpoint` keeps it). The data order after a resume is not bit-identical; it is
  statistically equivalent.
- **Reference runs** (top-level `train.py`) have no checkpoints: re-run them from the start.
- **Stop the queue**: `pkill -f detail_lora/queue.sh`. It stops dispatching; started jobs go on, and jobs still
  waiting in `gpu_run.sh` start when a GPU frees, so also run `pkill -f "gpu_run.sh --gpus 1 2 -- bash -c"` to drop
  those.
- **Stop one run**: find its `gpu_run.sh` pid with `ps -u $USER -o pid,etime,args | grep "[g]pu_run.sh"`, then
  `kill <pid>`. It stops the run and releases the GPU lock. A lock left by a killed process is cleared
  automatically at the next `gpu_run.sh` check.

## Report facts (fill in from `metrics.json` and `compare.py`)

| fact | value / where |
|---|---|
| total parameters | 111,216,584 (`parameters`); reference 107,039,257 |
| trainable parameters | 26,198,984 (`trainable_parameters`): LoRA 4,718,592 + unfrozen pretrained 1,023,744 + detail branch 339,616 + heads 20,117,032 (`parameter_breakdown`) |
| GPU type / count | NVIDIA L40S 46 GB; 3 in the server, runs on GPUs 1–2, one GPU per run (`gpu`) |
| final training time | `train_hours` of `runs/full_seed0/detaillora/metrics.json`: ___ h |
| total project GPU hours (incl. tuning, smoke, baselines) | last line of `compare.py workspace/detail_lora/runs` + `workspace/baseline/runs`: ___ h |
| peak GPU memory | `peak_gpu_memory_gb` (allocated by PyTorch; the process needs ~2.5 GB more): ___ GB |
| inference time per image | `inference_ms_per_image`: batch 16 ___ ms, batch 1 ___ ms (forward only, bf16); with data loading: `predict.py`'s printout ___ ms |
| input size, preprocessing | gray, resized to 512 × 512 (non-square stretched, as the reference), normalised with train mean/std 0.2526 / 0.2133; training only: H/V flips, random transpose (p 0.5), random gamma 0.8–1.25 (p 0.5) |
| pretrained models | ViT-B/16 `vit_base_patch16_224.augreg_in21k` (timm / Hugging Face, ImageNet-21k, Steiner et al. 2022). Baselines only: TerraMind-1.0-base (IBM/ESA, via terratorch). No SAR-pretrained model |
| external libraries vs own code | PyTorch, torchvision (Faster R-CNN RPN / RoI heads, anchors, NMS), timm (ViT), terratorch (TerraMind), pycocotools (mAP); TA reference code `sarbench/`, `train.py`, `make_submission.py`; own: `workspace/detail_lora/` |
| validation split | 10 % holdout of train, class × image-size stratified, near-duplicate groups kept whole, seed 0 (`docs/DATA.md` §5–6, `dataset/holdout/split.json`) |
| seeds | 0, 1, 2 for every holdout configuration; full-train run seed 0 |
| Phase 2 choice | the design chosen on the holdout (3 seeds), scoring chosen by A2 on the holdout, confirmed on Phase 1 val; retrained on all of train; one upload |
| ablations | A1 (detail branch + anchors), A2 (detection scoring), from `compare.py` |
| datasets to cite | SARDet-100K, RSAR, M4-SAR, HRSID (`dataset/SOURCES_AND_LICENSES.md`) |

## Rules check

- **Parameters**: ≤ 130 M total, ≤ 40 M trainable, asserted before training (`detail_model.check_budget`).
- **One model, each image fed once**: one forward pass per image at 512 × 512, no TTA, no ensemble. The
  class-guided scoring reuses the same forward pass's classification output; worth one line in the report.
- **Data**: train split only. No val/test labels or images used in training or analysis; no SAR-pretrained weights.
- **Inherited geometry**: the reference resizes 256² images up to 512². DetailLoRA keeps that geometry unchanged.
