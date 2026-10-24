# WideLoRA-SP with joint decoding (`workspace/sarmtl/`)

Team `cs701-team05`'s design for the CS701 SAR multi-task benchmark. A ViT-B/16 (ImageNet-21k) gets three changes:
- **WideLoRA:** rank-64 LoRA on every linear layer, plus the small pretrained tensors unfrozen;
- **SP:** a 0.36 M spatial-prior conv branch into the ViTDet neck's stride-4/8 levels;
- **JD:** joint decoding of the two heads at inference.

**115,956,473 total, 31,021,817 trainable** (budget 130 M / 40 M; asserted at startup).

- The design, decision rules, run plan and submission rule: [docs/DESIGN.md](docs/DESIGN.md).
- The data facts and how the holdout was drawn: [docs/DATA.md](docs/DATA.md).
- The candidates: [docs/cand_adapt.md](docs/cand_adapt.md) and [docs/cand_detect.md](docs/cand_detect.md).
- The critic's review: [docs/REVIEW.md](docs/REVIEW.md).
- Results: [RESULTS.md](RESULTS.md).

Every command below runs **from the repo root**. **Every GPU command goes through `workspace/gpu_run.sh`**, which
waits for a GPU that is healthy, free, idle and unlocked. GPU 0 is skipped while it fails the health check. Check the
state with `workspace/gpu_run.sh --status`. Python always runs through uv (`uv run python`), or `.venv/bin/python`
inside the queue.

## Files

| file | what |
|---|---|
| `sarmtl_model.py` | WideLoRA, SP, JD and the model builder: unchanged `sarbench/` classes, subclassed or wrapped |
| `train.py` | trains a configuration: root `train.py`'s CLI and loop, budget asserted, paired seeding, JD on/off scores, `metrics.json` with `train_hours`, peak memory and inference ms/image |
| `evaluate.py` | any checkpoint (ours or root `train.py`'s) on the holdout: reference decoding, JD and JD-posterior, whole holdout and halves, written to `jd_eval.json` |
| `crossfit.py` | the 2-fold cross-fit of the decodings on the holdout halves (DESIGN.md §4, rule 3); `compare.py` uses it |
| `predict.py` | the leaderboard zip for val or test (`make_submission.write_submission`), checked at the end; `--jd on\|off\|posterior` |
| `check_submission.py` | the zip against the sample's format, the rules and the leaderboard's 25 MB limit |
| `compare.py` | the results tables: Δm per seed vs the local ViT + LoRA, the JD keep rule, C*, per-class AP, GPU hours |
| `run.sh`, `jobs/plan.txt` | the experiment queue: priority order, one job per GPU, each through `gpu_run.sh` |
| `jobs/post_queue.sh` | the GPU jobs after the queue: A1's `evaluate.py` runs and the val zips |
| `data/` | data-eng's holdout pipeline (DATA.md §6) |
| `tests/` | `uv run python -m pytest workspace/sarmtl/tests -q` (CPU, ~1 min) |
| `runs/` | every run (gitignored); `runs/queue/` holds the queue's logs and state |
| `submissions/` | the zips to upload; `SUBMISSIONS.md` records every upload, its id and its scores |

## 1. The holdout (CPU, once)

`dataset/holdout_sarmtl/` is already built and **frozen** (`data/frozen_holdout.json`): 940 holdout / 8,057 local
train / 395 buffer. To rebuild it byte-identically:

```bash
uv run python workspace/sarmtl/data/extract_features.py
uv run python workspace/sarmtl/data/near_dups.py              # ~1.5 h of CPU the first time, then cached
uv run python workspace/sarmtl/data/build_holdout.py
uv run python workspace/sarmtl/data/sweep_buffer.py
uv run python workspace/sarmtl/data/make_holdout.py --fraction 0.1 --seed 0   # asserts; writes split.json
uv run python workspace/sarmtl/data/sweep_buffer.py --check  # must find 0 crossings
uv run python workspace/sarmtl/data/verify_holdout.py
uv run python -m pytest workspace/sarmtl/tests -q             # includes the holdout checks
```

Besides the near-duplicate groups (`data/near_dup_groups.json`), `make_holdout.py` reads 4 tracked inputs in `data/`:
- `audit_links.csv`;
- `frozen_holdout.json`;
- `sweep_buffer.json`;
- `critic_h3_buffer.json`.

## 2. Smoke tests (each through `gpu_run.sh`, one at a time)

```bash
mkdir -p workspace/sarmtl/runs/smoke
# 1. the design on the holdout
HF_HUB_OFFLINE=1 workspace/gpu_run.sh --max-wait 10800 -- uv run python workspace/sarmtl/train.py \
    --data dataset/holdout_sarmtl --limit 64 --epochs 1 --seed 0 --out workspace/sarmtl/runs/smoke/design
# 2. root train.py, ViT + LoRA, on the holdout
HF_HUB_OFFLINE=1 workspace/gpu_run.sh --max-wait 10800 -- uv run python train.py --backbone vit --init pretrained \
    --adapt lora --data dataset/holdout_sarmtl --limit 64 --epochs 1 --seed 0 --out workspace/sarmtl/runs/smoke/ref
# 3. a val zip and its check
HF_HUB_OFFLINE=1 workspace/gpu_run.sh --min-free 8000 --max-wait 10800 -- uv run python workspace/sarmtl/predict.py \
    workspace/sarmtl/runs/smoke/design/vit_widelora_sp --split val --jd on
uv run python workspace/sarmtl/check_submission.py \
    workspace/sarmtl/runs/smoke/design/vit_widelora_sp/submission_val_jdon.zip --split val
# 4. the tables
uv run python workspace/sarmtl/compare.py
```

`gpu_run.sh` exits with 75 if no GPU qualifies within `--max-wait`.

## 3. The experiments (the queue)

`jobs/plan.txt` lists the approved runs in priority order (DESIGN.md §5):
- the reference and the primary, interleaved, seeds 0–2;
- the primary on all of train;
- TerraMind + MoE-LoRA, seeds 0–2;
- A1 (`--no-sp`), seeds 0–2;
- the `evaluate.py` jobs.

Launch it detached, so it survives the session:

```bash
mkdir -p workspace/sarmtl/runs/queue
nohup setsid workspace/sarmtl/run.sh workspace/sarmtl/jobs/plan.txt >> workspace/sarmtl/runs/queue/run.log 2>&1 &
workspace/sarmtl/run.sh --status workspace/sarmtl/jobs/plan.txt   # pending / running on GPU N / exit code, minutes
```

- **How it schedules:** one worker per candidate GPU. A worker claims the next job only once `gpu_run.sh` holds a
  GPU for it, so the highest-priority job left always gets the next free GPU.
- **Logs:**
  - `runs/queue/run.log` has the `gpu_run:` and `run.sh:` lines (start, GPU, end, exit code);
  - each job's output is in `runs/queue/<name>.out`;
  - each run directory has its own `log.txt` and `metrics.json`.
- **One run by hand,** e.g. the primary with seed 1:

  ```bash
  HF_HUB_OFFLINE=1 workspace/gpu_run.sh -- uv run python workspace/sarmtl/train.py --data dataset/holdout_sarmtl \
      --seed 1 --out workspace/sarmtl/runs/P_s1
  ```

  A1 adds `--no-sp`. The full-train run uses `--data dataset/repro`, whose train split is all 9,392 images; its
  val/test scores are fit on a sample of train.

- **After the queue,** `jobs/post_queue.sh` runs A1's `evaluate.py` jobs and builds the val zips (§5), each through
  `gpu_run.sh`:

  ```bash
  nohup setsid workspace/sarmtl/jobs/post_queue.sh >> workspace/sarmtl/runs/queue/post_runs.log 2>&1 &
  ```

### Resume after a crash or a lost session

1. Run `workspace/gpu_run.sh --status` and read the failed job's `runs/queue/<name>.out`. A CUDA, Xid or ECC error
   means a GPU problem: report the GPU, then rerun the seed through the queue.
2. Launch the same `nohup setsid workspace/sarmtl/run.sh …` command again. Jobs with a state directory in
   `runs/queue/` (finished, failed or killed) are skipped; pending ones run in order.
3. A job name runs at most once. To rerun one, append a line with a new name and the same command (e.g.
   `P_s1_r2 <command>`) to `jobs/plan.txt`, which `run.sh` re-reads before every claim, or launch it by hand as above.
   The rerun overwrites the partial run directory.
4. A job whose worker was killed shows `running` forever in `--status`, because it never wrote an exit code. Check
   that its process is gone (`ps`), then requeue it as in step 3.
5. A worker stops after 2 failed jobs in a row, so a broken GPU or a bug can't burn the queue. Fix the cause, then
   relaunch.

## 4. Decisions and the report tables

```bash
uv run python workspace/sarmtl/compare.py --out workspace/sarmtl/runs/tables.md
```

The output gives:
- the tables with the reference decoding (A2), Δm per seed against the local ViT + LoRA, then mean ± sample std;
- the JD keep rule per configuration (rule 3), the "as submitted" table and **C\*** (rule 4);
- per-class AP, the GPU hours (`train_hours`, plus `run.sh`'s wall-clock), and the `metrics.json` behind every row.

JD scores of root `train.py` runs come from `evaluate.py`:

```bash
HF_HUB_OFFLINE=1 workspace/gpu_run.sh --min-free 8000 -- uv run python workspace/sarmtl/evaluate.py <run_dir>
```

The queue's `eval_*` jobs run it.

## 5. The submission

C* (RESULTS.md §1.3) is TerraMind + MoE-LoRA, and the 2-fold cross-fit adopted the JD-posterior decoding
(`crossfit.py`). The submission model is root `train.py` on `dataset/repro`, whose train split is all 9,392 train
images (queue job `tmfull_s0`). From scratch:

```bash
uv run python workspace/baseline/make_repro_data.py   # dataset/repro (CPU, once; --fraction 0.05 --seed 0)
HF_HUB_OFFLINE=1 workspace/gpu_run.sh -- uv run python train.py --backbone terramind --init pretrained --adapt moelora \
    --data dataset/repro --seed 0 --out workspace/sarmtl/runs/tmfull_s0
mkdir -p workspace/sarmtl/submissions
HF_HUB_OFFLINE=1 workspace/gpu_run.sh --min-free 8000 -- uv run python workspace/sarmtl/predict.py \
    workspace/sarmtl/runs/tmfull_s0/terramind_pretrained_moelora --split val --jd posterior \
    --out workspace/sarmtl/submissions/submission_val_rebuild.zip
uv run python workspace/sarmtl/check_submission.py workspace/sarmtl/submissions/submission_val_rebuild.zip --split val
```

The two optional val zips (RESULTS.md §4) use the same command:
- `runs/Pfull_s0/vit_widelora_sp` with `--jd posterior`, written to `alt_design_widelora_sp_val.zip`;
- `workspace/baseline/runs/repro/seed0/vit_pretrained_lora` with `--jd off`, written to
  `context_vitlora_repro_s0_val.zip`.

- **The uploaded val zip** was built with this command and `--out workspace/sarmtl/submissions/submission_val_tmfull_s0_jdposterior.zip`.
  `submission_val.zip` is a byte-identical copy of it. A rebuild writes `submission_val_rebuild.zip`, so it never
  overwrites the copy of the uploaded zip.
- **Compression:** since 24 Oct, 23:20, `predict.py` compresses the zip after `write_submission` (ZIP_DEFLATED).
  The uploaded val zip predates that and is stored uncompressed (16.4 MB). A stored test zip would come close to the
  leaderboard's 25 MB limit.
- **Phase 2** uses the same command with `--split test` once `dataset/test/` exists, written to
  `submissions/submission_test.zip`.
- **Uploads** go to the course leaderboard, https://web-production-c241e.up.railway.app/, which replaced Codabench on
  24 Oct 2026. A zip is uploaded only when the user asks, and only after `check_submission.py` passes.
  `submissions/SUBMISSIONS.md` records each upload.
