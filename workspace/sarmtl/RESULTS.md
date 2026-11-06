# RESULTS: WideLoRA-SP and the baselines on the local holdout

Status, 24 Oct 2026:
- **Local results are complete**: every planned run ran to the end, with exit code 0.
- **Leaderboard val (Phase 1):** three of our zips are scored (§4):
  - the submission, TerraMind: Δm +3.92;
  - the design, WideLoRA-SP: Δm **+7.09**;
  - ViT + LoRA: Δm +0.30.

  The team's posted entry is a later upload from outside this workspace (+10.09).
- **Test (Phase 2): pending.**

The course leaderboard (https://web-production-c241e.up.railway.app/) replaced Codabench on 24 Oct 2026. Codabench no
longer counts (CLAUDE.md, "Leaderboard phases").

Where the numbers come from:
- **Local scores:** a `metrics.json` (`train.py`) or a `jd_eval.json` (`evaluate.py`) under `workspace/sarmtl/runs/`.
  §6 lists the paths.
- **Leaderboard scores:** the server's replies (§4).
- **Times:** the logs named in §5.

The tables are `compare.py`'s output
(`uv run python workspace/sarmtl/compare.py --out workspace/sarmtl/runs/tables.md`). Scores are in %, on the
last-epoch model. "±" is the sample std over seeds 0, 1 and 2. Δm is computed per seed against the **local**
reference: ViT + LoRA on the holdout, mean of its 3 seeds, macro-F1 76.92 and mAP 12.52, with the reference decoding.

## 0. Summary

- **The design clearly beats the local reference.**
  - With the reference decoding, WideLoRA-SP reaches Δm **+7.3 ± 1.2** against 0.0 ± 1.9 for ViT + LoRA.
  - The difference is 7.35, at a standard error of 1.32 (t = 5.5).
  - It also has the best macro-F1 of every configuration, 78.3 ± 1.4.
- **It does not beat TerraMind + MoE-LoRA on this holdout.**
  - TerraMind reaches Δm +9.4 ± 2.3, or +9.9 ± 2.4 as submitted. It has the best mAP (15.0) but the lowest F1 (75.9).
  - The difference to the design is −2.0 (reference decoding; standard error 1.51, t = −1.3) or −1.5 (as
    submitted). That is below what 3 seeds resolve here (≈ 2.3σ ≈ 3–5 Δm), so it is **inconclusive**.
  - By the pre-registered rule (DESIGN.md §4, rule 4: the highest mean decides), **C\* = TerraMind + MoE-LoRA**.
- **The submission** (val now, test in Phase 2) is TerraMind + MoE-LoRA trained on all 9,392 train images
  (`runs/tmfull_s0`, seed 0, last epoch), decoded with JD-posterior: `submissions/submission_val.zip`, checked in §4.
  - **Leaderboard val:** Δm **+3.92**, with macro-F1 90.92, mAP 28.85 and AP50 58.78.
  - The TA's TerraMind + MoE-LoRA scores +4.4 ± 0.3 there. Our run matches its mAP (28.8) and AP50 (58.7), but its
    macro-F1 is 1.0 lower (91.9 ± 0.4).
  - On val, the submission is slightly below the best in-budget reference (−0.5 Δm, all from macro-F1), not above
    it.
  - **The design does beat it on val.** WideLoRA-SP (all of train, seed 0, JD-posterior) scores Δm **+7.09**, with
    macro-F1 93.60 and mAP 29.76. That is 2.7 above the TA's TerraMind + MoE-LoRA mean.
- **Ablations:**
  - SP adds **+3.1 ± 1.1 Δm** over WideLoRA alone, paired, with a gain on every seed.
  - WideLoRA alone is +4.3 ± 0.7 over the reference (+5.0 ± 0.6 as submitted).
  - The pre-registered JD (product) gains nothing and fails its keep rule everywhere.
  - Its posterior variant gains +0.1 to +0.3 mAP on every seed of every configuration, and the 2-fold cross-fit
    adopts it.
- **Caveat: the holdout is leak-free and much harder than val/test.**
  - Every configuration's means fall in mAP 12.5–15.2 and macro-F1 75.9–78.3 here, against the TA's val 26.1 / 93.3.
  - DATA.md §5: 73 % of train images have a near-duplicate, so a random val/test would share scenes with train; our
    holdout shares none.
  - The leaderboard ranks them differently. On val the design is ahead of TerraMind (+7.09 vs +3.92, one seed each).
    That reverses the holdout's inconclusive order (§4).

## 1. Local holdout (`dataset/holdout_sarmtl`: 940 images, 336 independent scenes; DATA.md §6)

### 1.1 Reference decoding (JD off): every configuration, 3 seeds

| config | seeds | accuracy | macro-F1 | mAP | AP50 | Δm | total params | trainable params | GPU h per run | GPU h | peak GiB | eval ms/image |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ViT + LoRA (local reference, root `train.py`) | 0,1,2 | 76.4 ± 1.5 | 76.9 ± 1.7 | 12.5 ± 0.3 | 30.1 ± 0.5 | −0.0 ± 1.9 | 107.04 M | 21.00 M | 0.98 | 2.94 | 7.4 | 12.7 |
| TerraMind + MoE-LoRA (root `train.py`) | 0,1,2 | 75.6 ± 2.7 | 75.9 ± 3.3 | 15.0 ± 0.1 | 34.5 ± 0.5 | +9.4 ± 2.3 | 106.49 M | 21.15 M | 1.48 | 4.43 | 14.2 | 17.0 |
| A1: WideLoRA alone (`--no-sp`) | 0,1,2 | 76.9 ± 0.6 | 77.0 ± 0.5 | 13.6 ± 0.3 | 32.3 ± 0.9 | +4.3 ± 0.7 | 115.59 M | 30.66 M | 1.12 | 3.37 | 9.1 | 13.7 |
| **WideLoRA-SP (the design)** | 0,1,2 | 77.5 ± 1.0 | 78.3 ± 1.4 | 14.1 ± 0.2 | 33.6 ± 0.3 | +7.3 ± 1.2 | 115.96 M | 31.02 M | 1.20 | 3.59 | 10.4 | 14.2 |

- **eval ms/image:** holdout evaluation time per image at batch 16 under bf16. It includes data loading and COCO
  scoring, and the number of scoring passes differs:
  - `train.py`'s runs (A1, WideLoRA-SP; `metrics.json`) score once;
  - the root `train.py` runs (`evaluate.py`'s reference-decoding pass, `jd_eval.json`) score three times: the whole
    holdout and both halves.

  So the column is not like for like. §5 gives `predict.py`'s inference times.
- **Peak GiB:** `torch.cuda.max_memory_allocated`. The process needs ≈ 2.5 GiB more.

Per seed (`metrics.json` → `val`):

| config | seed 0: F1 / mAP / Δm | seed 1 | seed 2 |
|---|---|---|---|
| ViT + LoRA | 77.55 / 12.21 / −0.85 | 78.18 / 12.87 / +2.21 | 75.03 / 12.49 / −1.35 |
| TerraMind + MoE-LoRA | 73.18 / 14.94 / +7.22 | 74.94 / 15.12 / +9.07 | 79.64 / 15.03 / +11.79 |
| A1: WideLoRA alone | 77.38 / 13.39 / +3.75 | 76.45 / 13.88 / +5.11 | 77.13 / 13.50 / +4.03 |
| WideLoRA-SP | 77.95 / 14.32 / +7.86 | 79.80 / 14.12 / +8.25 | 77.08 / 13.98 / +5.93 |

### 1.2 Decoding: the JD keep rule and the 2-fold cross-fit (DESIGN.md §4, rule 3)

JD (product, α = 1, score threshold 0.001), paired on the same checkpoint (mAP points, JD − reference decoding):

| config | gains per seed | keep JD (> 0 on all 3) |
|---|---|---|
| ViT + LoRA | +0.00, +0.04, −0.05 | no |
| TerraMind + MoE-LoRA | −0.48, −0.37, −0.21 | no |
| A1: WideLoRA alone | −0.02, −0.08, +0.05 | no |
| WideLoRA-SP | −0.04, +0.04, −0.02 | no |

The cross-fit (`crossfit.py`):
- Choose the best of reference / JD / JD-posterior on one half of the holdout, by the 3-seed mean mAP. Then score it
  on the other half.
- A decoding is adopted only if both folds choose it and it gains on every seed in both. That is the pre-registered
  criterion (cand_detect.md §2, step 4).
- Until the final audit (REVIEW.md F12), `crossfit.py` checked only the mean gain. Every per-seed gain is positive, so
  no decision changes.

Mean out-of-fold mAP gain (points), with the per-seed gains in brackets:

| config | half A → half B | half B → half A | adopted |
|---|---|---|---|
| ViT + LoRA | JD-posterior, +0.14 (+0.13, +0.16, +0.14) | JD-posterior, +0.19 (+0.21, +0.19, +0.17) | JD-posterior |
| TerraMind + MoE-LoRA | JD-posterior, +0.17 (+0.16, +0.14, +0.20) | JD-posterior, +0.07 (+0.09, +0.05, +0.07) | JD-posterior |
| WideLoRA-SP | JD-posterior, +0.20 (+0.18, +0.26, +0.15) | JD-posterior, +0.29 (+0.31, +0.28, +0.27) | JD-posterior |
| A1: WideLoRA alone | JD-posterior, +0.15 (+0.18, +0.10, +0.17) | JD-posterior, +0.22 (+0.24, +0.22, +0.19) | JD-posterior |

JD-posterior keeps each RoI's objectness and moves its class mass towards the image's class. On the whole holdout it
gains on every seed: ViT + LoRA +0.17, +0.18, +0.11; TerraMind +0.13, +0.10, +0.15; A1 +0.20, +0.14, +0.17;
WideLoRA-SP +0.23, +0.27, +0.24 mAP points.

### 1.3 As submitted (each configuration with its adopted decoding) and C\*

| config | decoding | accuracy | macro-F1 | mAP | AP50 | Δm |
|---|---|---:|---:|---:|---:|---:|
| ViT + LoRA | JD-posterior | 76.4 ± 1.5 | 76.9 ± 1.7 | 12.7 ± 0.3 | 30.6 ± 0.5 | +0.6 ± 2.0 |
| **TerraMind + MoE-LoRA** | JD-posterior | 75.6 ± 2.7 | 75.9 ± 3.3 | 15.2 ± 0.1 | 34.8 ± 0.5 | **+9.9 ± 2.4** |
| A1: WideLoRA alone | JD-posterior | 76.9 ± 0.6 | 77.0 ± 0.5 | 13.8 ± 0.2 | 32.9 ± 0.9 | +5.0 ± 0.6 |
| WideLoRA-SP | JD-posterior | 77.5 ± 1.0 | 78.3 ± 1.4 | 14.4 ± 0.2 | 34.3 ± 0.2 | +8.3 ± 1.3 |

**C\* = TerraMind + MoE-LoRA** (+9.87 against +8.33 for the design). The margin of 1.54 Δm is inconclusive at 3
seeds (§0), and the pre-registered rule decides on the means.

### 1.4 Per-class AP (%), as submitted, mean over seeds

| config | aircraft | airport | bridge | car | harbor | oil_tank | playground | ship | wind_turbine | tiny-4 mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ViT + LoRA | 13.3 | 10.1 | 0.4 | 41.9 | 7.5 | 11.5 | 0.2 | 27.6 | 1.6 | 3.4 |
| TerraMind + MoE-LoRA | 15.3 | 7.5 | 0.6 | 48.0 | 9.4 | 16.1 | 1.2 | 36.3 | 2.1 | 5.0 |
| A1: WideLoRA alone | 14.0 | 11.5 | 0.5 | 43.5 | 9.1 | 12.4 | 0.3 | 31.5 | 0.9 | 3.6 |
| WideLoRA-SP | 14.9 | 10.9 | 1.0 | 44.9 | 8.9 | 13.7 | 0.9 | 32.2 | 2.1 | 4.4 |

- **The tiny-4 classes (bridge, playground, wind turbine, oil tank).** Bridge, playground and wind turbine are nearly
  undetected on new scenes: AP ≤ 2.1 for every model. Oil tank reaches 11.5–16.1.
- **Scenes per class** (holdout units, DATA.md §6):
  - few: airport 7, wind turbine 9, oil tank 12, car 14;
  - many: bridge 94 and playground 60, but those two are barely detected.
- So read per-class differences in the few-scene classes with care, and compare models on pooled metrics.

## 2. Ablations, and what didn't work

**A1, SP (the spatial-prior branch), paired: same seed, data order and initial weights.** WideLoRA-SP − WideLoRA alone:

| seed | Δ macro-F1 | Δ mAP | Δ Δm |
|---|---:|---:|---:|
| 0 | +0.57 | +0.94 | +4.11 |
| 1 | +3.35 | +0.24 | +3.15 |
| 2 | −0.05 | +0.48 | +1.89 |
| **mean ± std** | +1.29 ± 1.81 | +0.55 ± 0.35 | **+3.05 ± 1.11** |

SP raises mAP on every seed, at +7 % training time (1.20 vs 1.12 h per run) and 0.36 M parameters.

**WideLoRA (unpaired; reference decoding).** A1 − ViT + LoRA = +4.3 Δm: F1 +0.1, mAP +1.1.
- The gain is all in detection, at 30.66 M trainable parameters against 21.00 M.
- The run takes 1.15× the reference's training time (1.12 vs 0.98 h).

**A2, JD.**
- The pre-registered product rule does not help: −0.48 to +0.05 mAP. On this holdout the classifier is only 73–79 %
  accurate (per seed), so multiplying by its probability also suppresses the true boxes of the many misclassified
  images.
- JD-posterior helps a little but consistently (+0.10 to +0.27 mAP), and the cross-fit adopted it.

**What didn't work, or didn't transfer:**
- **The design does not beat TerraMind + MoE-LoRA on this holdout.** TerraMind's EO pretraining gives the best mAP on
  new scenes (+0.9 mAP over the design) but the worst F1 (−2.4). Δm weights mAP's relative change heavily, because
  mAP is small here (12.5), so mAP decides.
- **Tiny classes** (bridge, playground, wind turbine) stay near 0–2 AP for every model on unseen scenes, SP included.
  The wind-turbine holdout is one held-out block of a few tiled wind farms (DATA.md §6).
- **The holdout took several re-draws before it was leak-free.** It had three kinds of leak (REVIEW.md §5, H1–H3):
  flipped revisits, shifted tiles and 20 % strips. An early, leaky split would have flattered every configuration.

## 3. Published baselines (the TA's; a different evaluation set, not comparable with §1)

The TA's results, 3 seeds, trained on all of train, scored on the withheld val (1,426) and test (2,065) images.
CLAUDE.md "Baselines". Δm is against the TA's ViT + LoRA (val 93.31 / 26.13, test 93.99 / 25.41).

| backbone | adapt | trainable backbone params (M) | val macro-F1 | val mAP | val Δm | test macro-F1 | test mAP | test Δm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vit | lora | 0.88 | 93.3 ± 0.4 | 26.1 ± 0.4 | 0.0 ± 0.9 | 94.0 ± 0.2 | 25.4 ± 0.5 | 0.0 ± 0.9 |
| vit | moelora | 1.03 | 92.2 ± 0.8 | 27.1 ± 0.2 | +1.3 ± 0.9 | 93.4 ± 0.5 | 26.8 ± 0.3 | +2.5 ± 0.8 |
| terramind | lora | 0.88 | 92.2 ± 0.4 | 27.0 ± 0.5 | +1.0 ± 1.3 | 92.6 ± 0.4 | 26.5 ± 0.4 | +1.4 ± 1.0 |
| terramind | moelora | 1.03 | 91.9 ± 0.4 | 28.8 ± 0.1 | +4.4 ± 0.3 | 92.4 ± 0.1 | 27.8 ± 0.3 | +3.8 ± 0.7 |
| vit | full (over budget) | 86.0 | 94.4 ± 0.3 | 32.0 ± 0.4 | +11.8 ± 0.7 | 95.8 ± 0.3 | 31.8 ± 0.2 | +13.5 ± 0.4 |

On the TA's val/test, ViT + LoRA reaches mAP 26 and F1 93. On our leak-free holdout the same configuration reaches
mAP 12.5 and F1 76.9. The gap is consistent with val/test sharing scenes with train (DATA.md §5).

**Fit, not generalisation.** The full-train runs are scored on `dataset/repro`, whose val is a 5 % sample of train
(`compare.py` "Other runs"):

| run | macro-F1 | mAP |
|---|---:|---:|
| ViT + LoRA (3 seeds) | 100.0 | 33.8 ± 0.8 |
| TerraMind + MoE-LoRA (`tmfull_s0`) | 97.8 | 35.4 |
| WideLoRA-SP (`Pfull_s0`) | 100.0 | 39.7 |

## 4. Leaderboard val (Phase 1)

- **The leaderboard:** https://web-production-c241e.up.railway.app/. Its Δm uses the TA's val reference
  (93.31 / 26.13).
- **How the zips were made:** `predict.py` builds each one, through `gpu_run.sh`, and it passes `check_submission.py`.
- **The record:** `submissions/SUBMISSIONS.md` lists every upload with its id, and the server's replies are in
  `<zip stem>.response.json` and `<zip stem>.post.json`.

| upload | zip | accuracy | macro-F1 | mAP | AP50 | Δm |
|---|---|---:|---:|---:|---:|---:|
| **submission (pre-registered C\*)**: TerraMind + MoE-LoRA, all of train (`runs/tmfull_s0`), JD-posterior; uploaded and posted 24 Oct 21:39–21:40 SGT, id `s_VlNljpmqLS`; no longer the posted entry | `submissions/submission_val.zip` | 91.23 | 90.92 | 28.85 | 58.78 | +3.92 |
| **WideLoRA-SP (the design)**, all of train (`runs/Pfull_s0`), JD-posterior; uploaded 24 Oct 23:46, id `s_4mpsHTozQ1`, not posted | `submissions/alt_design_widelora_sp_val.zip` | 94.04 | 93.60 | 29.76 | 60.13 | **+7.09** |
| context: ViT + LoRA, all of train (repro queue seed 0), reference decoding; uploaded 23:46, id `s_Oycs9hsMlh`, not posted | `submissions/context_vitlora_repro_s0_val.zip` | 93.48 | 93.19 | 26.32 | 54.92 | +0.30 |
| the TA's TerraMind + MoE-LoRA (published, 3 seeds) | – | 92.2 ± 0.5 | 91.9 ± 0.4 | 28.8 ± 0.1 | 58.7 ± 0.1 | +4.4 ± 0.3 |
| the TA's ViT + LoRA (published, 3 seeds; the leaderboard's reference) | – | 93.6 ± 0.5 | 93.3 ± 0.4 | 26.1 ± 0.4 | 54.7 ± 0.6 | 0.0 ± 0.9 |

- **The 23:46 zips:** both pass `check_submission.py`. They are written compressed (3.5 MB and 1.4 MB) by the
  post-queue jobs (`runs/queue/post_runs.log`, 23:22–23:26), and the user asked for both uploads.
- **Sources:**
  - the scores are the server's replies, `submissions/<zip stem>.response.json`;
  - `submission_val.zip` is a byte-identical copy of the uploaded `submission_val_tmfull_s0_jdposterior.zip`
    (md5 `54695fb689b8c8f428af2195c31eb672`).
- **The submission against the TA's TerraMind + MoE-LoRA:**
  - mAP and AP50 match;
  - macro-F1 is 1.0 lower, so Δm is 0.5 lower;
  - JD-posterior only rescores boxes, so the classification output, and with it F1, is the plain model's. The F1 gap
    is a training difference (one run against the TA's 3-seed mean), not a decoding effect.
- **On val, the design beats TerraMind and the best in-budget reference:**
  - WideLoRA-SP scores +7.09 against TerraMind's +3.92, one full-train seed each;
  - it leads on both metrics: macro-F1 93.60 vs 90.92, mAP 29.76 vs 28.85;
  - it is 2.7 above the TA's TerraMind + MoE-LoRA (+4.4 ± 0.3).
- **Our pipeline reproduces the TA's:** our ViT + LoRA scores +0.30 (93.19 / 26.32), within the TA's 0.0 ± 0.9
  (93.3 ± 0.4 / 26.1 ± 0.4).
- **Holdout against val, the levels:** they differ (holdout Δm +9.9 and +8.3, val +3.92 and +7.09), because the two
  use different references and scenes.
- **Holdout against val, the order:** it reverses too. TerraMind's inconclusive 1.5 Δm lead on the holdout becomes a
  3.2 lead for the design on val.
- **A possible reason, untested:**
  - the holdout has only unseen scenes, which may favour TerraMind's Earth-observation pretraining;
  - val shares scenes with train (DATA.md §5), which may favour the design's 31 M trainable parameters.
- **The team's posted Phase 1 entry** is `s_msr0DiAUXC` (Δm +10.09, macro-F1 90.91, mAP 32.08). It was uploaded at
  22:02 from outside this workspace, and it replaced the TerraMind entry (`submissions/SUBMISSIONS.md`).
- **Step 6:** the pre-registered rule names TerraMind (C\*), and val favours the design. Both feed the choice of the 2
  Phase 2 uploads. Phase 1 allows 5 scored uploads per SGT day, and 4 of 24 Oct's 5 are used (the replies' `quota`).

`check_submission.py` on the submission zip (`uv run python workspace/sarmtl/check_submission.py
workspace/sarmtl/submissions/submission_val.zip --split val`):

```
workspace/sarmtl/submissions/submission_val.zip (val; columns and keys from sample_submission_val.zip)
  zip size: 16.4 MB (limit 25 MB)
  images: 1426
  classification rows: 1426
  labels predicted per class: [148, 173, 142, 92, 114, 169, 171, 157, 260]
  detections: 107060
  detections per image (min / median / max): (4, 100.0, 100)
  images without detections: 0
  images with more than 100 detections (COCO scores the top 100): 0
  detections of the image's predicted class: 83.7%
  score range: (0.0010000744368880987, 0.9999873638153076)
OK: the zip passes every check
```

## 5. Report facts (CLAUDE.md "Report must state")

**Parameters:**

| model | total | trainable | of which: heads | backbone adaptation | other |
|---|---:|---:|---:|---:|---:|
| **TerraMind + MoE-LoRA (the submission)** | **106,493,209** | **21,145,369** | 20,113,177 | 1,032,192 (MoE-LoRA) | – |
| WideLoRA-SP (the design) | 115,956,473 | 31,021,817 | 20,113,177 | 10,543,872 (LoRA 9,437,184 + unfrozen 1,106,688) | SP 364,768 |
| A1: WideLoRA alone | 115,591,705 | 30,657,049 | 20,113,177 | 10,543,872 | – |
| ViT + LoRA (local reference) | 107,039,257 | 20,997,913 | 20,113,177 | 884,736 | – |

- Every model is within the budget: ≤ 130 M total, frozen included, and ≤ 40 M trainable, heads included.
- `train.py` asserts both at startup, and root `train.py` logs the counts.

**Compute:**
- **GPUs:** NVIDIA L40S, 46 GB. The server has 3 (driver 550.163.01).
  - The first repro attempt ran on GPU 0 and was crashed by its ECC memory fault (0.82 GPU-h, counted below).
  - Every later run used GPUs 1 and 2.
  - One GPU per run, no DDP, bf16 autocast.
- **Final training time:**
  - the submission model (`tmfull_s0`, all 9,392 images, 24 epochs) took **1.69 h** of `train_hours`, or 103 min of
    wall-clock for the whole job;
  - the design's full-train run took 1.39 h (85 min of wall-clock).
- **Peak GPU memory** (`max_memory_allocated`):
  - the submission model, 14.2 GiB;
  - the design, 10.4 GiB;
  - the process needs about 2.5 GiB more (PyTorch's caching allocator; docs/DETAILS.md).
- **Inference time per image** (`predict.py`): the 1,426 val images, batch 16, bf16, one L40S, data loading included,
  no scoring. These numbers are like for like; §1.1's holdout times include COCO scoring.
  - TerraMind + MoE-LoRA (the submission): 12.8 ms (`submissions/predict_val_tmfull_s0_jdposterior.log`);
  - WideLoRA-SP (`Pfull_s0`, JD-posterior): 10.5 ms (`runs/queue/post_runs.log`);
  - ViT + LoRA (repro seed 0, reference decoding): 8.9 ms (same log).
- **Total GPU hours** (wall-clock of the GPU-holding jobs; `train_hours` excludes model building and the final test
  pass):

  | item | GPU-h |
  |---|---:|
  | this design's queue (`run.sh`: 14 training runs + 1 smoke evaluation + 9 evaluations; `runs/queue/*/start,end`) | 17.89 |
  | smoke tests 1–3 (`runs/smoke/*.out`) | 0.02 |
  | post-queue jobs: A1's 3 evaluations and the 3 val zips (`runs/queue/post_runs.log`; the submission zip's `predict.py` in `submissions/predict_val_tmfull_s0_jdposterior.log`) | 0.10 |
  | the reference repro on all of train (`workspace/baseline/runs/repro/queue.log`, 3 seeds) | 3.47 |
  | the first repro attempt, crashed by GPU 0's ECC fault (`repro_attempt1_gpu0_ecc_crash/queue.log`) | 0.82 |
  | the TA code's smoke runs (`workspace/baseline/runs/smoke`, 8 configurations; log times) | ≈ 0.06 |
  | **this pipeline, total** | **22.4** |
  | another team session's design (`workspace/detail_lora/`: 17 finished runs, 2 of them smoke; `train_hours` only) | ≥ 21.4 |
  | **the team project, total** (both designs) | **≥ 43.7** |

**Input and preprocessing:**
- Every image is read as one grey channel (`PIL convert('L')`) and resized to 512 × 512 (bilinear with antialias).
  256² and 224² images are upsampled ×2 and ×2.29, and the 17 odd-sized ones are stretched, as in the baseline.
- Pixels are scaled to [0, 1] and normalised with mean 0.2526 and std 0.2133 (train statistics).
- Training augmentation is random horizontal and vertical flips, as in the baseline. There is no TTA: each image goes
  through one forward pass.

**Pretrained models** (both cached; nothing downloaded during the project):
- ViT-B/16: timm `vit_base_patch16_224.augreg_in21k` (Hugging Face `timm/vit_base_patch16_224.augreg_in21k`),
  ImageNet-21k with AugReg. It backs the design and the reference.
- TerraMind-1.0-base: `ibm-esa-geospatial/TerraMind-1.0-base` from IBM and ESA, Earth-observation pretraining, used
  through terratorch's S1GRD input. It backs the submission model.
- **TerraMind and SAR:** TerraMind's pretraining data (TerraMesh) includes Sentinel-1 SAR. The rules allow TerraMind by
  name (CLAUDE.md: "generic pretrained models (ImageNet models, CLIP, TerraMind, …)"), and it is the TA's own baseline.
- No other SAR-pretrained model, no SAR detection model and no other SAR dataset was used.

**External libraries and own code:**
- **External:**
  - PyTorch 2.9.0 (cu126) and torchvision 0.24.0: Faster R-CNN RPN, RoI heads, anchors, RoIAlign, transforms;
  - timm 1.0.24 (ViT), terratorch 1.2.7 (TerraMind);
  - pycocotools 2.0.11 (COCO mAP), NumPy 2.5.4, Pillow 13.0.0, huggingface_hub 1.33.0;
  - for the holdout pipeline (`data/`): OpenCV 5.0.0 (opencv-python-headless), SciPy 1.18.1, pandas 3.1.0.
- **The TA's reference code**, unchanged apart from the data-layout paths. `git diff` shows that `sarbench/data.py`
  and root `train.py` read `dataset/{split}/labels.csv` and `instances.json`.
  - `sarbench/`: data loading, backbones, LoRA and MoE-LoRA, the ViTDet neck with Faster R-CNN, metrics;
  - root `train.py`: training loop, evaluation, schedule;
  - `make_submission.py`.
- **Own code** (`workspace/sarmtl/`):
  - `sarmtl_model.py`: WideLoRA wiring, the SP branch and fused neck, joint decoding;
  - `train.py`: CLI, parameter groups, paired seeding, JD scoring; it imports root `train.py`'s loop;
  - `evaluate.py`, `crossfit.py`, `predict.py`, `compare.py`, `check_submission.py`, `run.sh`;
  - `data/`: the holdout pipeline, i.e. near-duplicate search, group-aware stratified split, leak audits;
  - `tests/`.

**Validation split:**
- A fixed, group-aware, class × image-size stratified 10 % holdout of train (seed 0), drawn by data-eng (DATA.md §6;
  `dataset/holdout_sarmtl/split.json`):
  - 940 holdout images, 8,057 local train, and a 395-image buffer excluded from local train;
  - no near-duplicate link (copies, revisits, overlapping tiles) joins holdout and local train, and the critic found
    0 crossing pairs;
  - one giant wind-turbine component (873 images) has images on both sides, kept apart by an 87-image buffer.
- The holdout has no near-black 256² ship chips: holdout ship AP covers the brighter ship chips.
- The full-train runs use all 9,392 train images.

**Seeds:** 0, 1 and 2 for every holdout configuration; seed 0 for the full-train runs.

**How the Phase 2 submission is chosen:**
- **Pre-registered (DESIGN.md §4 and §6):**
  - C\* is the configuration with the highest 3-seed mean holdout Δm, each scored with its adopted decoding;
  - C\* is trained on all of train (seed 0, last epoch);
  - the same checkpoint makes the val and test zips.
- **At Step 6,** the 2 test uploads are recommended from the holdout and leaderboard val evidence only.
- **The val evidence (§4)** favours the design: +7.09 against TerraMind's +3.92. That is the reverse of the holdout's
  order.

**Datasets to cite:** SARDet-100K (Li et al., NeurIPS 2024), RSAR (Zhang et al., CVPR 2025), M4-SAR (Wang et al.)
and HRSID (Wei et al., IEEE Access 2020), via SARFact (`dataset/SOURCES_AND_LICENSES.md`).

## 6. Sources

- **Holdout runs** (`metrics.json`, and `jd_eval.json` for every one of the 12 runs, in the same directory):
  `workspace/sarmtl/runs/{ref,P,A1,tm}_s{0,1,2}/<config>/`, where `<config>` is:
  - `vit_pretrained_lora` (ref);
  - `vit_widelora_sp` (P);
  - `vit_widelora` (A1);
  - `terramind_pretrained_moelora` (tm).
- **Full-train runs:** `workspace/sarmtl/runs/tmfull_s0/terramind_pretrained_moelora/` (the submission) and
  `workspace/sarmtl/runs/Pfull_s0/vit_widelora_sp/`. The repro reference is
  `workspace/baseline/runs/repro/seed{0,1,2}/vit_pretrained_lora/`.
- **Queue:** `workspace/sarmtl/runs/queue/run.log` (`gpu_run:` and `run.sh:` lines), `<job>.out`, and
  `<job>/{gpu,start,end,rc}`.
- **Tables:** `workspace/sarmtl/runs/tables.md` (`compare.py`).
