# Leaderboard submissions of MTDet (team `cs701-team05`)

Leaderboard: https://web-production-c241e.up.railway.app. Scores in %; Δm against the leaderboard reference (val
93.31 / 26.13). Each zip's server reply is in `<zip stem>.response.json`, and the reply to posting it is in
`<zip stem>.post.json`.

| uploaded (SGT) | phase | submission id | zip | model | decoding | Δm | macro-F1 | mAP | accuracy | AP50 | posted |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| 26 Oct 2026 08:59 | 1 (val) | `s_DRkVHQAQAo` | `YA_s0_val.zip` | YOLO26-L multi-task, 100 epochs, all of train, seed 0, EMA weights (`runs/YA_s0`) | `o2m_nms0.7_a0_k100` | +35.81 | 92.06 | 45.19 | 92.29 | 70.08 | yes at 08:59, replacing r50fpn's `s_ngdhzNxVaK` (+18.45) |
| 26 Oct 2026 15:24 | 1 (val) | `s_6bBQH7vFS2` | `YA150_val.zip` | YOLO26-L multi-task, 150 epochs, all of train, seed 0, EMA weights (`runs/YA150`) | `o2m_nms0.7_a0_k100` | +37.98 | 92.38 | 46.24 | 92.64 | 70.96 | yes at 15:24, replacing `s_DRkVHQAQAo` (+35.81) |
| 27 Oct 2026 05:58 | 1 (val) | `s_iNmyE4oAin` | `YA250_val.zip` | YOLO26-L multi-task, 250 epochs, all of train, seed 0, EMA weights (`runs/YA250`) | `o2m_nms0.7_a0_k100` | +39.93 | 92.43 | 47.25 | 92.64 | 70.96 | yes at 05:59, replacing `s_6bBQH7vFS2` (+37.98) |
| 27 Oct 2026 11:33 | 1 (val) | `s_aUZ9SopRu8` | `YA250_n8t0_val.zip` | the same `runs/YA250` model, decoded without the score floor | `o2m_nms0.8_a0_k100_t0` | +41.11 | 92.43 | 47.86 | 92.64 | 71.61 | yes at 11:34, replacing `s_iNmyE4oAin` (+39.93) |
| 27 Oct 2026 13:49 | 1 (val) | `s_VNaV9r0uDV` | `YASA250_n8t0_val.zip` | YOLO26-L multi-task, 250 epochs, SAR speckle and gamma augmentation, all of train, seed 0, EMA weights (`runs/YASA250`) | `o2m_nms0.8_a0_k100_t0` | +43.50 | 93.40 | 48.84 | 93.55 | 72.56 | yes at 13:49, replacing `s_aUZ9SopRu8` (+41.11) |
| 28 Oct 2026 04:58 | 1 (val) | `s_dlz80pdFun` | `YASA350_val.zip` | YOLO26-L multi-task, 350 epochs, SAR speckle and gamma augmentation, all of train, seed 0, EMA weights (`runs/YASA350`) | `o2m_nms0.8_a0_k100_t0` | +43.56 | 92.46 | 49.14 | 92.64 | 72.96 | yes at 04:58, replacing `s_VNaV9r0uDV` (+43.50) |

## `s_DRkVHQAQAo` (YOLO26-L, 100 epochs)

- **Why this model:** an interim upload, made before the design's screens finish, to measure how ND transfers to val
  (`jobs/plan.txt`).
  - Its configuration, `Y_s0`, had the best ND Δm at the time: +36.48 as decoded here, from one seed.
  - The 3-seed rule (CLAUDE.md) decides the design's final upload.
- **Why this decoding:** M1.1 in `EXPERIMENTS.md`. On ND, the one-to-many head with per-class NMS 0.7 beat YOLO26's
  native one-to-one decoding by 1.9 Δm (`Y_s0`) and 2.7 (`YP2_s0`).
- **Projection vs result:**
  - FitDet's rule, val ≈ ND + 2, projected +38.5.
  - Got +35.81, which is ND − 0.67:
    - mAP was 0.29 above ND's 44.90;
    - macro-F1 was 2.28 below ND's 94.34.
  - On ND, classification accuracy is 85.9 % on the 270 images with no near-duplicate partner in local train, against
    97.7–99.1 % on the copies and crops (`analyze.py`). Val seems to have more of the former.
- **Checks before upload:**
  - `predict.py` and the upload helper each ran `check_submission.py`, and both passed.
  - The zip is 0.57 MB and holds 37,715 detections, with a median of 9 per image, a maximum of 100 and every score
    ≥ 0.001.
  - Per class, the predicted label counts are within 8 images of the two r50fpn uploads.
- **Posting rule:** +35.81 beats both the posted +18.45 and the TA's +4.4, so it was posted.
- **Quota after it:** 1 of 26 Oct's 5 used, and both reset cards unused.
- **Board at 09:00:** 5th of 5. Ahead:

  | team | Δm | macro-F1 | mAP |
  |---|---:|---:|---:|
  | team09 | +41.66 | 95.29 | 47.35 |
  | team02 | +40.64 | 94.16 | 47.13 |
  | team04 | +39.12 | 95.89 | 45.86 |
  | team03 | +38.51 | 95.11 | 45.76 |

  The 5.86 Δm to team09 splits into about 4.1 from mAP and 1.7 from macro-F1.
- **Commands:** the `YA_s0` and `YA_val` lines of `jobs/plan.txt`:
  - `train.py --arch yolo26l --epochs 100 --seed 0 --data dataset/repro --evals nd`;
  - then `predict.py runs/YA_s0 --split val --decoding o2m_nms0.7_a0_k100`.

## `s_6bBQH7vFS2` (YOLO26-L, 150 epochs)

- **Why this model:** its configuration, `Y150_s0`, had the best ND Δm: +40.04 as decoded here, from one seed.
  - 150 epochs beat 100 by +3.54 Δm on ND, and the paired unit bootstrap puts it at [+1.38, +5.39] (`EXPERIMENTS.md`).
  - Like `s_DRkVHQAQAo`, it is an interim upload: the 3-seed rule decides the design's final one.
- **Projection vs result:** ND +40.04 gave val +37.98, ND − 2.06 (`YA_s0`: ND − 0.67).
  - Against `s_DRkVHQAQAo`: mAP 46.24, up 1.05; macro-F1 92.38, up 0.32.
- **Checks before upload:**
  - `predict.py` and the upload helper each ran `check_submission.py`, and both passed.
  - The zip is 0.41 MB and holds 26,189 detections, with a median of 7 per image, a maximum of 100 and every score
    ≥ 0.001; 8 images have none.
- **Posting rule:** +37.98 beats both the posted +35.81 and the TA's +4.4, so it was posted.
- **Quota after it:** 3 of 26 Oct's 5 used (`s_HZe1Hx0Bms` at 11:18, +20.24, is not MTDet's), and both reset cards unused.
- **Board at 15:25:** 5th of 6. Ahead:

  | team | Δm | macro-F1 | mAP |
  |---|---:|---:|---:|
  | team09 | +41.66 | 95.29 | 47.35 |
  | team02 | +40.64 | 94.16 | 47.13 |
  | team04 | +39.12 | 95.89 | 45.86 |
  | team03 | +38.51 | 95.11 | 45.76 |

  Our mAP is the third best, but our macro-F1 is the lowest of the five. The 3.68 Δm to team09 splits into about 2.1
  from mAP (1.1 points) and 1.6 from macro-F1 (2.9 points).
- **Commands:** the `YA150` and `YA150_val` lines of `jobs/plan.txt`:
  - `train.py --arch yolo26l --epochs 150 --seed 0 --data dataset/repro --evals nd`;
  - then `predict.py runs/YA150 --split val --decoding o2m_nms0.7_a0_k100`.

## `s_iNmyE4oAin` (YOLO26-L, 250 epochs)

- **Why this model:** its configuration, `Y250_s0`, had the best ND Δm when it was queued: +43.23 as decoded here, from
  one seed, +3.19 over `Y150_s0`. Interim, like the two before it: the 3-seed rule decides the design's final upload.
- **Projection vs result:** ND +43.23 gave val +39.93, ND − 3.30 (`YA150`: ND − 2.06; `YA_s0`: ND − 0.67).
  - Against `s_6bBQH7vFS2`: mAP 47.25, up 1.01 (ND projected +1.64); macro-F1 92.43, up 0.05 (ND: +0.08).
  - Each longer schedule transfers less: val gains 61 % of ND's gain from 150 to 250 epochs, which fits the gain being
    fitting of training scenes (FIT mAP 75.9 → 85.0).
- **Checks before upload:**
  - `predict.py` and the upload helper each ran `check_submission.py`, and both passed.
  - The zip is 0.19 MB and holds 11,690 detections (`YA150`: 26,189), with a median of 4 per image, a maximum of 93 and
    every score ≥ 0.001; 18 images have none. The 250-epoch model is more confident, so fewer boxes clear the 0.001
    floor; ND was scored with the same decoding.
- **Posting rule:** +39.93 beats both the posted +37.98 and the TA's +4.4, so it was posted.
- **Quota after it:** 2 of 27 Oct's 5 used (FitDet's `s_3lHu6mo2fp` at 02:12, +36.82, is not MTDet's), and both reset
  cards unused.
- **Board at 05:59:** 3rd of 6. Ahead:

  | team | Δm | macro-F1 | mAP |
  |---|---:|---:|---:|
  | team09 | +41.66 | 95.29 | 47.35 |
  | team02 | +40.64 | 94.16 | 47.13 |

  Our mAP is 0.10 below team09's, so the 1.73 Δm to it is almost all macro-F1: 2.86 points, about 1.5 Δm.
- **Commands:** the `YA250` line of `jobs/plan.txt`:
  - `train.py --arch yolo26l --epochs 250 --seed 0 --data dataset/repro --evals nd`;
  - then `predict.py runs/YA250 --split val --decoding o2m_nms0.7_a0_k100 --out submissions/YA250_val.zip`.

## `s_aUZ9SopRu8` (YOLO26-L, 250 epochs, no score floor)

- **Why:** the same model as `s_iNmyE4oAin`, decoded with the best decoding of the sweep by the mean ND Δm of the three
  finished 250-epoch runs (`Y250_s0`, `Y250_s1`, `YSA250_s0`): `o2m_nms0.8_a0_k100_t0`, +44.12 against +43.27 for the
  decoding of `s_iNmyE4oAin` (`EXPERIMENTS.md` M1.6). It keeps each image's 100 best boxes, however low their score,
  with NMS at IoU 0.8; it beat the next best, `o2m_nms0.7_a0_k100_t0`, on all three runs (+0.14 to +0.19). The
  decoding was chosen on ND only; val decided nothing but the posting.
- **Projection vs result:** on the two plain 250-epoch runs, the decoding added ND +1.01 and +0.69 Δm (mAP +0.53 and
  +0.37). On val it added +1.18 (mAP 47.25 → 47.86, +0.61; macro-F1 unchanged, since the classifier is untouched): the
  full ND gain or more, as expected for a change of decoding rather than of training.
- **Checks before upload:** `predict.py` and the upload helper each ran `check_submission.py`, and both passed. The zip
  is 1.9 MB with exactly 100 detections per image (142,600) and scores down to 0.0; no image lacks detections.
- **Posting rule:** +41.11 beats both the posted +39.93 and the TA's +4.4, so it was posted.
- **Quota after it:** 4 of 27 Oct's 5 used (FitDet's `s_3lHu6mo2fp` at 02:12, `s_iNmyE4oAin` at 05:58, another
  session's `s_YbqE0XcDYU` at 07:31, +37.90, not MTDet's, and this one); both reset cards unused.
- **Board at 11:34:** 2nd of 6, behind team09 (+41.66, macro-F1 95.29, mAP 47.35). Our mAP is the board's best; the
  0.55 Δm gap is macro-F1 (2.86 points).
- **Commands:** after the `YA250` job (see `s_iNmyE4oAin`),
  `workspace/gpu_run.sh --gpus 2 --min-free 8000 -- env HF_HUB_OFFLINE=1 .venv/bin/python workspace/mtdet/predict.py
  workspace/mtdet/runs/YA250 --split val --decoding o2m_nms0.8_a0_k100_t0 --out workspace/mtdet/submissions/YA250_n8t0_val.zip`
  (output: `runs/queue/YA250_n8t0.out`).

## `s_VNaV9r0uDV` (YOLO26-L, 250 epochs, SAR augmentation)

- **Why this model:** SAR speckle and gamma augmentation (`--sar-aug`) won its single-seed comparison at 150 and 250
  epochs. At the submitted decoding, `YSA250_s0` scores ND +45.03 against `Y250_s0`'s +44.24. Interim, like the uploads
  before it: seeds 1 and 2 of both configurations are still running, and the 3-seed rule decides the design's final
  upload.
  Confirmed at 21:08 on 27 Oct: the 3-seed means are +45.63 ± 0.71 (SAR) against +43.42 ± 0.72 (plain), with SAR
  ahead on every seed, so this is the design's best configuration until round 3 (350 epochs) reports.
- **Projection vs result:** ND gave SAR +0.79 on one seed; val gave +2.39 over `s_aUZ9SopRu8`, the same decoding
  without SAR augmentation (macro-F1 92.43 → 93.40, mAP 47.86 → 48.84). The val gain is three times the ND gain, so
  the seeds have to confirm it.
- **Training:** epochs 1–138 ran before the 27 Oct reboot, and epochs 139–250 resumed from `last.pt` (same args; the
  data order depends only on seed and epoch). 6.01 h wall, about 4.0 GPU h, since it shared its GPU for most of it.
- **Checks before upload:** `predict.py` and the upload helper each ran `check_submission.py`, and both passed. The
  zip is 1.9 MB, with exactly 100 detections per image and scores down to 0.0.
- **Posting rule:** +43.50 beats both the posted +41.11 and the TA's +4.4, so it was posted.
- **Quota after it:** 5 of 27 Oct's 5 used; both reset cards unused.
- **Board at 13:49:** **1st of 6**, ahead of team09 (+41.66, macro-F1 95.29, mAP 47.35). Our mAP leads by 1.49; our
  macro-F1 trails theirs by 1.89.
- **Commands:**
  - Training: the `YASA250` line of `jobs/plan.txt`, resumed after the reboot by its `YASA250_r2` line:
    `train.py --arch yolo26l --epochs 250 --seed 0 --sar-aug --data dataset/repro --evals nd`.
  - The zip: the `YASA250_n8` line: `predict.py runs/YASA250 --split val --decoding o2m_nms0.8_a0_k100_t0`.

## `s_dlz80pdFun` (YOLO26-L, 350 epochs, SAR augmentation)

- **Why this model:** the design's best configuration by the 3-seed rule since 28 Oct 03:55. SAR augmentation at 350
  epochs scored holdout +47.23 ± 0.79 against +45.63 ± 0.71 at 250 epochs, ahead on every seed.
- **Projection vs result:** on val it added only +0.06 over `s_VNaV9r0uDV` (mAP 48.84 → 49.14, +0.30; macro-F1
  93.40 → 92.46, −0.94), where the holdout gave +1.60 (mAP +0.79, F1 +0.17). Earlier schedules kept ~61 % of the
  holdout gain on val; this one kept ~4 %.
  - The F1 drop is the classifier of one all-train seed against another.
  - The mAP gain shrinking says val rewards a longer schedule less from 350 epochs on.
- **Checks before upload:** `predict.py` and the upload helper each ran `check_submission.py`, and both passed. The
  zip is 1.8 MB, with exactly 100 detections per image and scores down to 0.0. The job's decoding was confirmed from
  `runs/queue/YASA350/cmd`.
- **Posting rule:** +43.56 beats both the posted +43.50 and the TA's +4.4, so it was posted.
- **Quota after it:** 1 of 28 Oct's 5 used; both reset cards unused.
- **Board at 04:58:** **1st of 6**, ahead of team09 (+42.29, macro-F1 95.32, mAP 47.67, up from +41.66).
- **Commands:** the `YASA350` line of `jobs/plan.txt`:
  - `train.py --arch yolo26l --epochs 350 --seed 0 --sar-aug --data dataset/repro --evals nd`;
  - then `predict.py runs/YASA350 --split val --decoding o2m_nms0.8_a0_k100_t0`. It ran on GPU 0 beside tom's
    job for its first hour, with the user's go-ahead.
