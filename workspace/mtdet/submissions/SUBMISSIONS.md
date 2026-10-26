# Leaderboard submissions of MTDet (team `cs701-team05`)

Leaderboard: https://web-production-c241e.up.railway.app. Scores in %; Δm against the leaderboard reference (val
93.31 / 26.13). Each zip's server reply is in `<zip stem>.response.json`, and the reply to posting it is in
`<zip stem>.post.json`.

| uploaded (SGT) | phase | submission id | zip | model | decoding | Δm | macro-F1 | mAP | accuracy | AP50 | posted |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| 26 Oct 2026 08:59 | 1 (val) | `s_DRkVHQAQAo` | `YA_s0_val.zip` | YOLO26-L multi-task, 100 epochs, all of train, seed 0, EMA weights (`runs/YA_s0`) | `o2m_nms0.7_a0_k100` | +35.81 | 92.06 | 45.19 | 92.29 | 70.08 | yes at 08:59, replacing r50fpn's `s_ngdhzNxVaK` (+18.45) |
| 26 Oct 2026 15:24 | 1 (val) | `s_6bBQH7vFS2` | `YA150_val.zip` | YOLO26-L multi-task, 150 epochs, all of train, seed 0, EMA weights (`runs/YA150`) | `o2m_nms0.7_a0_k100` | +37.98 | 92.38 | 46.24 | 92.64 | 70.96 | yes at 15:24, replacing `s_DRkVHQAQAo` (+35.81) |

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
