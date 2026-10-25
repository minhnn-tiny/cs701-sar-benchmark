# Leaderboard submissions of R50-FPN-MT (team `cs701-team05`)

Leaderboard: https://web-production-c241e.up.railway.app. Scores in %; Δm against the leaderboard reference (val
93.31 / 26.13). Each zip's server reply is in `<zip stem>.response.json`, and the reply to posting it is in
`<zip stem>.post.json`.

| uploaded (SGT) | phase | submission id | zip | model | decoding | Δm | macro-F1 | mAP | accuracy | AP50 | posted |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| 25 Oct 2026 09:09 | 1 (val) | `s_NL2VhcY4aE` | `r50fpn_F_s0_val_reference.zip` | R50-FPN-MT, all of train, seed 0, last epoch (`runs/F_s0/r50fpn`) | reference | +11.13 | 94.07 | 31.74 | 94.39 | 61.41 | yes at 09:09, replacing `s_msr0DiAUXC` (+10.09); replaced at 22:59 |
| 25 Oct 2026 22:59 | 1 (val) | `s_ngdhzNxVaK` | `r50fpn_FE_s0_val_off_d300_nms0.6.zip` | R50-FPN-MT, 36 epochs, all of train, seed 0, last epoch (`runs/FE_s0/r50fpn`) | `off_d300_nms0.6` | +18.45 | 93.90 | 35.61 | 94.11 | 64.53 | yes at 22:59, replacing `s_NL2VhcY4aE` (+11.13) |

## `s_ngdhzNxVaK` (36 epochs)

- **Why this model:** the user's choice on 25 Oct (README.md, Upload).
  - Val rewards how well a model fits its own training images. 36 epochs raised that fit from 36.2 to 42.8 mAP
    (3 seeds each).
  - The 3-seed holdout rule picks the same configuration: E +32.6 ± 1.7 as submitted, against R +30.1 ± 2.5 and
    H +29.3 ± 2.6.
- **Why this decoding:** `crossfit.py` adopted `off_d300_nms0.6` for E, because all 6 out-of-fold gains were positive.
- **Projection vs result:**
  - Projected: about +7.6 Δm over `F_s0`, from val ≈ 0.65 × fit + 0.35 × leak-free holdout.
  - Got: +7.32. mAP rose 3.87 and macro-F1 fell 0.17.
  - On `dataset/repro`'s sample of train, mAP is 43.99 against `F_s0`'s 37.96.
- **Checks before upload:**
  - `predict.py` and the upload helper each ran `check_submission.py`, and both passed.
  - The zip is 3.95 MB. 824 images have more than 100 detections, and the scorer counts the top 100 per image and
    class.
- **Posting rule:** +18.45 beats both the posted +11.13 and the TA's +4.4, so it was posted.
- **Quota after it:** 3 of 25 Oct's 5 used, and both reset cards unused. One of the 3, `s_BOeZvQVtbv` (11:15, +4.60,
  not posted), is detail_lora's (`workspace/detail_lora/submissions/SUBMISSIONS.md`).
- **Board at 23:00:** 5th of 5. The four teams ahead are team09 +41.66, team02 +40.64, team04 +39.12 and team03
  +38.51 (mAP 45.8–47.4).
- **Commands:** the `FE_s0` line of `jobs/plan.txt` (`train.py --data dataset/repro --epochs 36`, then `predict.py
  --decoding off_d300_nms0.6`).

## `s_NL2VhcY4aE` (24 epochs)

- **Why this model and decoding:**
  - The base configuration is the only one with seeds 0, 1 and 2 on the holdout: local Δm +29.3 ± 2.6, against
    TerraMind + MoE-LoRA's +9.9 on the same holdout.
  - `crossfit.py` kept the reference decoding. In the fold that chooses on half B, `posterior_d300_nms0.7` lost
    0.12 mAP on half A for seed 2, so not all 6 out-of-fold gains were positive.
- **Checks before upload:** `predict.py` ran `check_submission.py`, and the upload script ran it again; both passed.
  The zip is 0.6 MB, with 30,570 detections.
- **How it was sent:** the token went to curl through stdin and never appeared in any output. However, the helper's
  check that a reply holds no token ran `grep` with the token as an argument. So the token was visible to `ps` for a
  few milliseconds per call, at this upload and at the `/api/me` and list calls. Since 25 Oct 22:58, `grep` reads it
  from a pattern file that the `printf` builtin writes.
- **Posting rule** (CLAUDE.md, "Submission rule"): +11.13 beats both the team's previous entry (+10.09) and the TA's
  best in-budget val baseline (+4.4), so it was posted.
- **Quota after it:** 1 of 25 Oct's 5 used, and both reset cards unused.
- **Board at 09:10:** 5th of 5 posted teams. Teams 04, 02, 03 and 09 are at Δm +38.4 to +39.1 (mAP 45.5–45.9).

Commands (from the repo root):

```bash
uv run python workspace/r50fpn/crossfit.py      # -> adopted: reference
HF_HUB_OFFLINE=1 workspace/gpu_run.sh --min-free 8000 -- .venv/bin/python workspace/r50fpn/predict.py \
    workspace/r50fpn/runs/F_s0/r50fpn --split val --decoding reference \
    --out workspace/r50fpn/submissions/r50fpn_F_s0_val_reference.zip
```
