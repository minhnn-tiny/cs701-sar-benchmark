# Leaderboard submissions of R50-FPN-MT (team `cs701-team05`)

Leaderboard: https://web-production-c241e.up.railway.app. Scores in %; Δm against the leaderboard reference (val
93.31 / 26.13). Each zip's server reply is in `<zip stem>.response.json`, and the reply to posting it is in
`<zip stem>.post.json`.

| uploaded (SGT) | phase | submission id | zip | model | decoding | Δm | macro-F1 | mAP | accuracy | AP50 | posted |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| 25 Oct 2026 09:09 | 1 (val) | `s_NL2VhcY4aE` | `r50fpn_F_s0_val_reference.zip` | R50-FPN-MT, all of train, seed 0, last epoch (`runs/F_s0/r50fpn`) | reference | +11.13 | 94.07 | 31.74 | 94.39 | 61.41 | yes at 09:09, replacing `s_msr0DiAUXC` (+10.09) |

- **Why this model and decoding:**
  - The base configuration is the only one with seeds 0, 1 and 2 on the holdout: local Δm +29.3 ± 2.6, against
    TerraMind + MoE-LoRA's +9.9 on the same holdout.
  - `crossfit.py` kept the reference decoding. In the fold that chooses on half B, `posterior_d300_nms0.7` lost
    0.12 mAP on half A for seed 2, so not all 6 out-of-fold gains were positive.
- **Checks before upload:** `predict.py` ran `check_submission.py`, and the upload script ran it again; both passed.
  The zip is 0.6 MB, with 30,570 detections.
- **How it was sent:** the token went to curl through stdin, so it never appeared in a process's arguments or in any
  output.
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
