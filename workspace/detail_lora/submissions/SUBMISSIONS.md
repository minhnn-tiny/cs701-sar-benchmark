# DetailLoRA leaderboard submissions

Leaderboard: https://web-production-c241e.up.railway.app/ (team `cs701-team05`). Each zip's upload response is in
`<zip name>.response.json`, its idempotency key in `<zip name>.idempotency_key`.

| uploaded (SGT) | phase | submission id | zip | model | decoding | Δm | macro-F1 | mAP | accuracy | AP50 | posted |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| 25 Oct 2026 11:15 | 1 (val) | `s_BOeZvQVtbv` | `submission_val_tmd_full_seed0_soft.zip` | TerraMind + MoE-LoRA + detail + anchors (the best holdout configuration, README step 4b), all of train, seed 0, last epoch (`runs/full_seed0/detail_terramind_moelora_noaug`) | soft | +4.60 | 91.68 | 28.99 | 92.01 | 59.04 | no |

- **Why it was uploaded**: CLAUDE.md submission rule (25 Oct): once a design's runs are done, upload its best model.
  The user said "upload it". `check_submission.py` passed first (4.9 MB, 1,426 images, 32,199 detections).
- **Why it isn't posted**: +4.60 beats the TA's in-budget bar (+4.4 on val) but not the team's posted entry
  `s_msr0DiAUXC` (+10.09), so the board was left as it was.
- **Quota after it**: 2 of 25 Oct's 5 used (the other one wasn't from this workspace), both reset cards unused.
- **Holdout vs val**: the 3-seed holdout Δm was +5.8 ± 1.1 (soft +5.9) against our own ViT + LoRA runs on
  `dataset/holdout/`; the val score is against the leaderboard reference (93.31 / 26.13).

Command (from the repo root):

```bash
HF_HUB_OFFLINE=1 workspace/gpu_run.sh --gpus "1 2" --min-free 8000 -- uv run python workspace/detail_lora/predict.py \
    workspace/detail_lora/runs/full_seed0/detail_terramind_moelora_noaug --split val --fusion soft \
    --out workspace/detail_lora/submissions/submission_val_tmd_full_seed0_soft.zip
```
