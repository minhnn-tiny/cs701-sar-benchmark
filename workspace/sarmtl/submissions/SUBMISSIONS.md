# Leaderboard submissions (team `cs701-team05`)

Leaderboard: https://web-production-c241e.up.railway.app (replaced Codabench on 24 Oct 2026). Scores in %; Δm
against the leaderboard reference (val 93.31 / 26.13). Each zip's server reply is in `<zip stem>.response.json`, and
the reply to posting it is in `<zip stem>.post.json`.

| uploaded (SGT) | phase | submission id | zip | model | decoding | Δm | macro-F1 | mAP | accuracy | AP50 | posted |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| 24 Oct 2026 21:39 | 1 (val) | `s_VlNljpmqLS` | `submission_val_tmfull_s0_jdposterior.zip` | C\* = TerraMind + MoE-LoRA, all of train, seed 0, last epoch (`runs/tmfull_s0/terramind_pretrained_moelora`) | JD posterior | +3.92 | 90.92 | 28.85 | 91.23 | 58.78 | yes at 21:40, replaced by `s_msr0DiAUXC` (below) |
| 24 Oct 2026 23:46 | 1 (val) | `s_4mpsHTozQ1` | `alt_design_widelora_sp_val.zip` | WideLoRA-SP (the design), all of train, seed 0, last epoch (`runs/Pfull_s0/vit_widelora_sp`) | JD posterior | +7.09 | 93.60 | 29.76 | 94.04 | 60.13 | no |
| 24 Oct 2026 23:46 | 1 (val) | `s_Oycs9hsMlh` | `context_vitlora_repro_s0_val.zip` | ViT + LoRA (the TA's reference configuration), all of train, seed 0, last epoch (`workspace/baseline/runs/repro/seed0/vit_pretrained_lora`) | reference | +0.30 | 93.19 | 26.32 | 93.48 | 54.92 | no |

- **The two 23:46 uploads** were asked for by the user. They passed `check_submission.py` before upload.
- **How they were sent:** the token went to curl through stdin, so it never appeared in a process's arguments or in
  any output.
- **Quota after them:** 4 of 24 Oct's 5 used, and both reset cards unused.
- **Below the TA's bar** (CLAUDE.md, "Submission rule"): `s_VlNljpmqLS` (+3.92) and `s_Oycs9hsMlh` (+0.30) are
  under TerraMind + MoE-LoRA's val +4.4, so neither may be posted. Neither is posted.

**The team's posted Phase 1 entry is not from this workspace.** `/api/me` at 23:45 shows `s_msr0DiAUXC`:
- uploaded 24 Oct 2026 22:02 SGT and posted, 5,898,998 bytes;
- Δm +10.09, macro-F1 90.91, mAP 32.08, accuracy 91.16, AP50 61.76;
- no zip or record of it exists under `workspace/`, so it was uploaded from elsewhere with the team token.

For the TA's reproduction, its code and commands must be kept by whoever made it.

C\* was chosen by DESIGN.md §4 rule 4 from `compare.py` (`runs/tables.md`): local holdout Δm, as submitted,
TerraMind + MoE-LoRA +9.9 ± 2.4 vs WideLoRA-SP +8.3 ± 1.3 (inconclusive under rule 2; the mean decides).

Command (from the repo root):

```bash
HF_HUB_OFFLINE=1 workspace/gpu_run.sh --min-free 8000 -- uv run python workspace/sarmtl/predict.py \
    workspace/sarmtl/runs/tmfull_s0/terramind_pretrained_moelora --split val --jd posterior \
    --out workspace/sarmtl/submissions/submission_val_tmfull_s0_jdposterior.zip
```
