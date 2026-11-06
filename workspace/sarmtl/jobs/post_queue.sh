#!/usr/bin/env bash
# The GPU jobs after the queue (jobs/plan.txt), from the repo root, each through gpu_run.sh:
# - A1's decodings on the holdout (evaluate.py), whose jd_eval.json files feed the cross-fit;
# - the val zips (predict.py).
# On 24 Oct they ran as two scratchpad scripts and one command by hand. Their gpu_run: lines are in
# runs/queue/post_runs.log and submissions/predict_val_tmfull_s0_jdposterior.log.
# Usage: nohup setsid workspace/sarmtl/jobs/post_queue.sh >> workspace/sarmtl/runs/queue/post_runs.log 2>&1 &
set -e
export HF_HUB_OFFLINE=1
G=(workspace/gpu_run.sh --min-free 8000 --)
P=.venv/bin/python
S=workspace/sarmtl/submissions
for s in 0 1 2; do "${G[@]}" $P workspace/sarmtl/evaluate.py workspace/sarmtl/runs/A1_s$s/vit_widelora; done
# The submission (C*, DESIGN.md §6). The uploaded zip was written to submission_val_tmfull_s0_jdposterior.zip, then
# copied to submission_val.zip; a rebuild gets its own name, so it never overwrites the uploaded zip's copy.
"${G[@]}" $P workspace/sarmtl/predict.py workspace/sarmtl/runs/tmfull_s0/terramind_pretrained_moelora --split val \
    --jd posterior --out $S/submission_val_rebuild.zip
# Optional: val evidence for the leaderboard (RESULTS.md §4)
"${G[@]}" $P workspace/sarmtl/predict.py workspace/sarmtl/runs/Pfull_s0/vit_widelora_sp --split val --jd posterior \
    --out $S/alt_design_widelora_sp_val.zip
"${G[@]}" $P workspace/sarmtl/predict.py workspace/baseline/runs/repro/seed0/vit_pretrained_lora --split val --jd off \
    --out $S/context_vitlora_repro_s0_val.zip
