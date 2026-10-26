#!/usr/bin/env bash
# wait_run.sh <job name>: wait for the mtdet queue to finish the job, then summarise it (run from the repo root)
N=$1
until grep -q "end $N " workspace/mtdet/runs/queue/run.log 2>/dev/null; do sleep 30; done
grep "end $N " workspace/mtdet/runs/queue/run.log
grep "val:" workspace/mtdet/runs/$N/log.txt 2>/dev/null | sed 's/lf_macro_f1.*lf_mAP/| lf_mAP/; s/lf_AP50.*fit_mAP/| fit_mAP/; s/fit_AP50.*//' | cut -c1-200
grep "training " workspace/mtdet/runs/$N/log.txt 2>/dev/null
grep -E "delta_m [+-]|OK:|FAILED|Error|error" workspace/mtdet/runs/queue/$N.out | tail -14
uv run python workspace/mtdet/compare.py --per-class 2>/dev/null | grep -A2 "^$N "
