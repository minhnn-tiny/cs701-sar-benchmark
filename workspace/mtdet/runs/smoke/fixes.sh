#!/usr/bin/env bash
# the review fixes on GPU: a held-out run logs its overlap check; rerunning it is a no-op; other args are refused;
# an interrupted run refuses to resume with other args and resumes with the same ones; an all-train run is flagged
set -uo pipefail
cd "$(dirname "$0")/../../../.."
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled
T=".venv/bin/python workspace/mtdet/train.py --arch yolo26l --epochs 2 --limit 64 --max-steps 3 --eval-every 2"
o=workspace/mtdet/runs/smoke/fix_done; rm -rf $o
$T --out $o > $o.log 2>&1 && grep -o "images shared with train.*" $o/log.txt
$T --out $o 2>&1 | tail -n 1                                   # -> already complete
$T --out $o --seed 1 2>&1 | tail -n 1                          # -> refused (other args)
o=workspace/mtdet/runs/smoke/fix_resume; rm -rf $o
timeout 600 $T --out $o > $o.log 2>&1 & pid=$!
while [ ! -f $o/last.pt ]; do sleep 1; kill -0 $pid 2>/dev/null || break; done; kill $pid; wait $pid 2>/dev/null
$T --out $o --lr 1e-3 2>&1 | tail -n 1                         # -> refused to resume
$T --out $o >> $o.log 2>&1 && grep -c "resumed after epoch 1" $o/log.txt
o=workspace/mtdet/runs/smoke/fix_alltrain; rm -rf $o
$T --data dataset/repro --evals nd --out $o > $o.log 2>&1; grep -o "images shared with train.*" $o/log.txt
.venv/bin/python -c "import json; m = json.load(open('$o/metrics.json')); print('held_out', m['held_out'], m['eval_overlap'])"
