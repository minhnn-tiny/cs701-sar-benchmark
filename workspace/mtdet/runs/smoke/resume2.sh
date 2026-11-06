#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../../.."
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled
T=".venv/bin/python workspace/mtdet/train.py --arch yolo26l --epochs 2 --limit 64 --max-steps 3 --eval-every 2"
o=workspace/mtdet/runs/smoke/fix_resume2; rm -rf $o
timeout 600 $T --out $o > $o.log 2>&1 & pid=$!
while [ ! -f $o/last.pt ]; do sleep 1; kill -0 $pid 2>/dev/null || break; done; kill $pid; wait $pid 2>/dev/null
$T --out $o >> $o.log 2>&1 && echo "resume with the same args: $(grep -c 'resumed after epoch 1' $o/log.txt) resume line, metrics.json $(test -f $o/metrics.json && echo yes)" || { echo FAILED; tail -n 5 $o.log; }
