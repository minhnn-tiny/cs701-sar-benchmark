#!/usr/bin/env bash
# functional smoke: 2 short epochs per architecture, evaluation each epoch, then a resume check (kill after epoch 1)
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled
for arch in yolo26l rtdetr-l; do
  out=workspace/mtdet/runs/smoke/$arch; rm -rf "$out"
  .venv/bin/python workspace/mtdet/train.py --arch $arch --epochs 2 --limit 96 --eval-every 1 --out "$out" > "$out.log" 2>&1 \
    && echo "$arch OK: $(grep -c 'val:' "$out.log") evals, $(tail -n 1 "$out/log.txt")" || { echo "$arch FAILED"; tail -n 30 "$out.log"; }
done
# resume: run 1 epoch of a 2-epoch schedule, stop, rerun -> must resume after epoch 1
out=workspace/mtdet/runs/smoke/resume; rm -rf "$out"
timeout 600 .venv/bin/python workspace/mtdet/train.py --arch yolo26l --epochs 2 --limit 64 --eval-every 2 --max-steps 3 --out "$out" > "$out.log" 2>&1 &
pid=$!
while [ ! -f "$out/last.pt" ]; do sleep 2; kill -0 $pid 2>/dev/null || break; done
kill $pid 2>/dev/null; wait $pid 2>/dev/null || true
.venv/bin/python workspace/mtdet/train.py --arch yolo26l --epochs 2 --limit 64 --eval-every 2 --max-steps 3 --out "$out" >> "$out.log" 2>&1 \
  && echo "resume OK: $(grep -c 'resumed after epoch 1' "$out/log.txt") resume line(s), metrics $(test -f "$out/metrics.json" && echo yes)" || { echo "resume FAILED"; tail -n 20 "$out.log"; }
