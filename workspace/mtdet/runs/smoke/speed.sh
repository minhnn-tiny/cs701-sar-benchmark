#!/usr/bin/env bash
# resume check, then step time per architecture, default vs channels_last (125 steps on the first 2,000 images)
set -uo pipefail
cd "$(dirname "$0")/../../../.."
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled
out=workspace/mtdet/runs/smoke/resume; rm -rf "$out"
timeout 600 .venv/bin/python workspace/mtdet/train.py --arch yolo26l --epochs 2 --limit 64 --eval-every 2 --max-steps 3 --out "$out" > "$out.log" 2>&1 &
pid=$!
while [ ! -f "$out/last.pt" ]; do sleep 1; kill -0 $pid 2>/dev/null || break; done
kill $pid 2>/dev/null; wait $pid 2>/dev/null
.venv/bin/python workspace/mtdet/train.py --arch yolo26l --epochs 2 --limit 64 --eval-every 2 --max-steps 3 --out "$out" >> "$out.log" 2>&1 \
  && echo "resume OK: $(grep -c 'resumed after epoch 1' "$out/log.txt") resume line, metrics.json $(test -f "$out/metrics.json" && echo yes)" || { echo "resume FAILED"; tail -n 5 "$out.log"; }
for arch in yolo26l rtdetr-l convnext-b yolo26l-p2 yolo26x; do
  for cl in "" "--channels-last"; do
    [ "$arch" = convnext-b ] && [ -n "$cl" ] && continue  # FitDet's fast mode: already channels_last
    o=workspace/mtdet/runs/smoke/speed_${arch}${cl:+_cl}; rm -rf "$o"
    .venv/bin/python workspace/mtdet/train.py --arch $arch --epochs 1 --limit 2000 --evals nd --out "$o" $cl > "$o.log" 2>&1 \
      && echo "$arch ${cl:-default}: $(grep 'iter 100/\|iter 125/' "$o/log.txt" | sed 's/.*mem_gb \([0-9.]*\)).* \([0-9.]*\) images\/s/mem \1 GB, \2 img\/s/' | tr '\n' ' ') | $(tail -n 1 "$o/log.txt")" \
      || { echo "$arch $cl FAILED"; tail -n 5 "$o.log"; }
  done
done
