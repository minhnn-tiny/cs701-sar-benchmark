#!/usr/bin/env bash
# Reproduce the Δm reference (ViT + LoRA) on all of train, as the TA ran it: one seed per free GPU.
# The GPU server is shared by the team, so a seed starts only on a GPU that is healthy (gpu_ok), idle (< 20 %
# utilisation), has >= 25 GB free and runs none of our jobs. All GPUs are candidates: a faulty one (GPU 0 since
# 23 Oct 2026) fails gpu_ok and is skipped until it is repaired.
# From the repo root, after `uv run python workspace/baseline/make_repro_data.py`:
#   mkdir -p workspace/baseline/runs/repro
#   nohup setsid workspace/baseline/run_repro.sh > workspace/baseline/runs/repro/queue.log 2>&1 &
# Override with e.g. GPUS="1" SEEDS="2".
cd "$(dirname "$0")/../.." || exit 1
OUT=workspace/baseline/runs/repro
LOCKS=workspace/runs/gpu_locks  # shared by every queue in workspace/: gpu<N>/ exists while one of our runs uses GPU N
read -ra GPUS <<<"${GPUS:-0 1 2}"
read -ra SEEDS <<<"${SEEDS:-0 1 2}"
mkdir -p "$OUT" "$LOCKS"

# Healthy = no failed or pending row remap, and no uncorrectable DRAM error since the driver loaded.
gpu_ok() {
  local q; q=$(nvidia-smi -i "$1" -q -d ROW_REMAPPER,ECC) || return 1
  grep -qE 'Remapping Failure Occurred *: Yes|^ *Pending *: Yes' <<<"$q" && return 1
  [ "$(awk '/Volatile/{v=1} /Aggregate/{v=0} v && /DRAM Uncorrectable/{print $NF}' <<<"$q")" = 0 ]
}

for g in "${GPUS[@]}"; do gpu_ok "$g" || echo "$(date '+%F %T') GPU $g unhealthy (ECC / row remap), skipped while it stays so"; done
for s in "${SEEDS[@]}"; do
  while :; do
    for g in "${GPUS[@]}"; do
      gpu_ok "$g" || continue
      read -r free util <<<"$(nvidia-smi -i "$g" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ',')"
      if [ "$free" -ge 25000 ] && [ "$util" -lt 20 ] && mkdir "$LOCKS/gpu$g" 2>/dev/null; then
        echo "$(date '+%F %T') seed $s -> GPU $g (free $free MiB, util $util %)"
        ( CUDA_VISIBLE_DEVICES=$g .venv/bin/python train.py --backbone vit --init pretrained --adapt lora \
              --data dataset/repro --seed "$s" --out "$OUT/seed$s" > "$OUT/seed$s.out" 2>&1
          rc=$?
          echo "$(date '+%F %T') seed $s finished on GPU $g (exit $rc)"
          rmdir "$LOCKS/gpu$g" ) &
        sleep 120  # let the new run allocate its memory before looking again
        continue 3
      fi
    done
    sleep 60
  done
done
wait
echo "$(date '+%F %T') all seeds done"
