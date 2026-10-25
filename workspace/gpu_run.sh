#!/usr/bin/env bash
# Run one command on one GPU of the shared server, waiting until a GPU is healthy, free and not used by our runs.
# Every GPU command in workspace/ (smoke tests, training, predict) goes through it, so the checks are always the same.
#   workspace/gpu_run.sh [options] -- <command> [args...]
#   workspace/gpu_run.sh --status                       # print every GPU's state and exit
# Options:
#   --gpus "0 1 2"     candidate GPUs, tried in order (default: all three)
#   --min-free MiB     free memory needed (default 25000 for training; ~8000 is enough for predict)
#   --max-util %       GPU utilisation must be below this (default 20: nobody else is computing on it)
#   --poll s           seconds between checks while waiting (default 60)
#   --max-wait s       give up after this long with exit code 75 (default 0: wait forever)
# A GPU is taken only if it passes gpu_ok (no failed or pending row remap, no uncorrectable DRAM error since the
# driver loaded), has the free memory, is idle, and its lock workspace/runs/gpu_locks/gpu<N>/ can be created. The
# lock is shared with workspace/baseline/run_repro.sh and removed when the command ends, also on Ctrl-C or kill.
# The command runs with CUDA_VISIBLE_DEVICES=<N>; the exit code is the command's.
set -u
LOCKS="$(cd "$(dirname "$0")" && pwd)/runs/gpu_locks"
GPUS="0 1 2" MIN_FREE=25000 MAX_UTIL=20 POLL=60 MAX_WAIT=0 STATUS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) GPUS=$2; shift 2 ;;
    --min-free) MIN_FREE=$2; shift 2 ;;
    --max-util) MAX_UTIL=$2; shift 2 ;;
    --poll) POLL=$2; shift 2 ;;
    --max-wait) MAX_WAIT=$2; shift 2 ;;
    --status) STATUS=1; shift ;;
    --) shift; break ;;
    *) echo "gpu_run: unknown option $1 (the command goes after --)" >&2; exit 2 ;;
  esac
done
read -ra GPUS <<<"$GPUS"
mkdir -p "$LOCKS"
log() { echo "$(date '+%F %T') gpu_run: $*" >&2; }

gpu_ok() {
  local q; q=$(nvidia-smi -i "$1" -q -d ROW_REMAPPER,ECC) || return 1
  grep -qE 'Remapping Failure Occurred *: Yes|^ *Pending *: Yes' <<<"$q" && return 1
  [ "$(awk '/Volatile/{v=1} /Aggregate/{v=0} v && /DRAM Uncorrectable/{print $NF}' <<<"$q")" = 0 ]
}

# A lock whose owner (pid file written by this script) is gone is left over from a killed run: remove it.
clear_stale() {
  local pid; pid=$(cat "$LOCKS/gpu$1/pid" 2>/dev/null) || return 0
  kill -0 "$pid" 2>/dev/null || { log "removing stale lock of GPU $1 (pid $pid is gone)"; rm -rf "$LOCKS/gpu$1"; }
}

if [ "$STATUS" = 1 ]; then
  for g in "${GPUS[@]}"; do
    clear_stale "$g"
    read -r free util <<<"$(nvidia-smi -i "$g" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ',')"
    gpu_ok "$g" && health=healthy || health=UNHEALTHY
    [ -d "$LOCKS/gpu$g" ] && lock="locked by our run" || lock=unlocked
    echo "GPU $g: $health, free $free MiB, util $util %, $lock"
  done
  exit 0
fi
[ $# -gt 0 ] || { echo "gpu_run: no command given (put it after --)" >&2; exit 2; }

start=$SECONDS waited=0
while :; do
  for g in "${GPUS[@]}"; do
    gpu_ok "$g" || continue
    read -r free util <<<"$(nvidia-smi -i "$g" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ',')"
    [ "$free" -ge "$MIN_FREE" ] && [ "$util" -lt "$MAX_UTIL" ] || continue
    clear_stale "$g"
    mkdir "$LOCKS/gpu$g" 2>/dev/null || continue
    echo $$ > "$LOCKS/gpu$g/pid"
    lock="$LOCKS/gpu$g"
    break 2
  done
  [ "$waited" = 1 ] || { log "no GPU in (${GPUS[*]}) is healthy, has $MIN_FREE MiB free and is idle; waiting"; waited=1; }
  if [ "$MAX_WAIT" -gt 0 ] && [ $((SECONDS - start)) -ge "$MAX_WAIT" ]; then
    log "gave up after $MAX_WAIT s"; exit 75
  fi
  sleep "$POLL"
done

log "GPU $g (free $free MiB, util $util %) -> $*"
trap 'rm -rf "$lock"' EXIT
CUDA_VISIBLE_DEVICES=$g "$@" &
child=$!
trap 'kill -TERM "$child" 2>/dev/null; wait "$child"; exit 143' INT TERM
wait "$child"
rc=$?
log "GPU $g released (exit $rc)"
exit $rc
