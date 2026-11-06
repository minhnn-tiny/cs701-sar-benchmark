#!/usr/bin/env bash
# Run one command on one GPU of the shared server, waiting until a GPU is healthy, has the memory and has a free slot for
# our commands.
# Every GPU command in workspace/ (smoke tests, training, predict) goes through it, so the checks are always the same.
#   workspace/gpu_run.sh [options] -- <command> [args...]
#   workspace/gpu_run.sh --status                       # print every GPU's state and exit
# Options:
#   --gpus "0 1 2"     candidate GPUs, tried in order (default: all three)
#   --min-free MiB     free memory needed (default 25000 for training; ~8000 is enough for predict)
#   --margin MiB       extra free memory needed on a GPU where other users have processes, so that theirs keep room
#                      to grow (default 4000)
#   --max-util %       other users' processes must use less of the GPU than this (default 20: nobody else is
#                      computing on it; 100 = no limit, sharing the GPU with their jobs); our own commands don't count
#   --slots N          our commands one GPU may run at once (default 4, the user's rule since 27 Oct 2026; free memory
#                      usually binds first. A second run added 20-27 % throughput: workspace/mtdet/EXPERIMENTS.md M0.6)
#   --poll s           seconds between checks while waiting (default 60)
#   --max-wait s       give up after this long with exit code 75 (default 0: wait forever)
# A GPU is taken only if it passes gpu_ok (no pending row remap, no uncorrectable DRAM error since the driver loaded),
# has the free memory (plus the margin if other users have processes on it), other users' processes on it stay under
# --max-util, none of our commands started on it in the last 2 minutes (so its free memory already counts them), and
# one of its slot locks can be created, the highest first, so that callers with fewer slots still find slot 1:
# workspace/runs/gpu_locks/gpu<N>/ (slot 1, shared with workspace/baseline/run_repro.sh), gpu<N>.2/, ...
# gpu_ok ignores "Remapping Failure Occurred": GPU 0 keeps that flag from its 23 Oct 2026 fault, which the user reports
# the 27 Oct reboot fixed. A new uncorrectable error still takes a GPU out until the next reboot.
# The lock is removed when the command ends, also on Ctrl-C or kill.
# The command runs with CUDA_VISIBLE_DEVICES=<N>; the exit code is the command's.
set -u
LOCKS="$(cd "$(dirname "$0")" && pwd)/runs/gpu_locks"
GPUS="0 1 2" MIN_FREE=25000 MARGIN=4000 MAX_UTIL=20 SLOTS=4 POLL=60 MAX_WAIT=0 STATUS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) GPUS=$2; shift 2 ;;
    --min-free) MIN_FREE=$2; shift 2 ;;
    --margin) MARGIN=$2; shift 2 ;;
    --max-util) MAX_UTIL=$2; shift 2 ;;
    --slots) SLOTS=$2; shift 2 ;;
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
  grep -qE '^ *Pending *: Yes' <<<"$q" && return 1
  [ "$(awk '/Volatile/{v=1} /Aggregate/{v=0} v && /DRAM Uncorrectable/{print $NF}' <<<"$q")" = 0 ]
}

# Slot $2 of GPU $1 (slot 1 keeps the old lock name)
slot() { if [ "$2" = 1 ]; then echo "$LOCKS/gpu$1"; else echo "$LOCKS/gpu$1.$2"; fi; }

# Our locks on GPU $1. A lock whose owner (pid file written by this script) is gone is left over from a killed run:
# remove it.
locks_of() {
  local l pid
  for l in "$LOCKS/gpu$1" "$LOCKS/gpu$1".*; do
    [ -d "$l" ] || continue
    if pid=$(cat "$l/pid" 2>/dev/null) && ! kill -0 "$pid" 2>/dev/null; then
      log "removing stale lock $l (pid $pid is gone)"; rm -rf "$l"; continue
    fi
    echo "$l"
  done
}

# Other users' compute processes on GPU $1 (a pid that ps can't see counts as another user's)
others_on() {
  local me; me=$(id -u)
  nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader |
    while read -r pid; do [ "$(ps -o uid= -p "$pid" | tr -d ' ')" = "$me" ] || echo "$pid"; done | wc -l
}

# SM % that other users' processes use on GPU $1
others_util() {
  local me; me=$(id -u)
  nvidia-smi pmon -i "$1" -c 1 -s u | awk '!/^#/ && $4 ~ /^[0-9]+$/ {print $2, $4}' |
    while read -r pid sm; do [ "$(ps -o uid= -p "$pid" | tr -d ' ')" = "$me" ] || echo "$sm"; done |
    awk '{s += $1} END {print s + 0}'
}

if [ "$STATUS" = 1 ]; then
  for g in "${GPUS[@]}"; do
    n=$(locks_of "$g" | wc -l)
    read -r free util <<<"$(nvidia-smi -i "$g" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ',')"
    gpu_ok "$g" && health=healthy || health=UNHEALTHY
    echo "GPU $g: $health, free $free MiB, util $util %, $n of our runs, $(others_on "$g") processes of other users"
  done
  exit 0
fi
[ $# -gt 0 ] || { echo "gpu_run: no command given (put it after --)" >&2; exit 2; }

start=$SECONDS waited=0
while :; do
  for g in "${GPUS[@]}"; do
    gpu_ok "$g" || continue
    ours=$(locks_of "$g")
    read -r free util <<<"$(nvidia-smi -i "$g" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ',')"
    need=$MIN_FREE; [ "$(others_on "$g")" = 0 ] || need=$((MIN_FREE + MARGIN))
    [ "$free" -ge "$need" ] || continue
    [ "$MAX_UTIL" -ge 100 ] || [ "$(others_util "$g")" -lt "$MAX_UTIL" ] || continue
    for l in $ours; do  # a command of ours that started in the last 2 min may not hold its memory yet
      [ $(( $(date +%s) - $(stat -c %Y "$l" 2>/dev/null || echo 0) )) -ge 120 ] || continue 2
    done
    for k in $(seq "$SLOTS" -1 1); do
      mkdir "$(slot "$g" "$k")" 2>/dev/null || continue
      lock=$(slot "$g" "$k")
      echo $$ > "$lock/pid"
      break 3
    done
  done
  [ "$waited" = 1 ] || { log "no GPU in (${GPUS[*]}) is healthy, has $MIN_FREE MiB free ($MARGIN more beside other users' processes) and one of $SLOTS slots free; waiting"; waited=1; }
  if [ "$MAX_WAIT" -gt 0 ] && [ $((SECONDS - start)) -ge "$MAX_WAIT" ]; then
    log "gave up after $MAX_WAIT s"; exit 75
  fi
  sleep "$POLL"
done

log "GPU $g (free $free MiB, util $util %, slot $k) -> $*"
trap 'rm -rf "$lock"' EXIT
CUDA_VISIBLE_DEVICES=$g "$@" &
child=$!
trap 'kill -TERM "$child" 2>/dev/null; wait "$child"; exit 143' INT TERM
wait "$child"
rc=$?
log "GPU $g released (exit $rc)"
exit $rc
