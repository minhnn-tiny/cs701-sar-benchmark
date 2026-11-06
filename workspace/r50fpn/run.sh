#!/usr/bin/env bash
# The experiment queue of workspace/r50fpn, a copy of workspace/sarmtl/run.sh with its own queue directory: runs the
# jobs of a job file in priority order, one job per GPU, each through workspace/gpu_run.sh (healthy, free, idle,
# shared locks). A worker claims a job only once gpu_run.sh holds
# a GPU for it, so the highest-priority job left always gets the next free GPU.
#   nohup setsid workspace/r50fpn/run.sh workspace/r50fpn/jobs/<file>.txt >> workspace/r50fpn/runs/queue/run.log 2>&1 &
#   workspace/r50fpn/run.sh --status workspace/r50fpn/jobs/<file>.txt
# Job file: one job per line, "<name> <command...>", in priority order; '#' lines and blank lines are skipped.
# A job's output goes to runs/queue/<name>.out, its state to runs/queue/<name>/ (cmd, gpu, start, end, rc).
# A job name runs at most once: to rerun a job (e.g. after a GPU fault), add a line with a new name (ref_s0_r2).
# After a crash or a lost session, launch the same command again: claimed jobs are skipped. A job whose worker was
# killed has no rc; check its .out and requeue it under a new name.
# Env: WORKERS (default 3, one per candidate GPU), GPUS (default "0 1 2"; gpu_run.sh skips unhealthy ones),
# MIN_FREE (MiB, default 25000), MAX_WAIT (s, default 0 = wait forever; a worker that gets no GPU in that time stops).
# A worker also stops after 2 failed jobs in a row.
set -u
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/../.." || exit 1
Q=workspace/r50fpn/runs/queue
WORKERS=${WORKERS:-3} GPUS=${GPUS:-0 1 2} MIN_FREE=${MIN_FREE:-25000} MAX_WAIT=${MAX_WAIT:-0}
mkdir -p "$Q"
log() { echo "$(date '+%F %T') run.sh: $*"; }
jobs_of() { grep -v -e '^\s*#' -e '^\s*$' "$1"; }

unclaimed_left() {
  local name cmd
  while read -r name cmd; do [ -d "$Q/$name" ] || return 0; done < <(jobs_of "$JOBS")
  return 1
}

# Called by gpu_run.sh (CUDA_VISIBLE_DEVICES set): claim the first unclaimed job, run it, return its exit code
# (99: no job left). mkdir is atomic, so each job is claimed by exactly one worker.
next_job() {
  local name cmd child rc
  while read -r name cmd; do
    mkdir "$Q/$name" 2>/dev/null || continue
    echo "$cmd" > "$Q/$name/cmd"
    echo "$CUDA_VISIBLE_DEVICES" > "$Q/$name/gpu"
    date +%s > "$Q/$name/start"
    log "start $name on GPU $CUDA_VISIBLE_DEVICES: $cmd"
    bash -c "$cmd" > "$Q/$name.out" 2>&1 < /dev/null &
    child=$!
    trap 'kill -TERM "$child" 2>/dev/null' INT TERM  # gpu_run.sh forwards a kill to us; pass it on to the job
    wait "$child"; rc=$?
    while kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
    date +%s > "$Q/$name/end"
    echo "$rc" > "$Q/$name/rc"
    log "end $name on GPU $CUDA_VISIBLE_DEVICES: exit $rc after $(( ($(cat "$Q/$name/end") - $(cat "$Q/$name/start")) / 60 )) min"
    return "$rc"
  done < <(jobs_of "$JOBS")
  return 99
}

status() {
  local name cmd state
  while read -r name cmd; do
    if [ ! -d "$Q/$name" ]; then state=pending
    elif [ ! -f "$Q/$name/rc" ]; then state="running on GPU $(cat "$Q/$name/gpu") since $(date -d @"$(cat "$Q/$name/start")" '+%F %T')"
    else state="exit $(cat "$Q/$name/rc") on GPU $(cat "$Q/$name/gpu"), $(( ($(cat "$Q/$name/end") - $(cat "$Q/$name/start")) / 60 )) min"
    fi
    printf '%-16s %s\n' "$name" "$state"
  done < <(jobs_of "$JOBS")
}

worker() {
  local w=$1 fails=0 rc
  while unclaimed_left; do
    "$(dirname "$SELF")/../gpu_run.sh" --gpus "$GPUS" --min-free "$MIN_FREE" --max-wait "$MAX_WAIT" \
        -- "$SELF" --next "$JOBS"
    rc=$?
    [ "$rc" = 99 ] && break
    if [ "$rc" = 75 ]; then log "worker $w: no GPU within $MAX_WAIT s, stopping"; break; fi
    if [ "$rc" = 0 ]; then fails=0; else fails=$((fails + 1)); fi
    if [ "$fails" -ge 2 ]; then
      log "worker $w: 2 failed jobs in a row, stopping (check gpu_run.sh --status and the jobs' .out files)"
      break
    fi
  done
}

case "${1:-}" in
  --next) JOBS=$2; next_job; exit $? ;;
  --status) JOBS=$2; status; exit 0 ;;
  ''|-*) echo "usage: $0 <job file> | --status <job file>" >&2; exit 2 ;;
esac
JOBS=$(realpath "$1")
[ -f "$JOBS" ] || { echo "run.sh: no job file $1" >&2; exit 2; }
dups=$(jobs_of "$JOBS" | awk '{print $1}' | sort | uniq -d)
[ -z "$dups" ] || { echo "run.sh: duplicate job names: $dups" >&2; exit 2; }
log "queue $JOBS: $(jobs_of "$JOBS" | wc -l) jobs; $WORKERS workers on GPUs ($GPUS), $MIN_FREE MiB free needed"
for w in $(seq 1 "$WORKERS"); do
  worker "$w" &
  sleep 5  # stagger the workers' first GPU checks
done
wait
log "queue $JOBS: no unclaimed job left"
