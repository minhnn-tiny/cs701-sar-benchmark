#!/usr/bin/env bash
# Run the commands of a job file, in order, each through workspace/gpu_run.sh (the team's shared GPU runner: GPU
# health check, >= 25 GB free, idle, and the shared locks in workspace/runs/gpu_locks/). One command per line ('#'
# lines and blank lines are skipped). The next job is dispatched once the previous one holds a GPU, so jobs start in
# file order and run in parallel on GPUs 1 and 2. From the repo root:
#   nohup setsid workspace/detail_lora/queue.sh workspace/detail_lora/jobs/holdout.txt \
#       >> workspace/detail_lora/runs/queue.log 2>&1 &
# Override the GPUs with GPUS="2" (default "1 2"; GPU 0 is unhealthy). Stop dispatching with
# pkill -f detail_lora/queue.sh; started jobs go on (stop one with: kill <pid of its gpu_run.sh>).
cd "$(dirname "$0")/../.." || exit 1
JOBS=$1
[ -f "$JOBS" ] || { echo "usage: $0 <job file>"; exit 1; }
GPUS=${GPUS:-1 2}
LOCKS=workspace/runs/gpu_locks
mapfile -t JOBLIST < <(grep -v -e '^\s*#' -e '^\s*$' "$JOBS")  # an array, so `wait` below sees every job
for job in "${JOBLIST[@]}"; do
  echo "$(date '+%F %T') queued: $job"
  workspace/gpu_run.sh --gpus "$GPUS" -- bash -c "$job" </dev/null &  # gpu_run logs the GPU and the exit code
  pid=$!
  until grep -qsx "$pid" "$LOCKS"/gpu*/pid || ! kill -0 "$pid" 2>/dev/null; do sleep 10; done
  sleep 30  # let the job allocate its memory before the next one looks
done
wait
echo "$(date '+%F %T') all jobs of $JOBS done"
