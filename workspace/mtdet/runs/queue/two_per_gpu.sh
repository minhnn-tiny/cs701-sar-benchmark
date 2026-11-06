#!/usr/bin/env bash
# 26 Oct: once the single-worker loop has started YA150_val (so YA150 is done and YA150_r2 claimed), replace it with
# two loops on the same job file: 2 jobs on GPU 1, and 1 job of at most 12 GB on GPU 2 next to FitDet's queue.
cd "$(dirname "$0")/../../../.." || exit 1
until grep -q "start YA150_val " workspace/mtdet/runs/queue/run.log; do sleep 10; done
pkill -TERM -f '^bash workspace/mtdet/run.sh workspace/mtdet/jobs/plan.txt$'
sleep 2
GPUS=1 WORKERS=2 MIN_FREE=10000 nohup setsid workspace/mtdet/run.sh workspace/mtdet/jobs/plan.txt \
    >> workspace/mtdet/runs/queue/run.log 2>&1 < /dev/null &
sleep 10
GPUS=2 WORKERS=1 MIN_FREE=10000 MAX_NEED=12000 nohup setsid workspace/mtdet/run.sh workspace/mtdet/jobs/plan.txt \
    >> workspace/mtdet/runs/queue/run.log 2>&1 < /dev/null &
