# Server reboot: pause and resume (MTDet, 26 Oct 2026)

## State from ~15:30 SGT

- Since 26 Oct the user's rule is up to 2 of our runs per GPU (`EXPERIMENTS.md` M0.6). Two queue loops share
  `jobs/plan.txt` and its claims in `runs/queue/`:
  - loop A, GPU 1: 2 jobs at once;
  - loop B, GPU 2: 1 job that needs at most 12 GB, next to FitDet's queue, which its own session pauses and resumes.
- `runs/queue/two_per_gpu.sh` started both loops once the old single loop had started `YA150_val`.
- Every job writes `runs/<name>/last.pt` after each epoch. **GPU 0** has a memory fault, and `gpu_run.sh` skips it.
- **Leaderboard**: see `submissions/SUBMISSIONS.md`.

## What a reboot loses

- The epoch in progress of each running job: at most ~2 min for YOLO26-L while it shares a GPU.
- `/tmp`, including Claude's scratchpad: the system clears it at boot. The upload helper and the run waiter are in
  `workspace/mtdet/runs/queue/` (`lb_upload.sh`, `wait_run.sh`).
- Claude's background waiters. Nothing in the repo is lost.

## 1. Pause, before the reboot

Tell Claude "pause for the reboot", or run these from the repo root:

```bash
# 1a. Stop both queue loops so they claim no new job; the jobs already running carry on
pkill -TERM -f '^bash workspace/mtdet/run.sh workspace/mtdet/jobs/plan.txt$'
# 1b. Stop the running jobs by sending TERM to their python directly. In an `a && b` job, a TERM sent to gpu_run.sh
#     stops bash but leaves python running on the GPU. Once python dies, each job records exit 143 and gpu_run.sh
#     releases its slot lock. The pattern matches only the queue's jobs and their data-loader workers.
pkill -TERM -f '^\.venv/bin/python workspace/mtdet/'
# check, ~10 s later: nothing of ours left on GPUs 1 and 2, and the jobs show exit 143
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
workspace/mtdet/run.sh --status workspace/mtdet/jobs/plan.txt | grep -v pending
```

## 2. Resume, after the reboot

From the repo root:

```bash
# 2a. GPU health, memory and slot locks; a lock whose pid is gone is cleared automatically
workspace/gpu_run.sh --status
# 2b. Find the jobs that were cut off: "running ... since" (killed by the reboot) or "exit 143" (paused)
workspace/mtdet/run.sh --status workspace/mtdet/jobs/plan.txt | grep -v pending
```

**2c. Requeue every job that was cut off.**
- Add a line right after the job's own line, with the same command (keep its `MIN_FREE=` prefix) under a new name:
  `<name>_r2 <command>`. A job name runs only once.
- `train.py` resumes from `runs/<name>/last.pt`, because the args match. If training had already finished, it goes
  straight to the job's `evaluate.py` or `predict.py` part.

```bash
# 2d. Restart both loops
GPUS=1 WORKERS=2 MIN_FREE=10000 nohup setsid workspace/mtdet/run.sh workspace/mtdet/jobs/plan.txt >> workspace/mtdet/runs/queue/run.log 2>&1 &
GPUS=2 WORKERS=1 MIN_FREE=10000 MAX_NEED=12000 nohup setsid workspace/mtdet/run.sh workspace/mtdet/jobs/plan.txt >> workspace/mtdet/runs/queue/run.log 2>&1 &
# after ~3 min (a GPU is skipped for 2 min after one of our jobs starts on it): expect "start <name>_r2 on GPU ..."
tail -n 5 workspace/mtdet/runs/queue/run.log
```

- If the reboot comes before `YA150_val` has started (before ~15:30), YA150 may not have finished: start
  `runs/queue/two_per_gpu.sh` in the background and then the single loop
  (`GPUS=1 WORKERS=1 nohup setsid workspace/mtdet/run.sh ...` as above) instead of the two loops. `YA150_r2` then resumes
  YA150 before anything else claims a job.

**2e. Restart Claude** in the repo with `claude --resume`, pick this conversation, and say "resume after the reboot
(workspace/mtdet/REBOOT.md)". Claude then re-arms its waiters and handles any pending upload.

## Notes

- A resumed run opens a new W&B run, because `workspace/wandb_log.py` starts one per call.
- The ETA moves by the downtime.
- If `gpu_run.sh --status` reports GPU 1 or 2 unhealthy after the reboot, the queue waits instead of running. Tell Claude.
