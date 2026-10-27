# Server reboot: pause and resume (MTDet)

## Last reboot: 27 Oct 2026

- 09:41: both queue loops stopped; the 3 running jobs were to stop right after their next checkpoint (1b below).
  YSA250_s1 did (epoch 130); the reboot came at 09:43, before the others' checkpoints, and cut C_s0 (epoch 34 of 72,
  300 of 453 steps lost) and YASA250 (epoch 139 of 250, 550 of 587 lost): under 0.1 GPU h in all.
- 11:21: resumed as `C_s0_r2`, `YASA250_r2` and `YSA250_s1_r2` in `jobs/plan.txt` (`YASA250_r2` also writes the
  no-floor val zips; the job that waited for them, `YASA250_t0`, was dropped). After the reboot tom ran jobs on GPUs 0
  and 1, so only GPU 2 was free: loop B took `YASA250_r2`, a one-off queue ran the short `YT0` in GPU 2's other slot
  (FitDet's, whose queue was down), and loop A waits for GPU 1.
- Leaderboard: see `submissions/SUBMISSIONS.md`.

## How the queue runs

- Up to 2 of our runs per GPU (`EXPERIMENTS.md` M0.6). Two queue loops share `jobs/plan.txt` and its claims in
  `runs/queue/`: loop A, GPU 1, 2 jobs at once; loop B, GPU 2, 1 job that needs at most 12 GB, next to FitDet's queue,
  which its own session pauses and resumes.
- Since 27 Oct 13:29, loop C (GPU 2, 1 job of any size) also runs, in the slot kept for FitDet while its queue is down.
- Every job writes `runs/<name>/last.pt` after each epoch. **GPU 0** has a memory fault, and `gpu_run.sh` skips it.

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
# 1b. Stop each running job right after its next checkpoint (YOLO26-L: within ~1.5 min; ConvNeXt-B: ~8 min). TERM goes
#     to python itself: in an `a && b` job, a TERM sent to gpu_run.sh stops bash but leaves python on the GPU. The job
#     then records exit 143 and gpu_run.sh releases its slot. A job in its evaluate.py / predict.py part (minutes) isn't
#     covered: let it finish. Reboot only once the check below shows nothing of ours.
for pid in $(pgrep -f '^\.venv/bin/python workspace/mtdet/train\.py '); do
  [ "$(ps -o comm= -p "$(ps -o ppid= -p "$pid" | tr -d ' ')")" = bash ] || continue  # skip data-loader workers (children of the trainer)
  ck=$(tr '\0' ' ' < /proc/$pid/cmdline | grep -o -- '--out [^ ]*' | cut -d' ' -f2)/last.pt
  ( t=$(stat -c %Y "$ck"); while [ "$(stat -c %Y "$ck")" = "$t" ]; do sleep 2; done; kill -TERM "$pid" $(pgrep -P "$pid") ) &
done; wait
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
  `<name>_r2 <command>`. A job name runs only once. Fold a pending job that waits for a cut-off run (like
  `YASA250_t0`) into its `_r2` line: its wait would time out.
- `train.py` resumes from `runs/<name>/last.pt`, because the args match. If training had already finished, it goes
  straight to the job's `evaluate.py` or `predict.py` part.

```bash
# 2d. Restart both loops
GPUS=1 WORKERS=2 MIN_FREE=10000 nohup setsid workspace/mtdet/run.sh workspace/mtdet/jobs/plan.txt >> workspace/mtdet/runs/queue/run.log 2>&1 &
GPUS=2 WORKERS=1 MIN_FREE=10000 MAX_NEED=12000 nohup setsid workspace/mtdet/run.sh workspace/mtdet/jobs/plan.txt >> workspace/mtdet/runs/queue/run.log 2>&1 &
# loop C, only while FitDet's queue is down: GPU 2's other slot, any job size
GPUS=2 WORKERS=1 MIN_FREE=10000 nohup setsid workspace/mtdet/run.sh workspace/mtdet/jobs/plan.txt >> workspace/mtdet/runs/queue/run.log 2>&1 &
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
