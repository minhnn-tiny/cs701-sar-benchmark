"""Train one multi-task model (--arch) jointly for classification and detection, and score it on ND (the holdout,
<data>/val), LF (leak-free) and FIT (training images).
- Loss: cross-entropy (x --cls-weight) + the architecture's detection loss. AdamW; trainable backbone parameters at
  --backbone-lr, the rest at --lr; no weight decay on norms and biases. Linear warmup, cosine decay to 1 % of the lr.
  Batch 16, 512 px, bf16 autocast (boxes and box losses in fp32). The model kept and scored is the EMA of the weights
  (--ema 0: the last weights).
- A checkpoint (last.pt) is written after every epoch, and a run directory holding one resumes from it. The data order
  of an epoch depends only on (seed, epoch), so a resumed run sees the same images.
- Output in <out>/: log.txt, metrics.json (args, parameter counts, GPU hours, peak memory, step time, every
  evaluation), model.pt (the scored weights). W&B mirrors the log when WANDB_API_KEY is set (workspace/wandb_log.py).
Example: uv run python workspace/mtdet/train.py --arch yolo26l --epochs 60 --seed 0 --out workspace/mtdet/runs/Y_s0"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import HOLDOUT, ROOT, check_budget, eval_dirs, evaluate, make_loader, split_images

os.environ.pop('TORCH_CUDNN_V8_API_DISABLED', None)  # set by terratorch if anything imported it: slow bf16 convs
import wandb_log  # noqa: E402  (workspace/, on sys.path through common)

# per-architecture defaults for the options left unset: Ultralytics' fine-tuning recipe for YOLO26 (AdamW, weak decay,
# 3 warmup epochs, clip 10), RT-DETR's (backbone at 0.1 x lr, clip 0.1), the R50-FPN-MT protocol for ConvNeXt-B;
# channels_last measured 1.08-1.22x more images/s for the Ultralytics models (ConvNeXt-B's fast mode sets it itself)
DEFAULTS = {
    'yolo26': dict(lr=5e-4, backbone_lr=None, weight_decay=5e-4, warmup_epochs=3, clip=10.0, channels_last=True),
    'rtdetr': dict(lr=1e-4, backbone_lr=1e-5, weight_decay=1e-4, warmup_epochs=3, clip=0.1, channels_last=True),
    'convnext': dict(lr=1e-4, backbone_lr=None, weight_decay=0.05, warmup_epochs=1, clip=None, channels_last=False),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arch', required=True, choices=['yolo26l', 'yolo26l-p2', 'yolo26x', 'rtdetr-l', 'convnext-b'])
    p.add_argument('--data', type=Path, default=HOLDOUT, help='train/ and val/ (the holdout); dataset/repro: all train')
    p.add_argument('--evals', default='nd,lf,fit', help="sets scored during training ('nd' = <data>/val)")
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float)
    p.add_argument('--backbone-lr', type=float)
    p.add_argument('--weight-decay', type=float)
    p.add_argument('--warmup-epochs', type=float)
    p.add_argument('--clip', type=float, help='gradient-norm clip (default per architecture; 0: none)')
    p.add_argument('--cls-weight', type=float, default=1.0, help='weight of the classification cross-entropy')
    p.add_argument('--label-smoothing', type=float, default=0.0, help='label smoothing of that cross-entropy')
    p.add_argument('--cls-head', default='p5', choices=['p5', 'pyramid'],
                   help="the classifier's input: p5, the mean of one layer (ARCHS); pyramid, the mean and max of every "
                        "map the detection head reads (Ultralytics models)")
    p.add_argument('--ema', type=float, default=0.9998, help='EMA decay (0: score the last weights)')
    p.add_argument('--no-flip', action='store_true', help='no random flips')
    p.add_argument('--sar-aug', action='store_true', help='SAR photometric augmentation: speckle and gamma (common.SARAugment)')
    p.add_argument('--sar-p', type=float, default=0.5, help="probability of each of --sar-aug's two parts")
    p.add_argument('--zoom', type=float, default=0.0,
                   help='scale jitter half the time: resized by 1 +- ZOOM, then cropped or padded to 512 (common.Zoom)')
    p.add_argument('--mosaic', action='store_true', help='half the time a 2 x 2 same-class mosaic cut to 512 (common.Mosaic)')
    p.add_argument('--channels-last', action=argparse.BooleanOptionalAction, help='weights and input channels_last')
    p.add_argument('--eval-every', type=int, default=12, help='score every N epochs, and after the last one')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, help='first N images of every set (quick checks)')
    p.add_argument('--max-steps', type=int, help='stop each epoch after N steps (smoke tests)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    for key, value in DEFAULTS['yolo26' if args.arch.startswith('yolo26') else args.arch.split('-')[0]].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.backbone_lr = args.backbone_lr or args.lr
    return args


def build_model(args, coco=True):
    """coco=False: the same modules without loading pretrained weights (a run's model.pt follows)."""
    if args.arch == 'convnext-b':
        assert args.cls_head == 'p5', '--cls-head pyramid: Ultralytics models only'
        from convnext_model import ConvNeXtMultiTask
        return ConvNeXtMultiTask(coco=coco)
    from ultra_model import UltraMultiTask
    return UltraMultiTask(args.arch, coco=coco, epochs=args.epochs, cls_head=args.cls_head)


def log(message, file):
    """Print to the console and to log.txt (W&B mirrors these lines: root train.py's formats)."""
    print(message, flush=True)
    print(message, file=file, flush=True)


def param_groups(model, args):
    """AdamW groups: backbone / rest x decay / no decay (norm weights and biases)."""
    groups = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            key = (model.is_backbone(name), p.ndim > 1)
            groups.setdefault(key, []).append(p)
    return [{'params': ps, 'lr': args.backbone_lr if backbone else args.lr,
             'weight_decay': args.weight_decay if decay else 0.0} for (backbone, decay), ps in sorted(groups.items())]


def lr_factor(step, warmup, total, final=0.01):
    """Linear warmup over `warmup` steps, then cosine decay to `final` x the lr at step `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return final + (1 - final) * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


def source_hash():
    """sha256 of this design's Python files: the code a run used (the git commit may predate it)."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob('*.py')):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


NOT_TRAINING = ('workers', 'eval_every', 'evals')  # args that change what is scored, not what is trained
LATER = {'label_smoothing': 0.0, 'cls_head': 'p5', 'sar_aug': False, 'zoom': 0.0, 'mosaic': False, 'sar_p': 0.5}  # args added after runs had started: left out at their default, so those runs still match


def training_args(args):
    """The args that define a run's training, JSON-plain, for the resume and completed-run checks."""
    return {k: v for k, v in json.loads(json.dumps(vars(args), default=str)).items()
            if k not in NOT_TRAINING and not (k in LATER and v == LATER[k])}


def train_one_epoch(model, ema, loader, optimizer, schedule, epoch, args, log_file):
    model.train()
    start, seen, times = time.time(), 0, []
    steps = min(len(loader), args.max_steps or len(loader))
    trunk = [p for n, p in model.named_parameters() if p.requires_grad and model.is_backbone(n)]
    for step, (images, labels, targets) in enumerate(loader, 1):
        t0 = time.time()
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        targets = [{'boxes': t['boxes'].cuda(non_blocking=True), 'labels': t['labels'].cuda(non_blocking=True)}
                   for t in targets]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _, det_losses = model(images, targets)
        losses = {'classification': F.cross_entropy(logits.float(), labels, label_smoothing=args.label_smoothing),
                  **det_losses}
        loss = args.cls_weight * losses['classification'] + sum(det_losses.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f'epoch {epoch} step {step}: loss {loss.item()}')
        optimizer.zero_grad(set_to_none=True)
        diagnose = step % 50 == 0 or step == steps
        if diagnose:  # the classification part of the trunk's gradient (the rest is detection's), every 50 steps
            cls_grads = torch.autograd.grad(args.cls_weight * losses['classification'], trunk, retain_graph=True,
                                            allow_unused=True)
        loss.backward()
        if diagnose:
            pairs = [(c, p.grad) for c, p in zip(cls_grads, trunk) if p.grad is not None]
            trunk_cls = torch.linalg.vector_norm(torch.stack([c.norm() for c, _ in pairs if c is not None]))
            trunk_det = torch.linalg.vector_norm(torch.stack([(g - (0 if c is None else c)).norm() for c, g in pairs]))
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip or float('inf'))
        optimizer.step()
        schedule.step()
        if ema:
            ema.update(model)
        seen += len(images)
        if diagnose:
            torch.cuda.synchronize()
            times.append(time.time() - t0)  # one synchronised step, data waiting excluded
            parts = ' '.join(f'{name} {value.item():.3f}' for name, value in losses.items())
            log(f'epoch {epoch} iter {step}/{steps} loss {loss.item():.3f} ({parts} gnorm {gnorm.item():.2f} '
                f'trunk_gnorm_cls {trunk_cls.item():.3f} trunk_gnorm_det {trunk_det.item():.3f} mem_gb {torch.cuda.max_memory_allocated() / 2**30:.1f}) lr {optimizer.param_groups[-1]["lr"]:.2e} '
                f'{seen / (time.time() - start):.1f} images/s', log_file)
            start, seen = time.time(), 0
        if step == steps:
            break
    model.epoch_done()
    return times


def evaluate_all(model, dirs, args):
    results = {}
    for name, d in dirs.items():
        loader = make_loader(d.parent, d.name, batch_size=args.batch_size, workers=args.workers, limit=args.limit)
        results[name], _ = evaluate(model, loader, d / 'instances.json')
    return results


def headline(results):
    keys = ('macro_f1', 'mAP', 'AP50', 'delta_m')
    return ' '.join(f'{name}_{k} {m[k]:.4f}' for name, m in results.items() for k in keys)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True  # fixed 512 x 512 input
    args.out.mkdir(parents=True, exist_ok=True)
    resume, done = args.out / 'last.pt', args.out / 'metrics.json'
    if done.exists() and not resume.exists():  # a completed run is never retrained or overwritten
        if training_args(argparse.Namespace(**json.loads(done.read_text())['args'])) == training_args(args):
            print(f'{args.out}: already complete (metrics.json, model.pt); nothing to do')
            return
        raise SystemExit(f'refused: {args.out} holds a completed run with other args; use a new --out')
    log_file = open(args.out / 'log.txt', 'a' if resume.exists() else 'w')
    log(json.dumps({**vars(args), 'git': git_commit(), 'source': source_hash()}, default=str), log_file)
    wandb_log.init(f'mtdet/{args.arch}', args.out, {**vars(args), 'git': git_commit(), 'source': source_hash()},
                   sys.modules[__name__])

    model = build_model(args)
    total, trainable = check_budget(model)
    log(f'parameters: {total:,} total, {trainable:,} trainable', log_file)
    model.cuda()
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
        model.channels_last = True
    dirs = eval_dirs(args.data, args.evals.split(','))
    train_images = split_images(args.data, 'train')
    overlap = {name: len(split_images(d.parent, d.name) & train_images) for name, d in dirs.items()}
    if 'fit' in dirs:
        assert overlap['fit'] == len(split_images(dirs['fit'].parent, 'fit')), 'FIT must be training images'
    held_out = all(overlap.get(name, 0) == 0 for name in ('nd', 'lf'))  # False for all-train runs (dataset/repro)
    log(f'evaluation sets {dict((k, str(v)) for k, v in dirs.items())}: images shared with train {overlap}; '
        f'{"held-out run" if held_out else "NOT a held-out run: its ND/LF scores are in-train"}', log_file)
    loader = make_loader(args.data, 'train', train=True, flips=not args.no_flip, batch_size=args.batch_size,
                         workers=args.workers, limit=args.limit, seed=args.seed, sar_aug=args.sar_aug,
                         zoom=args.zoom, mosaic=args.mosaic, sar_p=args.sar_p)
    steps = min(len(loader), args.max_steps or len(loader))
    optimizer = torch.optim.AdamW(param_groups(model, args), lr=args.lr)
    warmup, total_steps = round(args.warmup_epochs * steps), args.epochs * steps
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, warmup, total_steps))
    ema = None
    if args.ema:
        from ultralytics.utils.torch_utils import ModelEMA
        ema = ModelEMA(model, decay=args.ema)

    first, history, hours, step_times = 1, [], 0.0, []
    if resume.exists():
        state = torch.load(resume, map_location='cuda', weights_only=False)
        if state['args'] != training_args(args):
            raise SystemExit(f"refused to resume {resume}: it was trained with other args "
                             f"{ {k: v for k, v in state['args'].items() if training_args(args).get(k) != v} }")
        if state['source'] != source_hash():
            log(f"note: the code changed since the checkpoint (source {state['source']} -> {source_hash()})", log_file)
        torch.set_rng_state(state['rng_cpu'].cpu())  # map_location moved them to the GPU; RNG states are CPU bytes
        torch.cuda.set_rng_state(state['rng_cuda'].cpu())
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        schedule.load_state_dict(state['schedule'])
        if ema:
            ema.ema.load_state_dict(state['ema'])
            ema.updates = state['ema_updates']
        first, history, hours, step_times = state['epoch'] + 1, state['history'], state['hours'], state['step_times']
        model.replay_epochs(state['epoch'])
        log(f'resumed after epoch {state["epoch"]}', log_file)

    for epoch in range(first, args.epochs + 1):
        start = time.time()
        loader.sampler.epoch = epoch
        step_times += train_one_epoch(model, ema, loader, optimizer, schedule, epoch, args, log_file)[1:]
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            results = evaluate_all(ema.ema if ema else model, dirs, args)
            history.append({'epoch': epoch, **results})
            log(f'epoch {epoch} val: {headline(results)}', log_file)
        hours += (time.time() - start) / 3600
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'schedule': schedule.state_dict(),
                    'ema': ema.ema.state_dict() if ema else None, 'ema_updates': ema.updates if ema else 0,
                    'epoch': epoch, 'history': history, 'hours': hours, 'step_times': step_times,
                    'args': training_args(args), 'source': source_hash(), 'rng_cpu': torch.get_rng_state(),
                    'rng_cuda': torch.cuda.get_rng_state()}, f'{resume}.tmp')
        os.replace(f'{resume}.tmp', resume)  # atomic: a crash mid-save keeps the previous checkpoint

    final = history[-1]
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    step_ms = 1000 * sum(step_times) / max(len(step_times), 1)
    log(f'training {hours:.2f} h, peak GPU memory {peak_gb:.1f} GB, {step_ms:.0f} ms per step', log_file)
    summary = {'args': vars(args), 'git': git_commit(), 'source': source_hash(), 'parameters': total,
               'trainable_parameters': trainable, 'train_hours': hours, 'peak_gpu_memory_gb': peak_gb,
               'step_ms': step_ms, 'eval_overlap': overlap, 'held_out': held_out, 'history': history, 'final': {k: v for k, v in final.items() if k != 'epoch'},
               'val': final.get('nd', next(v for k, v in final.items() if k != 'epoch'))}
    (args.out / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str))
    torch.save((ema.ema if ema else model).state_dict(), args.out / 'model.pt')
    resume.unlink()  # the run is complete: model.pt and metrics.json are its record
    wandb_log.finish(summary)


if __name__ == '__main__':
    main()
