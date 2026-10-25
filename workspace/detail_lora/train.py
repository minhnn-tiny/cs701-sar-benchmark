"""Train DetailLoRA (docs/DESIGN.md) jointly for classification and detection, and score it on the holdout.
Example, from the repo root:
  CUDA_VISIBLE_DEVICES=1 uv run python workspace/detail_lora/train.py --seed 0 --out workspace/detail_lora/runs/seed0
Same protocol as the reference train.py (AdamW, batch 16, 24 epochs, lr 1e-4, one warmup epoch + cosine, bf16,
validation every 4 epochs, last-epoch model), except: the unfrozen pretrained tensors train at --pretrained-lr, and
the training images get the SAR augmentations of detail_data.py. Writes <out>/<config>/: log.txt, metrics.json,
predictions_val.json, model.pt; checkpoint.pt after every epoch while training (--resume continues from it)."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import ROOT, FUSIONS, evaluate_fusions, load_module, model_from_args, reference_train, time_inference
from detail_data import AugmentedSARMultiTask
from detail_model import check_budget, count_parameters, is_adapter
from sarbench.data import SARMultiTask, collate

wandb_log = load_module('wandb_log', ROOT / 'workspace' / 'wandb_log.py')
headline, lr_factor = reference_train.headline, reference_train.lr_factor
log = lambda message, file: reference_train.log(message, file)  # noqa: E731 (looked up per call: wandb_log wraps it)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=ROOT / 'dataset' / 'holdout',
                   help='a folder with train/ and val/ (dataset/holdout; dataset/repro to train on all of train)')
    # The design; the defaults are the primary model, A1 = --no-detail --reference-anchors
    p.add_argument('--backbone', default='vit', choices=['vit', 'terramind'])
    p.add_argument('--adapt', default='dense', choices=['dense', 'moelora'],
                   help="dense: DetailLoRA's dense LoRA (ViT only); moelora: the reference's MoE-LoRA, unchanged")
    p.add_argument('--rank', type=int, default=32, help='LoRA rank (alpha = 2 x rank); --adapt dense only')
    p.add_argument('--attn-only', action='store_true', help='LoRA on qkv and proj only, as the reference')
    p.add_argument('--freeze-pretrained', action='store_true', help='keep patch/position embedding and norms frozen')
    p.add_argument('--no-detail', action='store_true', help='no detail branch')
    p.add_argument('--reference-anchors', action='store_true', help="the reference's anchors (8-181 px, 6 per location)")
    p.add_argument('--no-augment', action='store_true', help="flips only, as the reference")
    p.add_argument('--fusion', default='soft', choices=FUSIONS, help='detection scoring of the reported metrics')
    # The protocol (the reference's defaults)
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4, help='LoRA, detail branch and heads')
    p.add_argument('--pretrained-lr', type=float, default=2e-5, help='unfrozen pretrained backbone tensors')
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--eval-every', type=int, default=4, help='validate every N epochs, and after the last one')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, help='use only the first N images of every split (quick checks)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'runs')
    p.add_argument('--resume', action='store_true', help='continue from <run>/checkpoint.pt if it exists')
    p.add_argument('--keep-checkpoint', action='store_true', help='keep checkpoint.pt (about 0.7 GB) after the run')
    args = p.parse_args(argv)
    if args.backbone == 'terramind' and args.adapt == 'dense':
        p.error('--backbone terramind needs --adapt moelora (dense LoRA targets the ViT block layout)')
    if args.adapt == 'moelora' and (args.rank != 32 or args.attn_only or args.freeze_pretrained):
        p.error('--rank, --attn-only and --freeze-pretrained apply to --adapt dense only')
    return args


def config_name(args):
    """Run directory name: the design and every switch that differs from the primary model."""
    base = 'detaillora' if (args.backbone, args.adapt) == ('vit', 'dense') else f'detail_{args.backbone}_{args.adapt}'
    parts = [base] + [f'r{args.rank}'] * (args.rank != 32) + ['attnonly'] * args.attn_only + \
        ['frozen'] * args.freeze_pretrained + ['nodetail'] * args.no_detail + \
        ['refanchors'] * args.reference_anchors + ['noaug'] * args.no_augment
    return '_'.join(parts)


def make_optimizer(model, args):
    """AdamW in three groups: LoRA (--lr), heads and detail branch (--lr), unfrozen pretrained tensors
    (--pretrained-lr). The heads are group 1, the group whose rate the reference train loop logs."""
    groups = {'adapters': [], 'heads': [], 'pretrained': []}
    for name, p in model.named_parameters():
        if p.requires_grad:
            part = 'heads' if not name.startswith('backbone.') else 'adapters' if is_adapter(name) else 'pretrained'
            groups[part].append(p)
    param_groups = [{'params': groups['adapters']}, {'params': groups['heads']}]
    if groups['pretrained']:
        param_groups.append({'params': groups['pretrained'], 'lr': args.pretrained_lr})
    return torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    run_dir = args.out / config_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / 'checkpoint.pt'
    resuming = args.resume and checkpoint.exists()
    log_file = open(run_dir / 'log.txt', 'a' if resuming else 'w')
    log(json.dumps(vars(args), default=str), log_file)
    wandb_log.init('detail_lora', run_dir, vars(args), reference_train)  # off unless WANDB_API_KEY is set (.env)

    # Data: the holdout is the val split; train.py-style test split is not scored (test -> val in our layouts)
    train_set = AugmentedSARMultiTask(args.data, 'train', limit=args.limit, augment=not args.no_augment)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers,
                              collate_fn=collate, persistent_workers=args.workers > 0)
    val_loader = DataLoader(SARMultiTask(args.data, 'val', limit=args.limit), args.batch_size,
                            num_workers=args.workers, collate_fn=collate)
    val_json = args.data / 'val' / 'instances.json'

    # Model, and the competition's parameter budget (asserted before any training)
    model = model_from_args(vars(args), pretrained=True).cuda()
    counts = count_parameters(model)
    check_budget(counts)
    log(f'parameters: {counts["total"]:,} total, {counts["trainable"]:,} trainable '
        f'({", ".join(f"{k} {v:,}" for k, v in counts.items() if k not in ("total", "trainable"))})', log_file)

    optimizer = make_optimizer(model, args)
    warmup, total = len(train_loader), args.epochs * len(train_loader)  # one epoch of warmup
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total))
    first_epoch, history, elapsed = 1, [], 0.0
    if resuming:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        schedule.load_state_dict(state['schedule'])
        torch.set_rng_state(state['rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        first_epoch, history, elapsed = state['epoch'] + 1, state['history'], state['elapsed']
        log(f'resumed after epoch {state["epoch"]} ({elapsed / 3600:.2f} h trained)', log_file)

    # Train, validating every --eval-every epochs; the reported model is the last-epoch model
    start, val_detections = time.time(), None
    for epoch in range(first_epoch, args.epochs + 1):
        reference_train.train_one_epoch(model, train_loader, optimizer, schedule, epoch, log_file)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val, val_detections = reference_train.evaluate(model, val_loader, val_json)
            history.append({'epoch': epoch, **val})
            log(f'epoch {epoch} val: {headline(val)}', log_file)
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'schedule': schedule.state_dict(),
                 'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(), 'epoch': epoch,
                 'history': history, 'elapsed': elapsed + time.time() - start}
        torch.save(state, run_dir / 'checkpoint.tmp')
        (run_dir / 'checkpoint.tmp').replace(checkpoint)  # atomic: a crash never leaves a half-written checkpoint
    train_hours = (elapsed + time.time() - start) / 3600
    if val_detections is None:  # resumed after the last epoch had finished: score it again
        val, val_detections = reference_train.evaluate(model, val_loader, val_json)

    # Ablation A2 (detection scoring), inference time, memory
    by_fusion = evaluate_fusions(model, val_loader, val_json, [f for f in FUSIONS if f != args.fusion])
    by_fusion[args.fusion] = {k: v for k, v in history[-1].items() if k not in ('epoch', 'confusion_matrix')}
    for fusion in FUSIONS:
        log(f'holdout, scoring {fusion}: {headline(by_fusion[fusion])}', log_file)
    inference_ms = time_inference(model, val_loader)
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    log(f'training {train_hours:.2f} h, peak GPU memory {peak_gb:.1f} GB, inference '
        + ', '.join(f'{k} {v:.1f} ms/image' for k, v in inference_ms.items()), log_file)

    summary = {'design': 'detail_lora', 'config': config_name(args), 'args': vars(args),
               'parameters': counts['total'], 'trainable_parameters': counts['trainable'],
               'parameter_breakdown': counts, 'train_hours': train_hours, 'peak_gpu_memory_gb': peak_gb,
               'inference_ms_per_image': inference_ms, 'gpu': torch.cuda.get_device_name(),
               'val_history': history, 'val': history[-1], 'val_by_fusion': by_fusion}
    (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / 'predictions_val.json').write_text(json.dumps(val_detections))
    torch.save(model.state_dict(), run_dir / 'model.pt')
    if not args.keep_checkpoint:
        checkpoint.unlink()
    wandb_log.finish(summary)


if __name__ == '__main__':
    main()
