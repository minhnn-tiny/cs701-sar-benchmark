"""Train WideLoRA-SP (docs/DESIGN.md) jointly for classification and detection, then score it on val and test with the
reference decoding and with joint decoding (JD). The CLI is root train.py's, with the design's defaults, plus
--rank, --pretrained-lr and the ablation flag --no-sp. The training loop, evaluation and schedule are root train.py's.
Example (from the repo root, through the GPU runner):
  workspace/gpu_run.sh -- uv run python workspace/sarmtl/train.py --data dataset/holdout_sarmtl --seed 0 \
      --out workspace/sarmtl/runs/P_s0"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the repo root, for sarbench/ and the root train.py
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))
sys.path.insert(2, str(ROOT / 'workspace'))
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import wandb_log  # noqa: E402
from sarbench.data import SARMultiTask, collate  # noqa: E402
from sarmtl_model import (JD_DECODING, REFERENCE_DECODING, build_model, count_parameters, param_groups,  # noqa: E402
                          set_decoding)

# root train.py under another name: this file is also called train.py
_spec = importlib.util.spec_from_file_location('root_train', ROOT / 'train.py')
root = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(root)

MAX_TOTAL, MAX_TRAINABLE = 130_000_000, 40_000_000  # the competition's budget, heads and frozen weights included


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=ROOT / 'dataset' / 'holdout_sarmtl')
    p.add_argument('--backbone', default='vit', choices=['vit', 'terramind'])
    p.add_argument('--init', default='pretrained', choices=['pretrained', 'scratch'])
    p.add_argument('--adapt', default='widelora', choices=['widelora', 'lora', 'moelora', 'full'])
    p.add_argument('--rank', type=int, default=64, help='WideLoRA rank (scale 8 / sqrt(rank))')
    p.add_argument('--no-sp', dest='sp', action='store_false', help='ablation A1: no spatial-prior branch')
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4, help='LoRA and heads (SP included)')
    p.add_argument('--backbone-lr', type=float, help='root train.py compatibility: same as --lr if not given')
    p.add_argument('--pretrained-lr', type=float, default=2e-5, help='the unfrozen pretrained tensors')
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--eval-every', type=int, default=4, help='validate every N epochs, and after the last one')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, help='use only the first N images of every split (quick checks)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=Path('runs'))
    args = p.parse_args()
    if args.backbone_lr is None:
        args.backbone_lr = args.lr
    return args


def config_name(args):
    rank = f'_r{args.rank}' if args.adapt == 'widelora' and args.rank != 64 else ''
    return f'{args.backbone}_{args.adapt}{rank}' + ('_sp' if args.sp else '')


def make_loader(args, split):
    """root train.py's loader, except that the training order comes from its own generator: runs of the same seed
    (the design and its A1 ablation) then see the same batches and flips."""
    train = split == 'train'
    dataset = SARMultiTask(args.data, split, train=train, limit=args.limit)
    generator = torch.Generator().manual_seed(args.seed) if train else None
    return DataLoader(dataset, args.batch_size, shuffle=train, num_workers=args.workers, collate_fn=collate,
                      persistent_workers=train and args.workers > 0, generator=generator)


def evaluate(model, loader, gt_json, decoding):
    """root train.py's evaluate under one decoding, timed: (metrics, detections, ms per image incl. data loading)."""
    set_decoding(model, **decoding)
    torch.cuda.synchronize()
    start = time.time()
    metrics, detections = root.evaluate(model, loader, gt_json)
    ms = 1000 * (time.time() - start) / len(loader.dataset)
    set_decoding(model, **REFERENCE_DECODING)
    return metrics, detections, ms


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    run_dir = args.out / config_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / 'log.txt', 'w')
    log = lambda message: root.log(message, log_file)  # noqa: E731
    log(json.dumps(vars(args), default=str))
    wandb_log.init('sarmtl', run_dir, vars(args), root)  # off unless WANDB_API_KEY is set (.env)

    # Data
    train_loader, val_loader, test_loader = (make_loader(args, split) for split in ('train', 'val', 'test'))

    # Model (on the CPU first: adapters and SP are created there), budget asserted before anything trains
    model = build_model(args.backbone, args.init, args.adapt, args.rank, args.sp, args.seed)
    parameters, trainable, kinds = count_parameters(model)
    log(f'parameters: {parameters:,} total, {trainable:,} trainable ('
        + ', '.join(f'{k} {v:,}' for k, v in kinds.items()) + ')')
    assert parameters <= MAX_TOTAL, f'{parameters:,} total parameters > {MAX_TOTAL:,}'
    assert trainable <= MAX_TRAINABLE, f'{trainable:,} trainable parameters > {MAX_TRAINABLE:,}'
    log(f'budget OK: {parameters:,} <= {MAX_TOTAL:,} total, {trainable:,} <= {MAX_TRAINABLE:,} trainable')
    model.cuda()

    # Optimizer and schedule: root train.py's AdamW and warmup + cosine, with a third group for the pretrained tensors
    if args.adapt == 'widelora':
        groups = param_groups(model, args.lr, args.pretrained_lr)
    else:  # root train.py's two groups
        groups = [{'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': args.backbone_lr},
                  {'params': [p for n, p in model.named_parameters() if not n.startswith('backbone.')]}]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    warmup, total = len(train_loader), args.epochs * len(train_loader)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: root.lr_factor(step, warmup, total))

    # Train, validating with the reference decoding every --eval-every epochs; the reported model is the last-epoch one
    set_decoding(model, **REFERENCE_DECODING)
    history, start = [], time.time()
    for epoch in range(1, args.epochs + 1):
        root.train_one_epoch(model, train_loader, optimizer, schedule, epoch, log_file)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val, val_detections = root.evaluate(model, val_loader, args.data / 'val' / 'instances.json')
            history.append({'epoch': epoch, **val})
            log(f'epoch {epoch} val: {root.headline(val)}')
    train_hours = (time.time() - start) / 3600

    # Final scores: val with JD (timed: inference ms/image), test with both decodings
    val_jd, val_jd_detections, ms_per_image = evaluate(model, val_loader, args.data / 'val' / 'instances.json',
                                                        JD_DECODING)
    log(f'val, JD: {root.headline(val_jd)}; {ms_per_image:.1f} ms per image (batch {args.batch_size}, data loading '
        f'included)')
    test, test_detections, _ = evaluate(model, test_loader, args.data / 'test' / 'instances.json', REFERENCE_DECODING)
    test_jd, _, _ = evaluate(model, test_loader, args.data / 'test' / 'instances.json', JD_DECODING)
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    log(f'test: {root.headline(test)}\ntest, JD: {root.headline(test_jd)}\n'
        f'training {train_hours:.2f} h, peak GPU memory {peak_gb:.1f} GB')

    # Save: root train.py's metrics.json keys (val/test: reference decoding), plus the JD scores and the costs
    summary = {'args': vars(args), 'config': config_name(args), 'parameters': parameters,
               'trainable_parameters': trainable, 'trainable_by_kind': kinds, 'train_hours': train_hours,
               'peak_gpu_memory_gb': peak_gb, 'inference_ms_per_image': ms_per_image,
               'gpu': torch.cuda.get_device_name(), 'decodings': {'reference': REFERENCE_DECODING, 'jd': JD_DECODING},
               'val_history': history, 'val': history[-1], 'val_jd': val_jd, 'test': test, 'test_jd': test_jd}
    (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / 'predictions_val.json').write_text(json.dumps(val_detections))
    (run_dir / 'predictions_val_jd.json').write_text(json.dumps(val_jd_detections))
    (run_dir / 'predictions_test.json').write_text(json.dumps(test_detections))
    torch.save(model.state_dict(), run_dir / 'model.pt')
    wandb_log.finish(summary)


if __name__ == '__main__':
    main()
