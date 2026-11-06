"""Train R50-FPN-MT (r50fpn_model.py) jointly for classification and detection, then score it on the data's val and
test splits with the reference decoding. The training loop, evaluation and schedule are root train.py's; the protocol
is the baseline's (AdamW, batch 16, 24 epochs, 512 px, the same augmentation).
Example (from the repo root, through the GPU runner):
  workspace/gpu_run.sh -- uv run python workspace/r50fpn/train.py --data dataset/holdout_sarmtl --seed 0 \
      --out workspace/r50fpn/runs/H_s0"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # the repo root, for sarbench/ and the root train.py
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / 'workspace' / 'sarmtl'), str(ROOT / 'workspace')]
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import wandb_log  # noqa: E402
from r50fpn_model import REFERENCE_DECODING, build_model, count_parameters, set_decoding  # noqa: E402
from sarbench.data import SARMultiTask, collate  # noqa: E402

# root train.py under another name: this file is also called train.py
_spec = importlib.util.spec_from_file_location('root_train', ROOT / 'train.py')
root = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(root)

MAX_TOTAL, MAX_TRAINABLE = 130_000_000, 40_000_000  # the competition's budget, heads and frozen weights included


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=ROOT / 'dataset' / 'holdout_sarmtl')
    p.add_argument('--lora-rank', type=int, default=256, help='LoRA rank on the frozen fc6 of the box head')
    p.add_argument('--rot90', action='store_true', help='variant: random 90-degree rotations on top of the flips')
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4, help='everything outside the ResNet body')
    p.add_argument('--backbone-lr', type=float, help='the ResNet body (default: --lr)')
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--eval-every', type=int, default=4, help='validate every N epochs, and after the last one')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, help='use only the first N images of every split (quick checks)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=HERE / 'runs')
    args = p.parse_args()
    if args.backbone_lr is None:
        args.backbone_lr = args.lr
    return args


def rot90(image, boxes, k):
    """Rotate a square [C, S, S] image counter-clockwise by k x 90 degrees, with its XYXY boxes: each step maps a point
    (x, y) to (y, S - x)."""
    size = image.shape[-1]
    for _ in range(k):
        boxes = torch.stack([boxes[:, 1], size - boxes[:, 2], boxes[:, 3], size - boxes[:, 0]], dim=1)
    return torch.rot90(image, k, dims=(-2, -1)), boxes


class Rot90(torch.utils.data.Dataset):
    """Training images rotated by a random multiple of 90 degrees (after sarbench's flips, so together they draw from
    all 8 symmetries of the square: overhead imagery has no canonical up). The draw uses the worker's seeded RNG."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        image, label, target = self.base[i]
        image, boxes = rot90(image, target['boxes'].as_subclass(torch.Tensor), int(torch.randint(4, ())))
        return image, label, {**target, 'boxes': boxes}


def make_loader(args, split):
    """root train.py's loader, with the training order from its own generator (seeded runs see the same batches)."""
    train = split == 'train'
    dataset = SARMultiTask(args.data, split, train=train, limit=args.limit)
    if train and args.rot90:
        dataset = Rot90(dataset)
    generator = torch.Generator().manual_seed(args.seed) if train else None
    return DataLoader(dataset, args.batch_size, shuffle=train, num_workers=args.workers, collate_fn=collate,
                      persistent_workers=train and args.workers > 0, generator=generator)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    run_dir = args.out / 'r50fpn'
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / 'log.txt', 'w')
    log = lambda message: root.log(message, log_file)  # noqa: E731
    log(json.dumps(vars(args), default=str))
    wandb_log.init('r50fpn', run_dir, vars(args), root)  # off unless WANDB_API_KEY is set (.env)

    train_loader, val_loader, test_loader = (make_loader(args, split) for split in ('train', 'val', 'test'))

    # Model, budget asserted before anything trains
    model = build_model(coco=True, lora_rank=args.lora_rank)
    parameters, trainable, parts = count_parameters(model)
    log(f'parameters: {parameters:,} total, {trainable:,} trainable ('
        + ', '.join(f'{k} {v:,}' for k, v in parts.items()) + ')')
    assert parameters <= MAX_TOTAL, f'{parameters:,} total parameters > {MAX_TOTAL:,}'
    assert trainable <= MAX_TRAINABLE, f'{trainable:,} trainable parameters > {MAX_TRAINABLE:,}'
    log(f'budget OK: {parameters:,} <= {MAX_TOTAL:,} total, {trainable:,} <= {MAX_TRAINABLE:,} trainable')
    model.cuda()

    # root train.py's AdamW and warmup + cosine; group 1 (whose lr root train.py logs) holds the non-body parameters
    body = [p for p in model.body.parameters() if p.requires_grad]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('body.')]
    optimizer = torch.optim.AdamW([{'params': body, 'lr': args.backbone_lr}, {'params': rest}], lr=args.lr,
                                  weight_decay=args.weight_decay)
    warmup, total = len(train_loader), args.epochs * len(train_loader)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: root.lr_factor(step, warmup, total))

    # Train, validating every --eval-every epochs; the reported model is the last-epoch one
    # shortcut: no per-epoch checkpoint or resume, so a crash loses the run (<= 2.2 h at 36 epochs); add one before
    # longer schedules
    set_decoding(model, **REFERENCE_DECODING)
    history, start = [], time.time()
    for epoch in range(1, args.epochs + 1):
        root.train_one_epoch(model, train_loader, optimizer, schedule, epoch, log_file)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val, val_detections = root.evaluate(model, val_loader, args.data / 'val' / 'instances.json')
            history.append({'epoch': epoch, **val})
            log(f'epoch {epoch} val: {root.headline(val)}')
    train_hours = (time.time() - start) / 3600
    torch.save(model.state_dict(), run_dir / 'model.pt')  # before the final evaluation: a failure there keeps it

    torch.cuda.synchronize()
    start = time.time()
    test, test_detections = root.evaluate(model, test_loader, args.data / 'test' / 'instances.json')
    ms_per_image = 1000 * (time.time() - start) / len(test_loader.dataset)
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    log(f'test: {root.headline(test)}; {ms_per_image:.1f} ms per image (batch {args.batch_size}, data loading '
        f'included)\ntraining {train_hours:.2f} h, peak GPU memory {peak_gb:.1f} GB')

    summary = {'args': vars(args), 'config': 'r50fpn', 'parameters': parameters, 'trainable_parameters': trainable,
               'trainable_by_part': parts, 'train_hours': train_hours, 'peak_gpu_memory_gb': peak_gb,
               'inference_ms_per_image': ms_per_image, 'gpu': torch.cuda.get_device_name(),
               'decoding': REFERENCE_DECODING, 'val_history': history, 'val': history[-1], 'test': test}
    (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / 'predictions_val.json').write_text(json.dumps(val_detections))
    (run_dir / 'predictions_test.json').write_text(json.dumps(test_detections))
    wandb_log.finish(summary)


if __name__ == '__main__':
    main()
