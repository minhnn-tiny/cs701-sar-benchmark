"""Parameter counts (and forward/backward FLOPs) of the reference and of cv-detect's candidates, on the CPU.
Random weights (pretrained=False builds the same architecture); nothing is downloaded (HF_HUB_OFFLINE=1).
Run from the repo root:  HF_HUB_OFFLINE=1 uv run python workspace/sarmtl/scratch/cv-detect/params.py [--flops]
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_modules import DetectModel  # noqa: E402  (puts the repo root on sys.path)
from sarbench.adapters import add_adapters  # noqa: E402
from sarbench.backbones import build_backbone  # noqa: E402
from sarbench.model import MultiTaskModel  # noqa: E402

NEW = ('neck.prior.', 'neck.lateral.')  # D2's modules


def build(backbone_name, adapt, variant):
    backbone = build_backbone(backbone_name, pretrained=False)
    add_adapters(backbone, adapt)
    routing = adapt == 'moelora'
    if variant == 'reference':
        return MultiTaskModel(backbone, task_routing=routing)
    if variant == 'D1':
        return DetectModel(backbone, routing, prior_alpha=1.0)
    if variant == 'D2':
        return DetectModel(backbone, routing, spatial_prior=True)
    if variant == 'D2-wide':
        return DetectModel(backbone, routing, spatial_prior=True, prior_channels=(128, 256))
    raise ValueError(variant)


def counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = {'heads': 0, 'backbone adaptation': 0, 'new (D2)': 0}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = 'new (D2)' if name.startswith(NEW) else 'backbone adaptation' if name.startswith('backbone.') else 'heads'
        trainable[key] += p.numel()
    return total, trainable


def flops(model, train):
    """GFLOPs (2 x MACs; matmul/conv only) of one 512 x 512 image: eval forward, or a training forward + backward."""
    torch.manual_seed(0)
    images = torch.randn(1, 1, 512, 512)
    if train:
        model.train()
        boxes = torch.tensor([[100., 100., 108., 108.], [200., 220., 216., 236.], [300., 40., 364., 90.]])
        targets = [{'boxes': boxes, 'labels': torch.tensor([9, 9, 9])}]
        with FlopCounterMode(display=False) as fc:
            logits, _, losses = model(images, targets)
            (logits.sum() + sum(losses.values())).backward()
    else:
        model.eval()
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(images)
    return fc.get_total_flops() / 1e9, fc.get_flop_counts()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--flops', action='store_true', help='also count FLOPs (CPU forward/backward at 512 px, ~1 min)')
    p.add_argument('--configs', default='vit:lora:reference,vit:lora:D1,vit:lora:D2,vit:lora:D2-wide,'
                                        'terramind:moelora:reference,terramind:moelora:D2')
    args = p.parse_args()
    torch.set_num_threads(16)
    ref = None
    for cfg in args.configs.split(','):
        backbone_name, adapt, variant = cfg.split(':')
        model = build(backbone_name, adapt, variant)
        total, tr = counts(model)
        line = (f'{backbone_name:9s} + {adapt:7s} {variant:9s}: total {total:,}  trainable {sum(tr.values()):,} = '
                + ' + '.join(f'{k} {v:,}' for k, v in tr.items()))
        print(line, flush=True)
        if args.flops and backbone_name == 'vit':
            ev, ev_by = flops(model, train=False)
            trn, _ = flops(model, train=True)
            mods = {k.split('.', 1)[1]: v for k, v in ev_by.items() if k.count('.') == 1}  # top-level submodules
            if variant == 'reference':
                ref = (ev, trn)
            rel = f'  (x{ev / ref[0]:.3f} eval, x{trn / ref[1]:.3f} train vs reference)' if ref else ''
            print(f'    GFLOPs per 512 px image: eval forward {ev:.1f}, train forward+backward {trn:.1f}{rel}')
            print('    eval forward GFLOPs by top-level module: ' + ', '.join(f'{k} {sum(v.values()) / 1e9:.1f}'
                                                                     for k, v in mods.items()), flush=True)


if __name__ == '__main__':
    main()
