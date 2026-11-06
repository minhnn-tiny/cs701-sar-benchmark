"""Backbone training FLOPs per image (forward + backward), counted on the meta device (no data, no GPU).
Run from the repo root:  HF_HUB_OFFLINE=1 uv run python workspace/sarmtl/scratch/cv-adapt/flops.py
Only the backbone is counted: the heads (neck, RPN, RoI) are identical in every configuration."""
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wide_lora import add_wide_lora  # noqa: E402
from sarbench.adapters import add_adapters, set_task  # noqa: E402
from sarbench.backbones import build_backbone  # noqa: E402


def count(adapt, passes=1, **kw):
    with torch.device('meta'):
        bb = build_backbone('vit', pretrained=False)
        if adapt in ('lora', 'moelora'):
            add_adapters(bb, adapt)
        elif adapt == 'wide':
            add_wide_lora(bb, **kw)
        # 'full': everything trains
        x = torch.randn(1, 1, 512, 512)
    fwd = FlopCounterMode(display=False)
    with fwd:
        outs = []
        for task in range(passes):
            set_task(bb, task)
            outs.append(bb(x))
    total = FlopCounterMode(display=False)
    with total:
        outs = []
        for task in range(passes):
            set_task(bb, task)
            outs.append(bb(x))
        sum(o.float().sum() for o in outs).backward()
    return fwd.get_total_flops(), total.get_total_flops()


def main():
    rows = [('lora r16 qkv+proj (reference)', count('lora')),
            ('moelora (2 backbone passes)', count('moelora', passes=2)),
            ('WideLoRA r64 + small tensors', count('wide', rank=64)),
            ('WideLoRA r32 + small tensors', count('wide', rank=32)),
            ('full fine-tuning (reference, over budget)', count('full'))]
    ref = rows[0][1][1]
    print(f'{"backbone config":45s} {"fwd GFLOP":>10s} {"fwd+bwd GFLOP":>14s} {"vs ref":>7s}')
    for name, (f, t) in rows:
        print(f'{name:45s} {f / 1e9:10.1f} {t / 1e9:14.1f} {t / ref:7.3f}')


if __name__ == '__main__':
    main()
