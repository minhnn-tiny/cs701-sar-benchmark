"""Parameter counts (CPU, random weights: same architecture as pretrained) for the cv-adapt candidates.
Run from the repo root:  HF_HUB_OFFLINE=1 uv run python workspace/sarmtl/scratch/cv-adapt/count_params.py
Every model is the reference MultiTaskModel (heads unchanged) with a different backbone adaptation."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wide_lora import add_wide_lora, param_groups, ROOT  # noqa: E402  (also puts the repo root on sys.path)
from sarbench.adapters import add_adapters  # noqa: E402
from sarbench.backbones import build_backbone  # noqa: E402
from sarbench.model import MultiTaskModel  # noqa: E402

TOTAL_CAP, TRAINABLE_CAP = 130_000_000, 40_000_000


def split(model):
    """total, trainable, and trainable split into heads / backbone LoRA / unfrozen pretrained backbone tensors."""
    total = sum(p.numel() for p in model.parameters())
    out = {'heads': 0, 'backbone_lora': 0, 'backbone_pretrained_unfrozen': 0}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not name.startswith('backbone.'):
            out['heads'] += p.numel()
        elif '.down.' in name or '.up.' in name or '.routers.' in name:
            out['backbone_lora'] += p.numel()
        else:
            out['backbone_pretrained_unfrozen'] += p.numel()
    return total, sum(out.values()), out


def report(label, model):
    total, trainable, parts = split(model)
    ok = 'OK' if total <= TOTAL_CAP and trainable <= TRAINABLE_CAP else 'OVER BUDGET'
    print(f'{label}\n  total {total:,}  trainable {trainable:,}  [{ok}; trainable headroom '
          f'{TRAINABLE_CAP - trainable:,}]')
    print('  trainable split: ' + ', '.join(f'{k} {v:,}' for k, v in parts.items()))
    return total, trainable, parts


def build(kind, adapt, **kw):
    backbone = build_backbone(kind, pretrained=False)
    if adapt in ('lora', 'moelora'):
        add_adapters(backbone, adapt)
    elif adapt == 'wide':
        add_wide_lora(backbone, kind=kind, **kw)
    return MultiTaskModel(backbone, task_routing=adapt == 'moelora')


def main():
    torch.manual_seed(0)
    print('--- references (must match train.py logs: ViT + LoRA 107,039,257 total, 20,997,913 trainable)')
    report('vit + lora (reference)', build('vit', 'lora'))
    report('vit + moelora', build('vit', 'moelora'))
    print('--- WideLoRA on ViT: LoRA on qkv, proj, fc1, fc2 of all 12 blocks (+ small tensors unfrozen)')
    for r in (16, 32, 64, 128):
        report(f'vit + wide LoRA r{r} (rank-stable alpha {8 * r ** 0.5:g}) + small tensors', build('vit', 'wide', rank=r))
    report('vit + wide LoRA r64, small tensors frozen (ablation)',
           build('vit', 'wide', rank=64, train_small=False))
    print('--- WideLoRA on TerraMind: LoRA on qkv, proj, fc1, fc3, fc2 (+ patch projection, norms unfrozen)')
    report('terramind + lora (reference)', build('terramind', 'lora'))
    report('terramind + moelora (reference)', build('terramind', 'moelora'))
    report('terramind + wide LoRA r64 + small tensors', build('terramind', 'wide', rank=64))

    print('--- WideLoRA r64 on ViT: what exactly is unfrozen, and the optimizer groups')
    m = build('vit', 'wide', rank=64)
    small = {}
    for name, p in m.backbone.named_parameters():
        if p.requires_grad and '.down.' not in name and '.up.' not in name:
            key = name.replace('vit.', '')
            key = 'blocks.*.' + key.split('.', 2)[2] if key.startswith('blocks.') else key
            small[key] = small.get(key, 0) + p.numel()
    for k, v in small.items():
        print(f'  {k}: {v:,}')
    for g, label in zip(param_groups(m), ('LoRA A/B @1e-4', 'unfrozen pretrained @2e-5', 'heads @1e-4')):
        print(f'  group {label}: {sum(p.numel() for p in g["params"]):,} in {len(g["params"])} tensors')
    print('--- XFuse (cross-task inference fusion) adds 0 parameters: it only rescales outputs of the same model')


if __name__ == '__main__':
    main()
