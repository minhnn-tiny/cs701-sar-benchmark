"""FLOPs of one full training step (forward + backward) per image, split backbone vs heads (neck, RPN, RoI, cls head),
for the reference ViT + LoRA and for WideLoRA, on the CPU with random weights (one image, 3 boxes).
Run from the repo root:  HF_HUB_OFFLINE=1 uv run python workspace/sarmtl/scratch/cv-adapt/step_flops.py
FLOPs only: NMS, anchor matching, sampling and data loading are not counted, and fp32 heads run slower per FLOP."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wide_lora import add_wide_lora  # noqa: E402
from sarbench.adapters import add_adapters  # noqa: E402
from sarbench.backbones import build_backbone  # noqa: E402
from sarbench.model import MultiTaskModel  # noqa: E402


def step_flops(adapt):
    torch.manual_seed(0)
    bb = build_backbone('vit', pretrained=False)
    add_adapters(bb, 'lora') if adapt == 'lora' else add_wide_lora(bb, rank=64, alpha=64)
    model = MultiTaskModel(bb, task_routing=False).train()
    x = torch.randn(1, 1, 512, 512)
    targets = [{'boxes': torch.tensor([[100., 100., 116., 116.], [200., 220., 240., 260.], [300., 40., 309., 49.]]),
                'labels': torch.tensor([9, 9, 9])}]
    with FlopCounterMode(display=False) as fc:
        logits, _, losses = model(x, targets)
        (F.cross_entropy(logits, torch.tensor([8])) + sum(losses.values())).backward()
    per_module = fc.get_flop_counts()  # {module name: {op: flops}}, nested names included
    total = fc.get_total_flops()
    backbone = sum(per_module.get('MultiTaskModel.backbone', {}).values())
    return total, backbone


def main():
    # CPU SDPA (_scaled_dot_product_flash_attention_for_cpu) has no FLOP formula, so attention is added analytically:
    # forward 4 N^2 d per block (QK^T, AV), backward 8 N^2 d; N = 1025 tokens, d = 768, 12 blocks. This equals the
    # gap between the meta-device backbone count in flops.out (466.6) and the CPU count here (350.4).
    n, d = 1025, 768
    attention = 12 * 12 * n * n * d
    print(f'attention added analytically: {attention / 1e9:.1f} GFLOP/image')
    for adapt in ('lora', 'wide'):
        total, backbone = step_flops(adapt)
        total, backbone = total + attention, backbone + attention
        print(f'{adapt:5s}: step {total / 1e9:6.1f} GFLOP/image, backbone {backbone / 1e9:6.1f} '
              f'({100 * backbone / total:.0f} %), rest (neck, RPN, RoI, cls head) {(total - backbone) / 1e9:6.1f}')


if __name__ == '__main__':
    main()
