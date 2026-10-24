"""Activation memory the backbone keeps for backward (bf16 autocast, as in training), per image, on the CPU.
Run from the repo root:  HF_HUB_OFFLINE=1 uv run python workspace/sarmtl/scratch/cv-adapt/activation_memory.py
Counts the unique storages autograd saves (saved_tensors_hooks); heads excluded (identical in every config)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wide_lora import add_wide_lora  # noqa: E402
from sarbench.adapters import add_adapters  # noqa: E402
from sarbench.backbones import build_backbone  # noqa: E402


def saved_bytes(adapt, **kw):
    torch.manual_seed(0)
    bb = build_backbone('vit', pretrained=False)
    if adapt == 'lora':
        add_adapters(bb, 'lora')
    elif adapt == 'wide':
        add_wide_lora(bb, **kw)
    storages = {}

    def pack(t):
        s = t.untyped_storage()
        storages[s.data_ptr()] = s.nbytes()
        return t

    x = torch.randn(1, 1, 512, 512)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = bb(x)
    out.float().sum().backward()
    params = {p.untyped_storage().data_ptr() for p in bb.parameters()}
    return sum(n for ptr, n in storages.items() if ptr not in params)  # weights are not activations


def main():
    ref = saved_bytes('lora')
    for name, adapt, kw in (('lora r16 qkv+proj (reference)', 'lora', {}),
                            ('WideLoRA r64 + small tensors', 'wide', dict(rank=64)),
                            ('full fine-tuning', 'full', {})):
        b = ref if adapt == 'lora' else saved_bytes(adapt, **kw)
        print(f'{name:36s} {b / 2**20:8.1f} MiB/image  x16 = {16 * b / 2**30:5.2f} GiB  '
              f'(+{16 * (b - ref) / 2**30:.2f} GiB vs reference)')


if __name__ == '__main__':
    main()
