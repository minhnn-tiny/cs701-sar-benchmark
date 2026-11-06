"""Bytes saved for backward (activations) per 512 px training image, reference vs D2, measured on the CPU under the
same bf16 autocast as train.py (RPN / RoI heads in fp32). Saved tensors are de-duplicated by storage. This is the part
of peak GPU memory that grows with the batch; x16 gives the batch-16 figure. Not a GPU measurement.
Run from the repo root:  HF_HUB_OFFLINE=1 uv run python workspace/sarmtl/scratch/cv-detect/memory.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from params import build  # noqa: E402


def saved_bytes(model):
    torch.manual_seed(0)
    images = torch.randn(1, 1, 512, 512)
    boxes = torch.tensor([[100., 100., 108., 108.], [200., 220., 216., 236.], [300., 40., 364., 90.]])
    targets = [{'boxes': boxes, 'labels': torch.tensor([9, 9, 9])}]
    seen, total, by_module = set(), [0], {}

    def pack(t):
        key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        if key not in seen:
            seen.add(key)
            total[0] += key[1]
        return t

    model.train()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logits, _, losses = model(images, targets)
    return total[0]


def main():
    torch.set_num_threads(16)
    ref = None
    for variant in ('reference', 'D2', 'D2-wide'):
        b = saved_bytes(build('vit', 'lora', variant))
        ref = ref or b
        print(f'vit + lora {variant:9s}: {b / 2**20:7.1f} MiB saved for backward per image, '
              f'x16 = {16 * b / 2**30:5.2f} GiB; vs reference {16 * (b - ref) / 2**30:+5.2f} GiB per batch of 16',
              flush=True)


if __name__ == '__main__':
    main()
