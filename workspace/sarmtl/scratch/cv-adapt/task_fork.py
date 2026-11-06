"""Candidate TaskFork: WideLoRA with task-specific LoRA in the top k blocks (shared blocks run once, top blocks twice).

    blocks 0..11-k : WideLoRA (shared LoRA r64 on qkv, proj, fc1, fc2) - run once
    blocks 12-k..11: frozen weights shared; LoRA per task (classification rank r_cls, detection rank r_det) - run twice
    final norm     : shared; classification = mean of the cls-path tokens, detection = the det-path tokens

Imports sarbench unchanged (MultiTaskModel is subclassed only to route the two token sets). CPU checks:
    HF_HUB_OFFLINE=1 uv run python workspace/sarmtl/scratch/cv-adapt/task_fork.py"""
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wide_lora import ROOT, TARGETS, _get, _set, add_wide_lora  # noqa: E402,F401  (ROOT: repo root on sys.path)
from sarbench.backbones import build_backbone  # noqa: E402
from sarbench.model import MultiTaskModel  # noqa: E402


class TaskLoRA(nn.Module):
    """y = W x + gamma_t B_t A_t x with one (A_t, B_t) per task t; B_t = 0 at start (the pretrained layer);
    gamma_t = 8 / sqrt(r_t), rank-stable as in wide_lora.py (r16: 2, the reference; r64: 1)."""

    def __init__(self, base: nn.Linear, ranks=(16, 64)):
        super().__init__()
        self.base = base
        self.down = nn.ModuleList(nn.Linear(base.in_features, r, bias=False) for r in ranks)
        self.up = nn.ModuleList(nn.Linear(r, base.out_features, bias=False) for r in ranks)
        for up in self.up:
            nn.init.zeros_(up.weight)
        self.scales = [8 / r ** 0.5 for r in ranks]
        self.task = 0

    def forward(self, x):
        return self.base(x) + self.scales[self.task] * self.up[self.task](self.down[self.task](x))


class ForkedViT(nn.Module):
    """Wraps sarbench's ViT (timm) so that blocks[:-k] run once and blocks[-k:] once per task."""

    def __init__(self, vit_backbone, k=4, ranks=(16, 64), shared_rank=64):
        super().__init__()
        self.inner = vit_backbone
        self.embed_dim = vit_backbone.embed_dim
        self.k = k
        add_wide_lora(vit_backbone, rank=shared_rank)  # every block (rank-stable alpha), then replace the top k
        for blk in vit_backbone.blocks[-k:]:
            for name in TARGETS['vit']:
                lora = _get(blk, name)  # sarbench LoRA from add_wide_lora: keep its (trainable) base, drop A/B
                _set(blk, name, TaskLoRA(lora.base, ranks))

    def set_task(self, task):
        for m in self.inner.blocks[-self.k:].modules():
            if isinstance(m, TaskLoRA):
                m.task = task

    def forward_both(self, x):
        """images [B, 1, H, W] -> (cls-path tokens, det-path tokens), each [B, N, 768] (CLS token dropped)."""
        vit = self.inner.vit
        x = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(x))))
        shared = vit.blocks[:-self.k](x)
        out = []
        for task in (0, 1):  # 0 = classification, 1 = detection (as sarbench.adapters.set_task)
            self.set_task(task)
            out.append(vit.norm(vit.blocks[-self.k:](shared))[:, vit.num_prefix_tokens:])
        return out

    def forward(self, x):  # detection path only (not used by TaskForkModel)
        return self.forward_both(x)[1]


class TaskForkModel(MultiTaskModel):
    """MultiTaskModel whose classification head reads the cls-path tokens and whose neck reads the det-path tokens;
    the rest of forward is the reference's (precision rule included)."""

    def __init__(self, backbone: ForkedViT, num_classes=9):
        super().__init__(backbone, task_routing=False, num_classes=num_classes)

    def forward(self, images, targets=None):
        from torchvision.models.detection.image_list import ImageList
        cls_tokens, det_tokens = self.backbone.forward_both(images)
        logits = self.cls_head(cls_tokens.mean(dim=1))
        h, w = images.shape[-2] // 16, images.shape[-1] // 16
        features = self.neck(det_tokens.transpose(1, 2).unflatten(2, (h, w)))
        with torch.autocast(images.device.type, enabled=False):
            features = {name: f.float() for name, f in features.items()}
            image_list = ImageList(images, [tuple(images.shape[-2:])] * len(images))
            proposals, rpn_losses = self.rpn(image_list, features, targets)
            detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
        return logits, detections, {**rpn_losses, **roi_losses}


def build(k=4, ranks=(16, 64), pretrained=False):
    return TaskForkModel(ForkedViT(build_backbone('vit', pretrained=pretrained), k=k, ranks=ranks))


def main():
    torch.manual_seed(0)
    for k in (4, 2):
        m = build(k=k)
        total = sum(p.numel() for p in m.parameters())
        parts = {'heads': 0, 'shared LoRA': 0, 'cls-path LoRA': 0, 'det-path LoRA': 0, 'unfrozen pretrained': 0}
        for name, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if not name.startswith('backbone.'):
                parts['heads'] += p.numel()
            elif '.down.0.' in name or '.up.0.' in name:
                parts['cls-path LoRA'] += p.numel()
            elif '.down.1.' in name or '.up.1.' in name:
                parts['det-path LoRA'] += p.numel()
            elif '.down.' in name or '.up.' in name:
                parts['shared LoRA'] += p.numel()
            else:
                parts['unfrozen pretrained'] += p.numel()
        print(f'TaskFork k={k}, ranks cls/det 16/64, shared r64: total {total:,}  trainable {sum(parts.values()):,}')
        print('  ' + ', '.join(f'{a} {b:,}' for a, b in parts.items()))

    # FLOPs of the backbone part (both paths), forward + backward, meta device, per image
    for k in (4, 2):
        with torch.device('meta'):
            bb = ForkedViT(build_backbone('vit', pretrained=False), k=k)
            x = torch.randn(1, 1, 512, 512)
        with FlopCounterMode(display=False) as fc:
            a, b = bb.forward_both(x)
            (a.float().sum() + b.float().sum()).backward()
        print(f'backbone fwd+bwd, k={k}: {fc.get_total_flops() / 1e9:.1f} GFLOP/image '
              f'({fc.get_total_flops() / 466.6e9:.3f}x the reference LoRA 466.6)')

    # CPU correctness: training step reaches every trainable tensor; at init both paths equal the WideLoRA backbone
    m = build(k=4)
    m.train()
    x = torch.randn(2, 1, 512, 512)
    targets = [{'boxes': torch.tensor([[10., 10., 30., 30.]]), 'labels': torch.tensor([3])} for _ in range(2)]
    with torch.no_grad():
        cls_t, det_t = m.backbone.forward_both(x)
    assert torch.allclose(cls_t, det_t), 'B = 0 at init: both paths must start identical'
    logits, _, losses = m(x, targets)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])) + sum(losses.values())
    loss.backward()
    # B starts at 0, so A gets no gradient on the first step (as in any LoRA); check every B (up) and the rest
    missing = [n for n, p in m.named_parameters() if p.requires_grad and '.down.' not in n
               and (p.grad is None or p.grad.abs().sum() == 0)]
    print('params without gradient after one step (A excluded, expected none):', missing[:5], len(missing))


if __name__ == '__main__':
    main()
