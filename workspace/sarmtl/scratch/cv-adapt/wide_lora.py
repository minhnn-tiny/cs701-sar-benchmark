"""Candidate WideLoRA: LoRA on every linear layer of every block, plus the small pretrained tensors unfrozen.
Reuses sarbench.adapters.LoRA unchanged; sarbench/ is never edited.
Scale gamma = alpha / r is rank-stable (rsLoRA): gamma = 8 / sqrt(r), i.e. alpha = 8 sqrt(r) -> r16: 2 (the reference),
r32: 1.41, r64: 1. Under Adam the weight-space step of LoRA grows with r at a fixed gamma, so a fixed gamma = 2 at r64
would take ~2-4x the reference's steps.

    backbone = build_backbone('vit', pretrained=True)
    add_wide_lora(backbone, rank=64, alpha=64)           # freezes the backbone, wraps, unfreezes the small tensors
    model = MultiTaskModel(backbone, task_routing=False)
    groups = param_groups(model, lr=1e-4, pretrained_lr=2e-5)  # 3 AdamW groups
"""
import sys
from pathlib import Path

from torch import nn

ROOT = Path(__file__).resolve().parents[4]  # repo root, for sarbench/
sys.path.insert(0, str(ROOT))
from sarbench.adapters import LoRA  # noqa: E402

# Linear layers wrapped in every block. timm ViT: Mlp(fc1, fc2); TerraMind: GatedMlp(fc1, fc3, fc2), no biases.
TARGETS = {'vit': ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'),
           'terramind': ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2', 'mlp.fc3')}


def _get(module, dotted):
    for part in dotted.split('.'):
        module = getattr(module, part)
    return module


def _set(module, dotted, value):
    *parents, last = dotted.split('.')
    for part in parents:
        module = getattr(module, part)
    setattr(module, last, value)


def rank_stable_alpha(rank):
    """alpha giving gamma = alpha / rank = 8 / sqrt(rank): the reference's gamma = 2 at rank 16."""
    return 8 * rank ** 0.5


def add_wide_lora(backbone, rank=64, alpha=None, kind='vit', train_small=True):
    """Freeze the backbone, wrap TARGETS[kind] of every block with LoRA(rank, alpha) (B = 0: starts at the
    pretrained model), then (train_small) unfreeze the small pretrained tensors: patch embedding, position
    embedding, CLS token, every LayerNorm, and the biases of the wrapped linear layers.
    Returns the list of unfrozen pretrained parameters (for their own learning rate)."""
    alpha = rank_stable_alpha(rank) if alpha is None else alpha
    backbone.requires_grad_(False)
    for blk in backbone.blocks:
        for name in TARGETS[kind]:
            _set(blk, name, LoRA(_get(blk, name), rank=rank, alpha=alpha))
    if not train_small:
        return []
    small = []
    for name, p in backbone.named_parameters():
        is_lora = '.down.' in name or '.up.' in name
        if is_lora:
            continue
        leaf = name.rsplit('.', 1)[-1]
        module_name = name.rsplit('.', 1)[0]
        module = _get(backbone, module_name) if '.' in name else backbone
        if ('LayerNorm' in type(module).__name__ or leaf in ('pos_embed', 'cls_token')  # TerraMind: own LayerNorm
                or 'patch_embed' in name or 'encoder_embeddings' in name or 'encoder_norm' in name
                or (leaf == 'bias' and isinstance(module, nn.Linear))):
            p.requires_grad_(True)
            small.append(p)
    return small


def param_groups(model, lr=1e-4, pretrained_lr=2e-5):
    """Three AdamW groups: LoRA A/B at lr (as the reference), unfrozen pretrained tensors at pretrained_lr
    (the TA's full fine-tuning backbone lr), all new modules (heads) at lr."""
    lora, pretrained, heads = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not name.startswith('backbone.'):
            heads.append(p)
        elif '.down.' in name or '.up.' in name:
            lora.append(p)
        else:
            pretrained.append(p)
    return [{'params': lora, 'lr': lr}, {'params': pretrained, 'lr': pretrained_lr}, {'params': heads, 'lr': lr}]
