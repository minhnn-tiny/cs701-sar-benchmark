"""DetailLoRA: the reference multi-task model (sarbench/model.py) with four changes, each switchable:

1. Dense LoRA: LoRA on every linear layer of the ViT blocks (attn.qkv, attn.proj, mlp.fc1, mlp.fc2), rank 32, and
   the small pretrained tensors that meet SAR statistics first (patch embedding, position embedding, CLS token,
   LayerNorms) unfrozen, to be trained at a gentle learning rate.
2. Detail branch: a light conv stem on the pixels gives stride-4 and stride-8 maps, added (zero-initialised) to the
   ViTDet pyramid's two finest levels before their projection, so 5-15 px objects get real pixel detail instead of
   features upsampled from 16 px tokens.
3. Small anchors: one sqrt(2) step below every level's sizes (5.7 px to 181 px, 9 anchors per location).
4. Class-guided detection (inference only): all boxes of an image have the image's class, so each box's class
   distribution is combined with the classification head's (product of experts), see ClassGuidedRoIHeads.

build_model(adapt='moelora') instead keeps the reference's MoE-LoRA (task routers, the backbone run once per task),
so #2-#4 can be added to the TerraMind + MoE-LoRA baseline alone.
"""
import re
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import boxes as box_ops

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo root, for sarbench/
from sarbench.adapters import LoRA, add_adapters, set_task  # noqa: E402
from sarbench.model import MultiTaskModel, SimpleFeaturePyramid  # noqa: E402

LORA_TARGETS = ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')
# Pretrained backbone tensors trained with the adapters (at --pretrained-lr): first layer, positions, norms
UNFROZEN = re.compile(r'(patch_embed\.|pos_embed$|cls_token$|norm\d?\.(weight|bias)$)')
SMALL_ANCHORS = tuple(tuple(round(2 ** (k / 2), 2) for k in (j, j + 1, j + 2)) for j in range(5, 15, 2))
# ((5.66, 8.0, 11.31), (11.31, 16.0, 22.63), (22.63, 32.0, 45.25), (45.25, 64.0, 90.51), (90.51, 128.0, 181.02))

MAX_TOTAL, MAX_TRAINABLE = 130_000_000, 40_000_000  # competition rules


def add_dense_lora(backbone, rank=32, alpha=64, targets=LORA_TARGETS, unfreeze=True):
    """Freeze the backbone, wrap `targets` of every block in LoRA (alpha / rank = 2, as the reference), then
    unfreeze the UNFROZEN tensors if `unfreeze`."""
    backbone.requires_grad_(False)
    for blk in backbone.blocks:
        for target in targets:
            parent, name = target.split('.')
            module = getattr(blk, parent)
            setattr(module, name, LoRA(getattr(module, name), rank, alpha))
    if unfreeze:
        for name, p in backbone.named_parameters():
            if UNFROZEN.search(name) and '.base.' not in name:
                p.requires_grad_(True)


def is_adapter(name):
    """LoRA / MoE-LoRA weights and routers (new, trained at --lr), as opposed to unfrozen pretrained tensors
    (--pretrained-lr)."""
    return name.endswith(('.down.weight', '.up.weight')) or '.routers.' in name


def conv_gn(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(8, cout), nn.GELU())


class DetailBranch(nn.Module):
    """Light conv stem on the image (after ViT-Adapter's spatial prior module, much lighter): [B, 1, H, W] ->
    maps at stride 4 (`dims[0]` channels) and stride 8 (`dims[1]`), matching SimpleFeaturePyramid's two finest
    rescaled maps. The output projections start at zero, so the model starts as the reference model."""

    def __init__(self, dims=(192, 384), widths=(32, 64, 128)):
        super().__init__()
        w1, w2, w3 = widths
        self.stride4 = nn.Sequential(conv_gn(1, w1, 2), conv_gn(w1, w2, 2), conv_gn(w2, w2))
        self.stride8 = nn.Sequential(conv_gn(w2, w3, 2), conv_gn(w3, w3))
        self.out = nn.ModuleList([nn.Conv2d(w2, dims[0], 1), nn.Conv2d(w3, dims[1], 1)])
        for conv in self.out:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, images):
        c4 = self.stride4(images)
        c8 = self.stride8(c4)
        return [self.out[0](c4), self.out[1](c8)]


class DetailFeaturePyramid(SimpleFeaturePyramid):
    """SimpleFeaturePyramid whose finest levels add the detail maps before their projection."""

    def forward(self, x, details=()):
        levels = []
        for k, (rescale, project) in enumerate(zip(self.rescale, self.project)):
            y = rescale(x)
            if k < len(details):
                y = y + details[k]
            levels.append(project(y))
        levels.append(F.max_pool2d(levels[-1], 1, stride=2))
        return OrderedDict(zip(['0', '1', '2', '3', 'pool'], levels))


class ClassGuidedRoIHeads(RoIHeads):
    """RoIHeads whose inference scores use the image class (training is unchanged). `image_probs` [B, C] is the
    classification head's softmax of the same forward pass (set by add_class_guidance). For a box with detector probabilities p (background
    p_0, classes p_c) and image probabilities q:
      'none': torchvision's score p_c;
      'soft': score_c = (1 - p_0) * p_c q_c / sum_k p_k q_k   (product of the two experts on the class);
      'hard': score = 1 - p_0, for the image's argmax class only.
    Boxes are the class-specific regressions, as in torchvision; thresholds, NMS and the cap are unchanged."""

    fusion = 'soft'
    image_probs = None

    @classmethod
    def from_heads(cls, heads):
        """The same modules and hyperparameters as an existing RoIHeads."""
        return cls(heads.box_roi_pool, heads.box_head, heads.box_predictor,
                   heads.proposal_matcher.high_threshold, heads.proposal_matcher.low_threshold,
                   heads.fg_bg_sampler.batch_size_per_image, heads.fg_bg_sampler.positive_fraction,
                   heads.box_coder.weights, heads.score_thresh, heads.nms_thresh, heads.detections_per_img)

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        if self.fusion == 'none' or self.image_probs is None:
            return super().postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_classes = class_logits.shape[-1]
        boxes_per_image = [len(p) for p in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        probs = F.softmax(class_logits, -1)
        all_boxes, all_scores, all_labels = [], [], []
        for boxes, p, q, shape in zip(pred_boxes.split(boxes_per_image), probs.split(boxes_per_image),
                                      self.image_probs.float(), image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, shape)[:, 1:]  # drop the background's box
            objectness, p = 1 - p[:, 0], p[:, 1:]
            if self.fusion == 'soft':
                joint = p * q
                scores = objectness[:, None] * joint / joint.sum(1, keepdim=True).clamp(min=1e-12)
            elif self.fusion == 'hard':
                scores = objectness[:, None] * F.one_hot(q.argmax(), num_classes - 1).to(p.dtype)
            else:
                raise ValueError(f'unknown fusion {self.fusion!r}')
            labels = torch.arange(1, num_classes, device=p.device).expand_as(scores)
            boxes, scores, labels = boxes.reshape(-1, 4), scores.reshape(-1), labels.reshape(-1)
            keep = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)[:self.detections_per_img]
            all_boxes.append(boxes[keep])
            all_scores.append(scores[keep])
            all_labels.append(labels[keep])
        return all_boxes, all_scores, all_labels


def add_class_guidance(model, fusion='soft'):
    """Swap model.roi_heads for ClassGuidedRoIHeads (same modules, so the same state_dict keys) and feed it the
    softmax of model.cls_head, which every forward pass runs before the detector. Works on sarbench's
    MultiTaskModel too (so trained reference runs can be re-scored), MoE-LoRA included."""
    model.roi_heads = heads = ClassGuidedRoIHeads.from_heads(model.roi_heads)
    heads.fusion = fusion
    model.cls_head.register_forward_hook(
        lambda module, inputs, logits: setattr(heads, 'image_probs', logits.detach().float().softmax(-1)))
    return model


class DetailMultiTaskModel(MultiTaskModel):
    """MultiTaskModel + optional detail branch, small anchors and class-guided detection. task_routing: MoE-LoRA's
    task routers, the backbone run once per task, as the reference."""

    def __init__(self, backbone, detail=True, small_anchors=True, fusion='soft', task_routing=False, num_classes=9):
        super().__init__(backbone, task_routing=task_routing, num_classes=num_classes)
        self.detail = DetailBranch((backbone.embed_dim // 4, backbone.embed_dim // 2)) if detail else None
        if detail:
            self.neck = DetailFeaturePyramid(backbone.embed_dim)
        if small_anchors:
            self.rpn.anchor_generator = AnchorGenerator(SMALL_ANCHORS, ((0.5, 1.0, 2.0),) * len(SMALL_ANCHORS))
            self.rpn.head = RPNHead(256, self.rpn.anchor_generator.num_anchors_per_location()[0])
        add_class_guidance(self, fusion)

    def forward(self, images, targets=None):
        """As MultiTaskModel.forward: (logits, detections (eval), Faster R-CNN losses (training))."""
        set_task(self.backbone, 0)  # 0 = classification, 1 = detection; a no-op without MoE-LoRA
        tokens = self.backbone(images)
        logits = self.cls_head(tokens.mean(dim=1))
        if self.task_routing:
            set_task(self.backbone, 1)
            tokens = self.backbone(images)
        h, w = images.shape[-2] // 16, images.shape[-1] // 16
        tokens = tokens.transpose(1, 2).unflatten(2, (h, w))
        if self.detail is not None:
            features = self.neck(tokens, self.detail(images))
        else:
            features = self.neck(tokens)
        # Precision rule of the reference: RPN and RoIHeads in fp32
        with torch.autocast(images.device.type, enabled=False):
            features = {name: f.float() for name, f in features.items()}
            image_list = ImageList(images, [tuple(images.shape[-2:])] * len(images))
            proposals, rpn_losses = self.rpn(image_list, features, targets)
            detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
        return logits, detections, {**rpn_losses, **roi_losses}


def build_model(backbone_name='vit', pretrained=True, rank=32, dense=True, unfreeze=True, detail=True,
                small_anchors=True, fusion='soft', adapt='dense'):
    """adapt='dense': the DetailLoRA model (dense=False puts LoRA on qkv/proj only, as the reference).
    adapt='moelora': the reference's MoE-LoRA, unchanged (rank, dense and unfreeze do not apply), with task routing."""
    from sarbench.backbones import build_backbone
    backbone = build_backbone(backbone_name, pretrained=pretrained)
    if adapt == 'moelora':
        add_adapters(backbone, 'moelora')  # also freezes the backbone
    elif adapt == 'dense':
        targets = LORA_TARGETS if dense else LORA_TARGETS[:2]
        add_dense_lora(backbone, rank=rank, alpha=2 * rank, targets=targets, unfreeze=unfreeze)
    else:
        raise ValueError(f'unknown adapt {adapt!r}')
    return DetailMultiTaskModel(backbone, detail=detail, small_anchors=small_anchors, fusion=fusion,
                                task_routing=adapt == 'moelora')


def count_parameters(model):
    """Total, trainable, and the trainable split by part, in parameters."""
    parts = {'adapters': 0, 'unfrozen_pretrained': 0, 'detail_branch': 0, 'heads': 0}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('backbone.'):
            parts['adapters' if is_adapter(name) else 'unfrozen_pretrained'] += p.numel()
        elif name.startswith('detail.'):
            parts['detail_branch'] += p.numel()
        else:
            parts['heads'] += p.numel()
    total = sum(p.numel() for p in model.parameters())
    return {'total': total, 'trainable': sum(parts.values()), **parts}


def check_budget(counts):
    """The competition's parameter limits; raises if a model is over."""
    assert counts['total'] <= MAX_TOTAL, f'{counts["total"]:,} total parameters > {MAX_TOTAL:,}'
    assert counts['trainable'] <= MAX_TRAINABLE, f'{counts["trainable"]:,} trainable parameters > {MAX_TRAINABLE:,}'
