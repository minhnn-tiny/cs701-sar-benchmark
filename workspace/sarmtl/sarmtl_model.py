"""WideLoRA-SP with joint decoding (docs/DESIGN.md), built on the unchanged sarbench/ (subclasses and wrappers only):
- WideLoRA: sarbench's LoRA at rank r on qkv, proj, fc1 and fc2 of every ViT block, plus the small pretrained tensors
  unfrozen (patch embedding, pos_embed, cls_token, LayerNorms, the biases of the wrapped layers);
- SP: a light conv branch on the pixels, added into the ViTDet neck's stride-4 and stride-8 levels;
- JD: at inference, each RoI's class score s_c becomes s_c * p_c (p = the classification softmax of the same pass).
With adapt lora/moelora and sp=False the model is the reference (same modules and state_dict keys), so the reference
checkpoints of root train.py load into it."""
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads

from sarbench.adapters import LoRA, add_adapters, set_task
from sarbench.backbones import build_backbone
from sarbench.model import MultiTaskModel, SimpleFeaturePyramid

TARGETS = ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')  # the linear layers of a timm ViT block
REFERENCE_DECODING = dict(jd_alpha=0.0, score_thresh=0.05)  # sarbench/model.py's RoIHeads settings
JD_DECODING = dict(jd_alpha=1.0, score_thresh=0.001)  # the pre-registered joint decoding (DESIGN.md §1)


def add_wide_lora(backbone, rank=64):
    """Freeze the ViT, wrap TARGETS of every block with LoRA(rank, alpha = 8 sqrt(rank)), and unfreeze every other
    pretrained tensor except the wrapped weights. The scale alpha / rank = 8 / sqrt(rank) is rank-stable: 2 at rank 16
    (the reference), 1 at rank 64. B = 0 at init, so training starts from the pretrained model."""
    backbone.requires_grad_(False)
    for blk in backbone.blocks:
        for name in TARGETS:
            parent, leaf = name.split('.')
            module = getattr(blk, parent)
            setattr(module, leaf, LoRA(getattr(module, leaf), rank=rank, alpha=8 * rank ** 0.5))
    for name, p in backbone.named_parameters():
        if '.down.' not in name and '.up.' not in name and not name.endswith('.base.weight'):
            p.requires_grad_(True)  # patch embedding, pos_embed, cls_token, LayerNorms, biases


def param_groups(model, lr, pretrained_lr):
    """AdamW groups: LoRA at lr, heads (everything outside the backbone, SP included) at lr, the unfrozen pretrained
    tensors at pretrained_lr. The heads are group 1, whose lr root train.py's train_one_epoch logs."""
    groups = {'lora': [], 'heads': [], 'pretrained': []}
    for name, p in model.named_parameters():
        if p.requires_grad:
            kind = 'heads' if not name.startswith('backbone.') else \
                'lora' if '.down.' in name or '.up.' in name else 'pretrained'
            groups[kind].append(p)
    return [{'params': groups['lora'], 'lr': lr}, {'params': groups['heads'], 'lr': lr},
            {'params': groups['pretrained'], 'lr': pretrained_lr}]


class JointDecodeRoIHeads(RoIHeads):
    """RoIHeads whose eval-mode postprocess can decode the two heads jointly (cv-detect's D1). Given the image
    log-probabilities log p [B, 9] of the classification head and RoI softmax scores s_bg, s_1..s_9:
      mode 'product' (the design): q_c = s_c p_c^alpha, q_bg = s_bg + sum_c s_c (1 - p_c^alpha). Scores never go up,
          so it cannot create a false positive; the mass it removes goes to the background.
      mode 'posterior' (only through the 2-fold cross-fit): q_c = (1 - s_bg) s_c p_c^alpha / sum_k s_k p_k^alpha, the
          objectness kept and the class mass moved towards the image class.
    Both become new logits for torchvision's own postprocess (score threshold, per-class NMS, top 100)."""
    decode = None  # dict(logp, alpha, mode), set by SARMTLModel.forward around the call

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        if self.decode is not None:
            counts = torch.tensor([len(p) for p in proposals], device=class_logits.device)
            prior = (self.decode['alpha'] * self.decode['logp']).repeat_interleave(counts, dim=0)  # log p^alpha [R, 9]
            if self.decode['mode'] == 'product':
                logs = class_logits.log_softmax(dim=-1)
                bg = (logs[:, :1].exp() + (logs[:, 1:].exp() * (1 - prior.exp())).sum(1, keepdim=True)).log()
                class_logits = torch.cat([bg, logs[:, 1:] + prior], dim=1)
            elif self.decode['mode'] == 'posterior':
                fg = class_logits[:, 1:]
                moved = fg + prior
                moved = moved - moved.logsumexp(dim=1, keepdim=True) + fg.logsumexp(dim=1, keepdim=True)
                class_logits = torch.cat([class_logits[:, :1], moved], dim=1)
            else:
                raise ValueError(self.decode['mode'])
        return super().postprocess_detections(class_logits, box_regression, proposals, image_shapes)


def conv_gn_relu(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(16, cout), nn.ReLU(inplace=True))


class SpatialPrior(nn.Module):
    """[B, 1, 512, 512] normalised pixels -> [B, c4, 128, 128] (stride 4) and [B, c8, 64, 64] (stride 8): 6 convs,
    trained from scratch with the heads."""

    def __init__(self, c4=64, c8=128, c2=32):
        super().__init__()
        self.to_s4 = nn.Sequential(conv_gn_relu(1, c2, 2), conv_gn_relu(c2, c4, 2), conv_gn_relu(c4, c4),
                                   conv_gn_relu(c4, c4))
        self.to_s8 = nn.Sequential(conv_gn_relu(c4, c8, 2), conv_gn_relu(c8, c8))

    def forward(self, images):
        s4 = self.to_s4(images)
        return [s4, self.to_s8(s4)]


class FusedPyramid(nn.Module):
    """The reference neck (its own rescale and project modules: same init, same state_dict keys) + SpatialPrior. At
    strides 4 and 8: P_k = GN(Conv3x3( GN(Conv1x1(rescale_k(vit))) + GN(Conv1x1(prior_k)) )), an FPN-style lateral added
    before the level's 3x3 conv. The new modules are built from their own seed inside fork_rng, so the global RNG (every
    shared module's init, the data order, the flips) is exactly as in the run without SP; the laterals' GN weight
    starts at 0, so the model starts as the one without SP."""

    def __init__(self, base: SimpleFeaturePyramid, seed, channels=256, prior_channels=(64, 128)):
        super().__init__()
        self.rescale, self.project = base.rescale, base.project
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.prior = SpatialPrior(*prior_channels)
            self.lateral = nn.ModuleList(nn.Sequential(nn.Conv2d(c, channels, 1, bias=False),
                                                       nn.GroupNorm(32, channels)) for c in prior_channels)
        for lateral in self.lateral:
            nn.init.zeros_(lateral[1].weight)

    def forward(self, x, images):
        priors = self.prior(images)
        levels = []
        for k, (rescale, project) in enumerate(zip(self.rescale, self.project)):
            y = project[1](project[0](rescale(x)))  # 1x1 conv + GN, as in the reference
            if k < len(priors):
                y = y + self.lateral[k](priors[k])
            levels.append(project[3](project[2](y)))  # 3x3 conv + GN, as in the reference
        levels.append(F.max_pool2d(levels[-1], 1, stride=2))
        return OrderedDict(zip(['0', '1', '2', '3', 'pool'], levels))


class SARMTLModel(MultiTaskModel):
    """MultiTaskModel + optional SP + JD (eval mode only). forward() is the reference forward with two changes, marked;
    the precision rule (RPN and RoI heads in fp32) is unchanged. Decoding is set with set_decoding()."""

    def __init__(self, backbone, task_routing: bool, sp=False, sp_seed=0, num_classes=9):
        super().__init__(backbone, task_routing, num_classes)
        self.sp = sp
        if sp:  # wraps the neck built above: no second SimpleFeaturePyramid, no extra global RNG draws
            self.neck = FusedPyramid(self.neck, seed=sp_seed)
        self.roi_heads.__class__ = JointDecodeRoIHeads  # same instance, hyper-parameters and keys
        self.jd_alpha, self.jd_mode = 0.0, 'product'

    def forward(self, images, targets=None):
        set_task(self.backbone, 0)
        tokens = self.backbone(images)
        logits = self.cls_head(tokens.mean(dim=1))
        if self.task_routing:
            set_task(self.backbone, 1)
            tokens = self.backbone(images)
        h, w = images.shape[-2] // 16, images.shape[-1] // 16
        grid = tokens.transpose(1, 2).unflatten(2, (h, w))
        features = self.neck(grid, images) if self.sp else self.neck(grid)  # change 1: SP
        with torch.autocast(images.device.type, enabled=False):
            features = {name: f.float() for name, f in features.items()}
            image_list = ImageList(images, [tuple(images.shape[-2:])] * len(images))
            proposals, rpn_losses = self.rpn(image_list, features, targets)
            if not self.training and self.jd_alpha > 0:  # change 2: JD
                self.roi_heads.decode = dict(logp=logits.float().log_softmax(dim=-1), alpha=self.jd_alpha,
                                             mode=self.jd_mode)
            try:
                detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
            finally:
                self.roi_heads.decode = None
        return logits, detections, {**rpn_losses, **roi_losses}


def set_decoding(model, jd_alpha, score_thresh, mode='product'):
    """Inference settings; they can be changed on a trained model. REFERENCE_DECODING and JD_DECODING are the two
    pre-registered ones."""
    model.jd_alpha, model.jd_mode, model.roi_heads.score_thresh = jd_alpha, mode, score_thresh


def build_model(backbone='vit', init='pretrained', adapt='widelora', rank=64, sp=True, seed=0):
    """The model of a run, from its arguments (train.py's, or root train.py's, which have no rank or sp). Adapters are
    created on the CPU, so this runs before .cuda()."""
    bb = build_backbone(backbone, pretrained=init == 'pretrained')
    if adapt == 'widelora':
        assert backbone == 'vit', 'WideLoRA targets the timm ViT blocks'
        add_wide_lora(bb, rank)
    elif adapt != 'full':
        add_adapters(bb, adapt)
    return SARMTLModel(bb, task_routing=adapt == 'moelora', sp=sp, sp_seed=1000 + seed)


def model_from_args(args):
    """build_model for the args dict of a metrics.json (train.py or root train.py)."""
    return build_model(args['backbone'], 'scratch', args['adapt'], args.get('rank', 64), args.get('sp', False),
                       args.get('seed', 0))


def count_parameters(model):
    """Total and trainable parameters, and the trainable ones split by kind."""
    kinds = {'heads': 0, 'sp': 0, 'lora': 0, 'pretrained': 0}
    for name, p in model.named_parameters():
        if p.requires_grad:
            kind = 'sp' if name.startswith(('neck.prior.', 'neck.lateral.')) else \
                'heads' if not name.startswith('backbone.') else \
                'lora' if '.down.' in name or '.up.' in name else 'pretrained'
            kinds[kind] += p.numel()
    total = sum(p.numel() for p in model.parameters())
    return total, sum(kinds.values()), kinds
