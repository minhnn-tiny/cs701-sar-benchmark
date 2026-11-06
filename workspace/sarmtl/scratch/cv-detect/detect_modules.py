"""Prototype (spec) of cv-detect's two candidates, on top of the unchanged sarbench/ (subclasses only).

D1  JointDecodeRoIHeads: at inference the two heads are decoded jointly from the one forward pass (all boxes of an image
    have the image's class): (D1b, lam) the image class also weighs the detector's top box score per class, then
    (D1, alpha) each RoI's class distribution is re-weighted by the image class probabilities at fixed objectness;
    torchvision's own postprocess (threshold, per-class NMS, top 100) then runs unchanged.
D2  SpatialPrior + FusedPyramid: a light conv branch on the input pixels whose stride-4 and stride-8 maps are added
    (1x1 conv + GN lateral) inside the ViTDet neck's stride-4 / stride-8 projections, before their 3x3 conv.

DetectModel(MultiTaskModel) wires both in; with spatial_prior=False and prior_alpha = evidence_lambda = 0 it is exactly
the reference (same modules, same state_dict keys, same forward), so reference checkpoints load into it for D1.
"""
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads

ROOT = Path(__file__).resolve().parents[4]  # the repo root, for sarbench/
sys.path.insert(0, str(ROOT))
from sarbench.adapters import set_task  # noqa: E402
from sarbench.model import MultiTaskModel, SimpleFeaturePyramid  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------
# D1: joint decoding of the two heads (inference only, no parameters)
# ---------------------------------------------------------------------------------------------------------------
class JointDecodeRoIHeads(RoIHeads):
    """RoIHeads whose eval-mode postprocess decodes the two heads jointly, given the image log-probabilities log p
    [B, 9] of the classification head:
      1. D1b (lam > 0): log p' = log_softmax(log p + lam * m), m_c = the highest raw score of class c over the image's
         RoIs (= its top box of class c, in [0, 1]); the reverse step of cv-adapt's XFuse. p' is the class output.
      2. D1 (alpha > 0), with the RoI softmax scores s_bg, s_1..s_9:
         mode 'product' (the a priori rule): score'_c = s_c * p'_c^alpha. Only the image's classes are annotated, so
             P(a box is a true positive of class c) ~ P(it is a class-c object) * P(the image is class c). Scores
             never go up and no mass moves to another class, so it cannot create a false positive; the mass it
             removes goes to the background.
         mode 'posterior' (analysis only): score'_c = (1 - s_bg) * s_c p'_c^alpha / sum_k s_k p'_k^alpha, i.e. the
             objectness is kept and the class mass moves towards the image class. Better for a true object the RoI
             head mislabelled, worse for a real object of another class (e.g. an unannotated ship in a harbor scene),
             which it can relabel as the image class.
         alpha = inf: p'^alpha = 1 for the arg-max class, 0 for the others.
      Both are written as new logits for torchvision's own postprocess (score threshold, per-class NMS, top 100),
      which then runs unchanged. alpha = lam = 0 is the reference."""
    decode = None  # dict(logp, alpha, lam, mode), set by DetectModel.forward around the call
    image_logp = None  # output of the call: log p' [B, 9]
    capture = None  # a list: the raw RoI outputs of every image are appended to it (d1_analysis.py caches them)

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        if self.capture is not None:
            counts = [len(p) for p in proposals]
            self.capture += [dict(proposals=p.cpu(), class_logits=c.cpu(), box_regression=r.cpu())
                             for p, c, r in zip(proposals, class_logits.split(counts), box_regression.split(counts))]
        if self.decode is not None:
            counts = [len(p) for p in proposals]
            logp, alpha, lam = self.decode['logp'], self.decode['alpha'], self.decode['lam']
            mode = self.decode.get('mode', 'product')
            if lam > 0:  # D1b: the detector's evidence per class
                scores = class_logits.softmax(dim=-1)[:, 1:]
                m = torch.stack([s.max(dim=0).values if len(s) else s.new_zeros(s.shape[1])
                                 for s in scores.split(counts)])
                logp = (logp + lam * m).log_softmax(dim=-1)
            self.image_logp = logp
            if alpha > 0:  # D1: the image class prior on every RoI
                if alpha == float('inf'):
                    prior = torch.where(logp == logp.max(dim=1, keepdim=True).values, 0.0, -1e4)
                else:
                    prior = alpha * logp
                prior = prior.to(class_logits.dtype).repeat_interleave(
                    torch.tensor(counts, device=class_logits.device), dim=0)  # [R, 9] = log p'^alpha
                if mode == 'product':  # q_c = s_c p_c^alpha, q_bg = s_bg + sum_c s_c (1 - p_c^alpha); logits log q
                    logs = class_logits.log_softmax(dim=-1)
                    fg = logs[:, 1:] + prior
                    bg = (logs[:, :1].exp() + (logs[:, 1:].exp() * (1 - prior.exp())).sum(1, keepdim=True)).log()
                    class_logits = torch.cat([bg, fg], dim=1)
                elif mode == 'posterior':  # same foreground mass, moved towards the image class
                    fg = class_logits[:, 1:]
                    moved = fg + prior
                    moved = moved - moved.logsumexp(dim=1, keepdim=True) + fg.logsumexp(dim=1, keepdim=True)
                    class_logits = torch.cat([class_logits[:, :1], moved], dim=1)
                else:
                    raise ValueError(mode)
        return super().postprocess_detections(class_logits, box_regression, proposals, image_shapes)


# ---------------------------------------------------------------------------------------------------------------
# D2: light conv branch on the pixels -> the stride-4 / stride-8 levels of the ViTDet neck
# ---------------------------------------------------------------------------------------------------------------
def conv_gn_relu(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(16, cout), nn.ReLU(inplace=True))


class SpatialPrior(nn.Module):
    """[B, 1, 512, 512] normalised pixels -> [B, c4, 128, 128] (stride 4, RF 23 px), [B, c8, 64, 64] (stride 8, RF 47 px).
    6 convs, no pretrained weights (trained from scratch on train with the heads)."""

    def __init__(self, c4=64, c8=128, c2=32):
        super().__init__()
        self.to_s4 = nn.Sequential(conv_gn_relu(1, c2, 2), conv_gn_relu(c2, c4, 2), conv_gn_relu(c4, c4),
                                   conv_gn_relu(c4, c4))
        self.to_s8 = nn.Sequential(conv_gn_relu(c4, c8, 2), conv_gn_relu(c8, c8))

    def forward(self, images):
        s4 = self.to_s4(images)
        return [s4, self.to_s8(s4)]


class FusedPyramid(nn.Module):
    """The reference neck (its own rescale / project modules: same init, same state_dict keys) + SpatialPrior.
    Level k in (stride 4, stride 8): P_k = GN(Conv3x3( GN(Conv1x1(rescale_k(vit))) + GN_k(Conv1x1_k(prior_k)) )),
    an FPN-style lateral added before the level's 3x3 conv; strides 16, 32, 64 as in the reference.
    - Common random numbers: D2's new modules are initialised from their own seed inside fork_rng, so the global RNG
      (hence every shared module, the data order and the samplers) is exactly as in the run without D2.
    - The lateral GN weight starts at 0: D2 starts as the reference function, and the branch is learned from there."""

    def __init__(self, base: SimpleFeaturePyramid, channels=256, prior_channels=(64, 128), seed=None):
        super().__init__()
        self.rescale, self.project = base.rescale, base.project
        with torch.random.fork_rng(devices=[]):  # CPU construction; the global RNG state is restored on exit
            if seed is not None:
                torch.default_generator.manual_seed(seed)
            self.prior = SpatialPrior(*prior_channels)
            self.lateral = nn.ModuleList(nn.Sequential(nn.Conv2d(c, channels, 1, bias=False),
                                                       nn.GroupNorm(32, channels)) for c in prior_channels)
        for lateral in self.lateral:
            nn.init.zeros_(lateral[1].weight)  # GN gamma = 0 (beta is 0 already): the lateral adds exactly 0 at init

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


# ---------------------------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------------------------
class DetectModel(MultiTaskModel):
    """MultiTaskModel + optional D2 (spatial_prior) + optional D1 joint decoding (eval mode only: prior_alpha,
    evidence_lambda; with evidence_lambda > 0 the returned classification logits are the fused log p').
    forward() is the reference forward with two changes, marked; the precision rule is unchanged."""

    def __init__(self, backbone, task_routing: bool, num_classes=9, spatial_prior=False, prior_channels=(64, 128),
                 prior_seed=None, prior_alpha=0.0, evidence_lambda=0.0, prior_mode='product', score_thresh=None,
                 nms_thresh=None):
        super().__init__(backbone, task_routing, num_classes)
        self.spatial_prior = spatial_prior
        if spatial_prior:  # wraps the neck built above: no second SimpleFeaturePyramid, no extra global RNG draws
            self.neck = FusedPyramid(self.neck, prior_channels=prior_channels, seed=prior_seed)
        self.roi_heads.__class__ = JointDecodeRoIHeads  # same instance, same hyper-parameters and keys; adds D1 only
        # inference settings: they can be changed on a trained model
        self.prior_alpha, self.evidence_lambda, self.prior_mode = prior_alpha, evidence_lambda, prior_mode
        if score_thresh is not None:
            self.roi_heads.score_thresh = score_thresh  # reference 0.05
        if nms_thresh is not None:
            self.roi_heads.nms_thresh = nms_thresh  # reference 0.5

    def forward(self, images, targets=None):
        set_task(self.backbone, 0)
        tokens = self.backbone(images)
        logits = self.cls_head(tokens.mean(dim=1))
        if self.task_routing:
            set_task(self.backbone, 1)
            tokens = self.backbone(images)
        h, w = images.shape[-2] // 16, images.shape[-1] // 16
        grid = tokens.transpose(1, 2).unflatten(2, (h, w))
        features = self.neck(grid, images) if self.spatial_prior else self.neck(grid)  # change 1 (D2)
        with torch.autocast(images.device.type, enabled=False):
            features = {name: f.float() for name, f in features.items()}
            image_list = ImageList(images, [tuple(images.shape[-2:])] * len(images))
            proposals, rpn_losses = self.rpn(image_list, features, targets)
            decode = not self.training and (self.prior_alpha > 0 or self.evidence_lambda > 0)  # change 2 (D1)
            if decode:
                self.roi_heads.decode = dict(logp=logits.float().log_softmax(dim=-1), alpha=self.prior_alpha,
                                             lam=self.evidence_lambda, mode=self.prior_mode)
            try:
                detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
                if decode:
                    logits = self.roi_heads.image_logp  # log p': same arg-max as the logits when lam = 0
            finally:
                self.roi_heads.decode = self.roi_heads.image_logp = None
        return logits, detections, {**rpn_losses, **roi_losses}
