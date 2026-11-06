"""Pinned copy of workspace/fitdet/fitdet_model.py (26 Oct 2026), the base of convnext_model.py, so that later changes to
FitDet cannot change these runs. Unchanged below this paragraph.

FitDet's model: R50-FPN-MT (workspace/r50fpn/r50fpn_model.py, copied here so that changes to that design cannot change
FitDet's runs) with two training-time options:
- nwd_rpn=C: the RPN matches anchors to ground truth by max(IoU, NWD) instead of IoU. NWD = exp(-W2 / C), W2 the
  2-Wasserstein distance between the boxes as Gaussians (Wang et al., "A Normalized Gaussian Wasserstein Distance for
  Tiny Object Detection", arXiv 2110.13389). An 8 px box cannot reach IoU 0.7 with any anchor, so under IoU it gets only
  its single best anchor; NWD gives it every anchor within ~C ln(1/0.7) px. The max keeps IoU for large boxes, where NWD
  with a small C is low. The RoI heads keep IoU (the paper's best setting is RPN only; NWD >= 0.5 would accept
  proposals ~C ln 2 px away from a tiny box, outside it).
- unfreeze_stem: conv1 and layer1 train too (r50fpn freezes them, torchvision's fine-tuning default).
Neither changes the inference graph, so a FitDet checkpoint predicts exactly like an R50-FPN-MT one.
- fast (speed, 1.63x per training step on an L40S, data/profile_step.py): the RPN head and the box heads compute in bf16
  and return fp32, so proposals and detections are still decoded in fp32 (sarbench's precision rule; R50-FPN-MT runs
  these heads in fp32, about half of a step's GPU time), and the convolution weights are channels_last.
Everything else is R50-FPN-MT: one shared ResNet-50 + FPN (COCO, torchvision fasterrcnn_resnet50_fpn_v2) feeding the
classification head (C5 average pool -> LayerNorm -> Linear) and v2's Faster R-CNN with sarbench's small anchors; fc6
frozen and adapted with sarbench's LoRA; JD decoding from workspace/sarmtl (a finished design)."""
import copy
import os
from functools import partial

import torch
from torch import nn
from torch.utils._pytree import tree_leaves, tree_map
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.ops import box_iou, clip_boxes_to_image

from sarbench.adapters import LoRA
from sarmtl_model import JointDecodeRoIHeads

# terratorch (imported through sarbench) sets TORCH_CUDNN_V8_API_DISABLED=1 for its Clay backbones. PyTorch reads it at
# the first convolution, and without cuDNN's v8 API every bf16 convolution runs on the slow im2col path.
os.environ.pop('TORCH_CUDNN_V8_API_DISABLED', None)

SMALL_ANCHORS = ((8, 11), (16, 23), (32, 45), (64, 91), (128, 181))  # sarbench/model.py's, two scales per octave
STEM = ('body.conv1.', 'body.bn1.', 'body.layer1.')
FC6 = ('roi_heads.box_head.5.base.',)  # 5: fc6, wrapped by LoRA
REFERENCE_DECODING = dict(jd_alpha=0.0, mode='product', score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)
# Cascade R-CNN (Cai & Vasconcelos, CVPR 2018), mmdetection's standard settings: per stage the IoU threshold, the box
# coder weights (1 / target std) and the loss weight; stage 1 is the R50-FPN-MT head as it is.
CASCADE = ((0.5, (10., 10., 5., 5.), 1.0), (0.6, (20., 20., 10., 10.), 0.5), (0.7, (30., 30., 15., 15.), 0.25))
CASCADE_LORA_RANK = 32  # stages 2-3: own LoRA on the shared frozen fc6; 32 keeps --unfreeze-stem within 40 M


def nwd_max(boxes1, boxes2, c):
    """max(IoU, NWD) of XYXY boxes [N, 4] x [M, 4] -> [N, M]. As Gaussians N((cx, cy), diag(w^2/4, h^2/4)), the
    2-Wasserstein distance of two boxes is the Euclidean distance of their (cx, cy, w/2, h/2) vectors."""
    def gauss(b):
        return torch.cat([(b[:, :2] + b[:, 2:]) / 2, (b[:, 2:] - b[:, :2]) / 2], dim=1)
    w2 = torch.cdist(gauss(boxes1), gauss(boxes2), compute_mode='donot_use_mm_for_euclid_dist')  # exact near 0
    return torch.maximum(box_iou(boxes1, boxes2), torch.exp(-w2 / c))


class _Bf16Forward:
    """Mixin of run_in_bf16: the forward in bf16 autocast on CUDA, its outputs back in fp32."""

    def forward(self, *args):
        device = tree_leaves(args)[0].device.type
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            out = super().forward(*args)
        return tree_map(lambda t: t.float(), out)


def run_in_bf16(module):
    """Make one module instance compute in bf16 and return fp32 (same parameters and state_dict keys)."""
    module.__class__ = type(f'Bf16{type(module).__name__}', (_Bf16Forward, type(module)), {})


class CascadeRoIHeads(JointDecodeRoIHeads):
    """R50-FPN-MT's RoI heads plus Cascade R-CNN stages 2 and 3 (`extra`: box head + class-agnostic predictor each).
    Training: every stage samples its own RoIs from the previous stage's refined boxes (plus the ground truth) at its IoU
    threshold and box coder, and adds its weighted Fast R-CNN losses. Inference: the class probabilities of the three
    stages are averaged and the boxes are the last stage's, then JointDecodeRoIHeads/torchvision's postprocess runs
    (score threshold, per-class NMS, top k, JD decoding) unchanged. FitDetModel sets `extra` (no class default: a class
    attribute would shadow the registered submodule)."""

    def _stage(self, t):
        iou, weights, loss_weight = CASCADE[t]
        head, predictor = (self.box_head, self.box_predictor) if t == 0 else self.extra[t - 1]
        return head, predictor, det_utils.Matcher(iou, iou, allow_low_quality_matches=False), \
            det_utils.BoxCoder(weights), loss_weight

    def _predict(self, features, boxes, image_shapes, head, predictor):
        logits, deltas = predictor(head(self.box_roi_pool(features, boxes, image_shapes)))
        if deltas.shape[1] == 4:  # class-agnostic: the same deltas for every class
            deltas = deltas.repeat(1, logits.shape[1])
        return logits, deltas

    @staticmethod
    def _refine(logits, deltas, boxes, coder, image_shapes, labels=None):
        """The boxes moved by their deltas: of the ground-truth class for positives in training, else of the most likely
        foreground class; decoded in fp32 and clipped."""
        cls = logits[:, 1:].argmax(dim=1) + 1
        if labels is not None:
            cls = torch.where(labels > 0, labels, cls)
        picked = deltas.reshape(len(cls), -1, 4)[torch.arange(len(cls), device=cls.device), cls]
        refined = coder.decode(picked.float(), boxes).reshape(-1, 4)
        return [clip_boxes_to_image(b, s).detach() for b, s in
                zip(refined.split([len(b) for b in boxes]), image_shapes)]

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            losses = {}
            for t in range(len(CASCADE)):
                head, predictor, matcher, coder, loss_weight = self._stage(t)
                saved = self.proposal_matcher, self.box_coder
                self.proposal_matcher, self.box_coder = matcher, coder  # torchvision's sampling, this stage's IoU
                try:
                    boxes, _, labels, regression_targets = self.select_training_samples(proposals, targets)
                finally:
                    self.proposal_matcher, self.box_coder = saved
                logits, deltas = self._predict(features, boxes, image_shapes, head, predictor)
                loss_cls, loss_box = fastrcnn_loss(logits, deltas, labels, regression_targets)
                suffix = '' if t == 0 else f'_s{t + 1}'
                losses[f'loss_classifier{suffix}'] = loss_weight * loss_cls
                losses[f'loss_box_reg{suffix}'] = loss_weight * loss_box
                if t < len(CASCADE) - 1:
                    proposals = self._refine(logits, deltas, boxes, coder, image_shapes, torch.cat(labels))
            return [], losses
        probs = []
        for t in range(len(CASCADE)):
            head, predictor, _, coder, _ = self._stage(t)
            logits, deltas = self._predict(features, proposals, image_shapes, head, predictor)
            probs.append(logits.float().softmax(dim=-1))
            proposals = self._refine(logits, deltas, proposals, coder, image_shapes)  # after the last: final boxes
        log_probs = torch.stack(probs).mean(dim=0).clamp_min(1e-12).log()
        # the final boxes are already decoded: zero deltas decode to themselves under any coder
        zeros = log_probs.new_zeros(len(log_probs), 4 * log_probs.shape[1])
        boxes, scores, labels = self.postprocess_detections(log_probs, zeros, proposals, image_shapes)
        return [{'boxes': b, 'labels': lab, 'scores': s} for b, s, lab in zip(boxes, scores, labels)], {}


class FitDetModel(nn.Module):
    """[B, 1, H, W] -> (classification logits [B, 9], per-image detections (eval mode), the Faster R-CNN losses
    (training mode)): the interface of sarbench's MultiTaskModel, so root train.py's loop and evaluate run it."""

    def __init__(self, coco=True, lora_rank=256, nwd_rpn=None, unfreeze_stem=False, cascade=False, fast=False,
                 num_classes=9):
        super().__init__()
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if coco else None  # None: same modules, random init
        det = fasterrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None)
        self.body, self.fpn = det.backbone.body, det.backbone.fpn  # C2..C5 ('0'..'3'), then P2..P6
        self.cls_head = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, num_classes))

        self.rpn = det.rpn  # COCO's two 3x3 convs; new 1x1 outputs for 6 anchors per location
        self.rpn.anchor_generator = AnchorGenerator(SMALL_ANCHORS, ((0.5, 1.0, 2.0),) * len(SMALL_ANCHORS))
        n = self.rpn.anchor_generator.num_anchors_per_location()[0]
        self.rpn.head.cls_logits, self.rpn.head.bbox_pred = nn.Conv2d(256, n, 1), nn.Conv2d(256, 4 * n, 1)
        for layer in (self.rpn.head.cls_logits, self.rpn.head.bbox_pred):  # torchvision's RPNHead init
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
        if nwd_rpn:
            self.rpn.box_similarity = partial(nwd_max, c=nwd_rpn)  # used only to match anchors in training

        self.roi_heads = det.roi_heads
        self.roi_heads.__class__ = JointDecodeRoIHeads  # same instance, hyper-parameters and keys
        self.roi_heads.box_predictor = FastRCNNPredictor(1024, num_classes + 1)  # + 1: background
        fc6 = self.roi_heads.box_head[5]  # Linear(256 * 7 * 7, 1024)
        self.roi_heads.box_head[5] = LoRA(fc6, rank=lora_rank, alpha=lora_rank)  # scale 1; B = 0: starts as COCO's
        if cascade:  # stages 2-3: COCO's four convs copied, fc6 shared (frozen) with their own LoRA, new predictors
            self.roi_heads.__class__ = CascadeRoIHeads
            stages = []
            for _ in CASCADE[1:]:
                head = nn.Sequential(*copy.deepcopy(list(self.roi_heads.box_head)[:5]),  # 4 convs + flatten
                                     LoRA(fc6, rank=CASCADE_LORA_RANK, alpha=CASCADE_LORA_RANK), nn.ReLU(inplace=True))
                predictor = FastRCNNPredictor(1024, num_classes + 1)
                predictor.bbox_pred = nn.Linear(1024, 4)  # class-agnostic
                stages.append(nn.ModuleList([head, predictor]))
            self.roi_heads.extra = nn.ModuleList(stages)
        if fast:
            for head in [self.rpn.head, self.roi_heads.box_head] + ([s[0] for s in self.roi_heads.extra] if cascade
                                                                   else []):
                run_in_bf16(head)
            self.to(memory_format=torch.channels_last)  # 4-D weights; activations follow them

        frozen = FC6 if unfreeze_stem else STEM + FC6
        for name, p in self.named_parameters():
            p.requires_grad_(not name.startswith(frozen))
        self.jd_alpha, self.jd_mode = 0.0, 'product'

    def forward(self, images, targets=None):
        c = self.body(images.expand(-1, 3, -1, -1))  # the gray channel as R, G and B
        logits = self.cls_head(c['3'].mean(dim=(2, 3)))
        features = self.fpn(c)
        # sarbench's precision rule: in bf16 the box coder would snap coordinates near 512 to a 2 px grid, so the RPN
        # and the RoI heads run in fp32
        with torch.autocast(images.device.type, enabled=False):
            features = {name: f.float() for name, f in features.items()}
            image_list = ImageList(images, [tuple(images.shape[-2:])] * len(images))
            proposals, rpn_losses = self.rpn(image_list, features, targets)
            if not self.training and self.jd_alpha > 0:
                self.roi_heads.decode = dict(logp=logits.float().log_softmax(dim=-1), alpha=self.jd_alpha,
                                             mode=self.jd_mode)
            try:
                detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
            finally:
                self.roi_heads.decode = None
        return logits, detections, {**rpn_losses, **roi_losses}


def set_decoding(model, jd_alpha, mode, score_thresh, nms_thresh, detections_per_img):
    """Inference settings; they can be changed on a trained model. jd_alpha 0 is torchvision's own decoding."""
    model.jd_alpha, model.jd_mode = jd_alpha, mode
    rh = model.roi_heads
    rh.score_thresh, rh.nms_thresh, rh.detections_per_img = score_thresh, nms_thresh, detections_per_img


def build_model(coco=True, lora_rank=256, nwd_rpn=None, unfreeze_stem=False, cascade=False, fast=False):
    model = FitDetModel(coco=coco, lora_rank=lora_rank, nwd_rpn=nwd_rpn, unfreeze_stem=unfreeze_stem, cascade=cascade,
                        fast=fast)
    set_decoding(model, **REFERENCE_DECODING)
    return model


def model_from_args(args):
    """The model of a run's metrics.json args (r50fpn's runs too), before its model.pt is loaded."""
    return build_model(coco=False, lora_rank=args['lora_rank'], nwd_rpn=args.get('nwd_rpn'),
                       unfreeze_stem=args.get('unfreeze_stem', False), cascade=args.get('cascade', False),
                       fast=args.get('fast', False))


def count_parameters(model):
    """Total and trainable parameters, and the trainable ones per part."""
    parts = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            part = 'lora' if '.down.' in name or '.up.' in name else name.split('.')[0]
            parts[part] = parts.get(part, 0) + p.numel()
    return sum(p.numel() for p in model.parameters()), sum(parts.values()), parts
