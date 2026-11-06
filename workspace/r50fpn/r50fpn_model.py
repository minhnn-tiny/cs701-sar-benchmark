"""R50-FPN-MT: torchvision's Faster R-CNN ResNet-50-FPN v2, pretrained on COCO, as the benchmark's multi-task model.
- One shared backbone (ResNet-50 + FPN, with v2's BatchNorm) feeds a classification head on the last ResNet stage
  (C5 global average pool -> LayerNorm -> Linear, as the reference head on the ViT tokens) and the Faster R-CNN detector.
- The detector is v2's (two-conv RPN head, four-conv + fc box head), with sarbench/model.py's small anchors (8 to 181
  px, 6 per location) and 9 + 1 classes. Its RPN and RoI-head hyper-parameters are torchvision's, as the reference's.
- Budget: conv1 and layer1 frozen (torchvision's fine-tuning default), and the box head's 12.8 M fc6 frozen and adapted
  with sarbench's LoRA; everything else trains.
- Inputs are sarbench/data.py's: [B, 1, 512, 512] gray, normalised with the SAR mean and std. The gray channel is
  repeated for COCO's RGB conv1. No GeneralizedRCNNTransform: nothing is resized, the baseline's geometry is kept.
- JD at inference: workspace/sarmtl's JointDecodeRoIHeads (each RoI's class scores reweighted by the classification
  softmax of the same pass), set with set_decoding together with the score threshold, NMS IoU and boxes per image.
workspace/sarmtl must be on sys.path (the scripts and tests/conftest.py put it there)."""
import torch
from torch import nn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList

from sarbench.adapters import LoRA
from sarmtl_model import JointDecodeRoIHeads

SMALL_ANCHORS = ((8, 11), (16, 23), (32, 45), (64, 91), (128, 181))  # sarbench/model.py's, two scales per octave
FROZEN = ('body.conv1.', 'body.bn1.', 'body.layer1.', 'roi_heads.box_head.5.base.')  # 5: fc6, wrapped by LoRA
REFERENCE_DECODING = dict(jd_alpha=0.0, mode='product', score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)


def decoding_grid():
    """The post-processing sweep (evaluate.py): torchvision's decoding, JD product or JD posterior, each with a low
    score threshold, 100 or 300 boxes per image and per-class NMS IoU 0.5, 0.6 or 0.7; plus the reference."""
    grid = {'reference': REFERENCE_DECODING}
    for jd, (alpha, mode) in {'off': (0.0, 'product'), 'product': (1.0, 'product'),
                              'posterior': (1.0, 'posterior')}.items():
        for boxes in (100, 300):
            for nms in (0.5, 0.6, 0.7):
                grid[f'{jd}_d{boxes}_nms{nms}'] = dict(jd_alpha=alpha, mode=mode, score_thresh=0.001, nms_thresh=nms,
                                                      detections_per_img=boxes)
    return grid


DECODINGS = decoding_grid()


class R50FPNMultiTask(nn.Module):
    """[B, 1, H, W] -> (classification logits [B, 9], per-image detections (eval mode), the Faster R-CNN losses
    (training mode)): the interface of sarbench's MultiTaskModel, so root train.py's loop and evaluate run it."""

    def __init__(self, coco=True, lora_rank=256, num_classes=9):
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

        self.roi_heads = det.roi_heads
        self.roi_heads.__class__ = JointDecodeRoIHeads  # same instance, hyper-parameters and keys
        self.roi_heads.box_predictor = FastRCNNPredictor(1024, num_classes + 1)  # + 1: background
        fc6 = self.roi_heads.box_head[5]  # Linear(256 * 7 * 7, 1024)
        self.roi_heads.box_head[5] = LoRA(fc6, rank=lora_rank, alpha=lora_rank)  # scale 1; B = 0: starts as COCO's

        for name, p in self.named_parameters():
            p.requires_grad_(not name.startswith(FROZEN))
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


def build_model(coco=True, lora_rank=256):
    """The model of a run. coco=False builds the same modules without downloading anything (to load a model.pt)."""
    model = R50FPNMultiTask(coco=coco, lora_rank=lora_rank)
    set_decoding(model, **REFERENCE_DECODING)
    return model


def model_from_args(args):
    """build_model for the args dict of a run's metrics.json, before its model.pt is loaded."""
    return build_model(coco=False, lora_rank=args['lora_rank'])


def count_parameters(model):
    """Total and trainable parameters, and the trainable ones per part."""
    parts = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            part = 'lora' if '.down.' in name or '.up.' in name else name.split('.')[0]
            parts[part] = parts.get(part, 0) + p.numel()
    return sum(p.numel() for p in model.parameters()), sum(parts.values()), parts
