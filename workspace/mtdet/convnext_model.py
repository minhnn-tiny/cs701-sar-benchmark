"""ConvNeXt-B + FPN + Cascade R-CNN as one multi-task model: FitDet's model (frcnn_base.py, R50-FPN-MT with Cascade
R-CNN stages 2-3, bf16 heads) with its ResNet-50 replaced by ConvNeXt-B.
- Backbone: timm `convnext_base.fb_in22k_ft_in1k` (ImageNet-22k, fine-tuned on ImageNet-1k; Liu et al., CVPR 2022),
  C2-C5 = strides 4-32, 128-1024 channels. Input: sarbench's normalised gray as R, G and B.
- FPN: new 1x1 laterals for ConvNeXt's widths; COCO's 3x3 output convs (from torchvision's R50-FPN v2). RPN, the box
  heads and the cascade are FitDet's (COCO-initialised where shapes allow), JD decoding included.
- Classification: C5 average pool -> LayerNorm -> Linear (as the reference head).
- Budget (<= 40 M trainable with the heads): the stem, stages 0-1 and the last TRAIN_BLOCKS blocks of stage 2 train;
  stage 2's other blocks are frozen; stage 3 is frozen and adapted with sarbench's LoRA (rank STAGE3_LORA) on its MLPs."""
import timm
import torch
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

from frcnn_base import REFERENCE_DECODING, FitDetModel, set_decoding
from sarbench.adapters import LoRA

TIMM_NAME = 'convnext_base.fb_in22k_ft_in1k'
TRAIN_BLOCKS = 9  # of stage 2's 27
STAGE3_LORA = 32


class ConvNeXtBody(nn.Module):
    """[B, 3, H, W] -> {'0': C2, '1': C3, '2': C4, '3': C5}, the dict FitDet's FPN and classification head read."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.net = timm.create_model(TIMM_NAME, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))

    def forward(self, x):
        return dict(zip('0123', self.net(x)))


def decoding_grid():
    """r50fpn's sweep: the reference decoding, then torchvision's / JD product / JD posterior, each at score threshold
    0.001 with 100 or 300 boxes per image and per-class NMS IoU 0.5, 0.6 or 0.7."""
    grid = {'reference': REFERENCE_DECODING}
    for jd, (alpha, mode) in {'off': (0.0, 'product'), 'product': (1.0, 'product'),
                              'posterior': (1.0, 'posterior')}.items():
        for boxes in (100, 300):
            for nms in (0.5, 0.6, 0.7):
                grid[f'{jd}_d{boxes}_nms{nms}'] = dict(jd_alpha=alpha, mode=mode, score_thresh=0.001, nms_thresh=nms,
                                                      detections_per_img=boxes)
    return grid


class ConvNeXtMultiTask(FitDetModel):
    def __init__(self, coco=True, cascade=True, fast=True, num_classes=9):
        super().__init__(coco=coco, cascade=cascade, fast=fast, num_classes=num_classes)
        coco_fpn = self.fpn
        self.body = ConvNeXtBody(pretrained=coco)
        self.fpn = FeaturePyramidNetwork(self.body.net.feature_info.channels(), 256, extra_blocks=LastLevelMaxPool(),
                                         norm_layer=nn.BatchNorm2d)
        self.fpn.layer_blocks.load_state_dict(coco_fpn.layer_blocks.state_dict())
        self.cls_head = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, num_classes))
        net = self.body.net
        for p in net.parameters():
            p.requires_grad_(False)
        for part in (net.stem_0, net.stem_1, net.stages_0, net.stages_1, net.stages_2.blocks[-TRAIN_BLOCKS:]):
            part.requires_grad_(True)
        for block in net.stages_3.blocks:  # LoRA's down/up train, its base stays frozen
            block.mlp.fc1 = LoRA(block.mlp.fc1, rank=STAGE3_LORA, alpha=STAGE3_LORA)
            block.mlp.fc2 = LoRA(block.mlp.fc2, rank=STAGE3_LORA, alpha=STAGE3_LORA)
        if fast:
            self.to(memory_format=torch.channels_last)
        set_decoding(self, **REFERENCE_DECODING)
        self.channels_last = fast

    def is_backbone(self, name):
        return name.startswith('body.')

    def epoch_done(self):
        pass

    def replay_epochs(self, epochs):
        pass

    def decodings(self):
        return decoding_grid()

    def apply_decoding(self, cfg):
        set_decoding(self, **cfg)
