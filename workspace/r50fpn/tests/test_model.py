"""R50-FPN-MT on the CPU: the budget, what is frozen, the COCO weights, both forward modes and reloading a run."""
from pathlib import Path

import pytest
import torch

from r50fpn_model import build_model, count_parameters, model_from_args, set_decoding

COCO = Path.home() / '.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    return build_model(coco=False)


def test_budget_and_frozen_parts(model):
    total, trainable, parts = count_parameters(model)
    assert total <= 130_000_000 and trainable <= 40_000_000
    assert 33_000_000 < trainable < 34_500_000  # ~33.7 M: README.md's figure
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert 'body.conv1.weight' in frozen and 'body.layer1.0.conv1.weight' in frozen
    assert {'roi_heads.box_head.5.base.weight', 'roi_heads.box_head.5.base.bias'} <= frozen
    assert all(n.startswith(('body.conv1.', 'body.bn1.', 'body.layer1.', 'roi_heads.box_head.5.base.')) for n in frozen)
    assert parts['lora'] == 256 * (256 * 7 * 7 + 1024)
    assert model.rpn.head.cls_logits.out_channels == 6 and model.roi_heads.box_predictor.cls_score.out_features == 10


@pytest.mark.skipif(not COCO.exists(), reason='COCO weights not downloaded')
def test_coco_weights_are_loaded():
    coco = torch.load(COCO, map_location='cpu')
    m = build_model(coco=True)
    pairs = {'body.layer3.2.conv2.weight': 'backbone.body.layer3.2.conv2.weight',
             'fpn.layer_blocks.0.0.weight': 'backbone.fpn.layer_blocks.0.0.weight',
             'rpn.head.conv.0.0.weight': 'rpn.head.conv.0.0.weight',
             'roi_heads.box_head.5.base.weight': 'roi_heads.box_head.5.weight',
             'roi_heads.box_head.0.0.weight': 'roi_heads.box_head.0.0.weight'}
    state = m.state_dict()
    for ours, theirs in pairs.items():
        assert torch.equal(state[ours], coco[theirs]), ours


def test_train_and_eval_forward(model):
    images = torch.randn(2, 1, 128, 128)
    targets = [{'boxes': torch.tensor([[10., 12., 30., 40.], [60., 60., 70., 75.]]), 'labels': torch.tensor([3, 3])},
               {'boxes': torch.tensor([[5., 5., 100., 90.]]), 'labels': torch.tensor([7])}]
    model.train()
    logits, _, losses = model(images, targets)
    assert logits.shape == (2, 9)
    assert set(losses) == {'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'}
    (sum(losses.values()) + logits.logsumexp(1).sum()).backward()
    assert model.roi_heads.box_head[5].up.weight.grad.abs().sum() > 0  # the LoRA on fc6 trains
    assert model.body.conv1.weight.grad is None and model.cls_head[1].weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)

    model.eval()
    for decoding in (dict(jd_alpha=0.0, mode='product'), dict(jd_alpha=1.0, mode='posterior')):
        set_decoding(model, score_thresh=0.0, nms_thresh=0.5, detections_per_img=50, **decoding)
        with torch.no_grad():
            logits, detections, _ = model(images)
        assert len(detections) == 2 and all(len(d['boxes']) <= 50 for d in detections)
        assert all(((d['labels'] >= 1) & (d['labels'] <= 9)).all() for d in detections)
        assert model.roi_heads.decode is None  # reset after every call


def test_a_run_reloads(model):
    copy = model_from_args({'lora_rank': 256})
    copy.load_state_dict(model.state_dict())  # strict: same modules and keys
