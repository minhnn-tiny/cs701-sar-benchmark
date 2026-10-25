"""DetailLoRA: parameter budget, where the adapters go, output shapes, and the class-guided scoring (CPU only)."""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from detail_model import (MAX_TOTAL, MAX_TRAINABLE, ClassGuidedRoIHeads, DetailBranch, DetailFeaturePyramid,
                          add_class_guidance, build_model, check_budget, count_parameters)
from sarbench.adapters import LoRA, add_adapters
from sarbench.backbones import build_backbone
from sarbench.model import MultiTaskModel, SimpleFeaturePyramid


@pytest.fixture(scope='module')
def primary():
    torch.manual_seed(0)
    return build_model(pretrained=False)  # same architecture as the pretrained one


def test_parameter_counts_are_within_the_budget(primary):
    counts = count_parameters(primary)
    check_budget(counts)
    assert counts['total'] == 111_216_584 <= MAX_TOTAL
    assert counts['trainable'] == 26_198_984 <= MAX_TRAINABLE
    assert counts['adapters'] == 4_718_592 and counts['unfrozen_pretrained'] == 1_023_744
    assert counts['detail_branch'] == 339_616 and counts['heads'] == 20_117_032
    a1 = count_parameters(build_model(pretrained=False, detail=False, small_anchors=False))
    assert (a1['total'], a1['trainable']) == (110_873_113, 25_855_513)
    with pytest.raises(AssertionError):
        check_budget({'total': MAX_TOTAL, 'trainable': MAX_TRAINABLE + 1})
    with pytest.raises(AssertionError):
        check_budget({'total': MAX_TOTAL + 1, 'trainable': 0})


def test_lora_on_every_linear_and_only_small_pretrained_tensors_unfrozen(primary):
    loras = [m for m in primary.backbone.modules() if isinstance(m, LoRA)]
    assert len(loras) == 12 * 4
    assert all(not p.requires_grad for m in loras for p in m.base.parameters())  # pretrained weights stay frozen
    unfrozen = {n for n, p in primary.backbone.named_parameters() if p.requires_grad and not n.endswith(('down.weight', 'up.weight'))}
    assert {'vit.patch_embed.proj.weight', 'vit.pos_embed', 'vit.cls_token', 'vit.norm.weight'} <= unfrozen
    assert len([n for n in unfrozen if '.norm' in n]) == 2 * (12 * 2 + 1)  # weight + bias of 25 LayerNorms
    assert all('.base.' not in n for n in unfrozen)


def test_detail_pyramid_starts_as_the_reference_pyramid():
    torch.manual_seed(0)
    reference, detailed, branch = SimpleFeaturePyramid(), DetailFeaturePyramid(), DetailBranch()
    detailed.load_state_dict(reference.state_dict())
    tokens, images = torch.randn(2, 768, 8, 8), torch.randn(2, 1, 128, 128)
    details = branch(images)
    assert [d.shape for d in details] == [(2, 192, 32, 32), (2, 384, 16, 16)]  # strides 4 and 8
    for a, b in zip(reference(tokens).values(), detailed(tokens, details).values()):
        assert torch.equal(a, b)  # zero-initialised outputs: training starts from the reference model


def test_forward_shapes_losses_and_detections(primary):
    torch.manual_seed(0)
    images = torch.randn(2, 1, 512, 512)
    targets = [{'boxes': torch.tensor([[10., 10., 16., 15.], [100., 100., 160., 150.]]), 'labels': torch.tensor([9, 9])},
               {'boxes': torch.tensor([[200., 200., 206., 205.]]), 'labels': torch.tensor([3])}]
    primary.train()
    logits, _, losses = primary(images, targets)
    assert logits.shape == (2, 9)
    assert set(losses) == {'loss_objectness', 'loss_rpn_box_reg', 'loss_classifier', 'loss_box_reg'}
    assert all(torch.isfinite(v) for v in losses.values())
    primary.eval()
    for fusion in ('none', 'hard', 'soft'):
        primary.roi_heads.fusion = fusion
        with torch.no_grad():
            logits, detections, _ = primary(images)
        for det, lg in zip(detections, logits):
            assert det['boxes'].shape[1] == 4 and len(det['boxes']) <= 100
            assert det['boxes'].min() >= 0 and det['boxes'].max() <= 512
            assert ((det['labels'] >= 1) & (det['labels'] <= 9)).all() and (det['scores'] <= 1).all()
            if fusion == 'hard':
                assert (det['labels'] == lg.argmax() + 1).all()  # every box takes the image's class
    primary.roi_heads.fusion = 'soft'


class _Stub(nn.Module):
    embed_dim = 768


def test_soft_and_hard_scores_follow_the_formula():
    heads = ClassGuidedRoIHeads.from_heads(MultiTaskModel(_Stub(), task_routing=False).roi_heads)
    class_logits = torch.tensor([[0.0, 2.0, 1.0] + [-9.0] * 7, [3.0, 0.0, 1.0] + [-9.0] * 7])  # bg, c1, c2, ...
    proposals = [torch.tensor([[10., 10., 50., 50.], [200., 200., 260., 240.]])]
    regression = torch.zeros(2, 40)  # boxes = proposals
    q = torch.tensor([[0.1, 0.9] + [0.0] * 7])  # the image is class 2 for the classifier
    heads.image_probs = q
    p = F.softmax(class_logits, -1)
    for fusion in ('soft', 'hard'):
        heads.fusion = fusion
        boxes, scores, labels = heads.postprocess_detections(class_logits, regression, proposals, [(512, 512)])
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            k = 0 if box[0] == 10 else 1
            joint = p[k, 1:] * q[0]
            expected = (1 - p[k, 0]) * (joint[label - 1] / joint.sum() if fusion == 'soft' else 1.0)
            assert torch.isclose(score, expected, atol=1e-6)
            if fusion == 'hard':
                assert label == 2
    heads.fusion = 'none'  # torchvision's own scoring
    _, scores, labels = heads.postprocess_detections(class_logits, regression, proposals, [(512, 512)])
    assert torch.isclose(scores[0].max(), p[0, 1])


def test_class_guidance_on_a_reference_model_keeps_its_weights():
    backbone = build_backbone('vit', pretrained=False)
    add_adapters(backbone, 'lora')
    model = MultiTaskModel(backbone, task_routing=False)
    keys = set(model.state_dict())
    add_class_guidance(model, 'hard')
    assert set(model.state_dict()) == keys  # reference model.pt files still load
    model.eval()
    with torch.no_grad():
        logits, detections, _ = model(torch.randn(1, 1, 512, 512))
    assert (detections[0]['labels'] == logits[0].argmax() + 1).all()


def test_terramind_moelora_variant_routes_both_tasks_within_budget():
    torch.manual_seed(0)
    model = build_model('terramind', pretrained=False, adapt='moelora', fusion='none')
    counts = count_parameters(model)
    check_budget(counts)
    # the TerraMind + MoE-LoRA reference (106,493,209 / 21,145,369) + detail branch + 9-anchor RPN head
    assert (counts['total'], counts['trainable']) == (106_836_680, 21_488_840)
    assert counts['adapters'] == 1_032_192 and counts['unfrozen_pretrained'] == 0
    calls = []  # (task, router) calls; MoE-LoRA's up weights start at 0, so routers get no gradient yet
    for m in model.backbone.modules():
        if hasattr(m, 'routers'):
            for task, router in enumerate(m.routers):
                router.register_forward_hook(lambda *_, task=task: calls.append(task))
    model.train()
    targets = [{'boxes': torch.tensor([[10., 10., 16., 15.]]), 'labels': torch.tensor([9])}]
    logits, _, losses = model(torch.randn(1, 1, 512, 512), targets)
    assert logits.shape == (1, 9) and all(torch.isfinite(v) for v in losses.values())
    n_layers = 12 * 2  # qkv and proj of 12 blocks
    assert calls == [0] * n_layers + [1] * n_layers  # the backbone ran once per task, classification first


def test_train_arguments_name_runs_and_reject_untested_combinations():
    from common import load_module  # by path: the reference train.py shadows the name `train` on sys.path
    train = load_module('detail_train', Path(__file__).resolve().parents[1] / 'train.py')
    assert train.config_name(train.parse_args([])) == 'detaillora'
    assert train.config_name(train.parse_args(['--no-detail', '--reference-anchors'])) == 'detaillora_nodetail_refanchors'
    args = train.parse_args(['--backbone', 'terramind', '--adapt', 'moelora', '--no-augment', '--fusion', 'none'])
    assert train.config_name(args) == 'detail_terramind_moelora_noaug'
    for bad in (['--backbone', 'terramind'], ['--adapt', 'moelora', '--rank', '16']):
        with pytest.raises(SystemExit):
            train.parse_args(bad)
