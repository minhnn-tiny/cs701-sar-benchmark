"""sarmtl_model.py on the CPU: output shapes, parameter counts within the budget, WideLoRA's adapters and unfrozen
tensors, SP starting as the model without it, and JD's decoding rules on hand-made RoI scores."""
import pytest
import torch

from sarbench.adapters import LoRA
from sarmtl_model import build_model, count_parameters, param_groups


@pytest.fixture(scope='module')
def primary():
    torch.manual_seed(0)
    return build_model(init='scratch', sp=True, seed=0)


@pytest.mark.parametrize('adapt, sp, total, trainable', [
    ('widelora', True, 115_956_473, 31_021_817),  # the primary design (DESIGN.md §2)
    ('widelora', False, 115_591_705, 30_657_049),  # ablation A1
    ('lora', False, 107_039_257, 20_997_913),  # the reference: root train.py's logged counts
])
def test_parameter_counts_match_the_design_and_fit_the_budget(adapt, sp, total, trainable):
    counted_total, counted_trainable, kinds = count_parameters(build_model(init='scratch', adapt=adapt, sp=sp))
    assert (counted_total, counted_trainable) == (total, trainable)
    assert counted_total <= 130_000_000 and counted_trainable <= 40_000_000
    assert kinds['heads'] == 20_113_177 and kinds['sp'] == (364_768 if sp else 0)


def test_wide_lora_wraps_every_linear_layer_at_rank_64_scale_1(primary):
    loras = [m for m in primary.backbone.modules() if isinstance(m, LoRA)]
    assert len(loras) == 48 and all(m.down.out_features == 64 and m.scale == 1.0 for m in loras)
    assert all(not m.base.weight.requires_grad and m.base.bias.requires_grad for m in loras)
    assert all(m.up.weight.abs().sum() == 0 for m in loras)  # B = 0: training starts at the pretrained model
    lora, heads, pretrained = param_groups(primary, 1e-4, 2e-5)
    assert [sum(p.numel() for p in g['params']) for g in (lora, heads, pretrained)] == [9_437_184, 20_477_945, 1_106_688]
    assert (lora['lr'], heads['lr'], pretrained['lr']) == (1e-4, 1e-4, 2e-5)


def test_outputs_and_losses_have_the_expected_shapes(primary):
    primary.eval()
    with torch.no_grad():
        logits, detections, losses = primary(torch.randn(2, 1, 512, 512))
    assert logits.shape == (2, 9) and len(detections) == 2 and losses == {}
    assert set(detections[0]) == {'boxes', 'labels', 'scores'} and detections[0]['boxes'].shape[1:] == (4,)
    primary.train()
    targets = [{'boxes': torch.tensor([[10.0, 20.0, 30.0, 44.0], [100.0, 100.0, 108.0, 107.0]]),
                'labels': torch.tensor([3, 3])}]
    logits, _, losses = primary(torch.randn(1, 1, 512, 512), targets)
    assert logits.shape == (1, 9)
    assert set(losses) == {'loss_objectness', 'loss_rpn_box_reg', 'loss_classifier', 'loss_box_reg'}
    assert all(loss.isfinite() for loss in losses.values())
    primary.eval()


def test_sp_starts_as_the_model_without_it():
    """Same seed: the shared weights are identical (SP is built from its own RNG) and, with the laterals' GN weight at
    0, so are the outputs: the paired A1 run starts from the same function."""
    torch.manual_seed(0)
    with_sp = build_model(init='scratch', sp=True, seed=0).eval()
    torch.manual_seed(0)
    without = build_model(init='scratch', sp=False, seed=0).eval()
    shared = without.state_dict()
    assert all(torch.equal(v, shared[k]) for k, v in with_sp.state_dict().items() if k in shared)
    images = torch.randn(1, 1, 512, 512)
    with torch.no_grad():
        (la, da, _), (lb, db, _) = with_sp(images), without(images)
    assert torch.equal(la, lb) and torch.equal(da[0]['scores'], db[0]['scores'])


def decode(heads, class_logits, logp, **decoding):
    """All RoI scores out of the (JD) postprocess: no threshold, no NMS, no top-k."""
    proposals = [torch.tensor([[10.0, 10, 50, 50], [60, 60, 120, 100], [200, 200, 260, 280], [5, 300, 40, 340]])]
    settings = heads.score_thresh, heads.nms_thresh, heads.detections_per_img
    heads.score_thresh, heads.nms_thresh, heads.detections_per_img = 0.0, 1.0, 1000
    heads.decode = dict(logp=logp, **decoding) if decoding else None
    try:
        _, scores, _ = heads.postprocess_detections(class_logits, torch.zeros(4, 40), proposals, [(512, 512)])
    finally:
        heads.decode = None
        heads.score_thresh, heads.nms_thresh, heads.detections_per_img = settings
    return scores[0].sort().values


def test_jd_product_is_s_times_p_and_never_raises_a_score(primary):
    torch.manual_seed(1)
    class_logits, logp = torch.randn(4, 10), torch.randn(1, 9).log_softmax(-1)
    s, p = class_logits.softmax(-1), logp.exp()
    assert torch.allclose(decode(primary.roi_heads, class_logits, logp), s[:, 1:].flatten().sort().values)
    product = decode(primary.roi_heads, class_logits, logp, alpha=1.0, mode='product')
    assert torch.allclose(product, (s[:, 1:] * p).flatten().sort().values, atol=1e-6)
    assert (s[:, 1:] * p <= s[:, 1:]).all()
    posterior = decode(primary.roi_heads, class_logits, logp, alpha=1.0, mode='posterior')
    expected = (1 - s[:, :1]) * s[:, 1:] * p / (s[:, 1:] * p).sum(1, keepdim=True)
    assert torch.allclose(posterior, expected.flatten().sort().values, atol=1e-6)
