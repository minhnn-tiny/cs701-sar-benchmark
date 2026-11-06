"""The decoding rule (crossfit.decide), the TA's configurations on the holdout and evaluate.py's guard against scoring
training images."""
from pathlib import Path

import pytest

from crossfit import HOLDOUT, complete, decide, ta_bar
from evaluate import check_disjoint, local_reference

ROOT = Path(__file__).resolve().parents[3]


def seed(ref, x, y):
    """A decode_eval.json with three decodings; each value is (half_A, half_B, all) mAP."""
    return {'decodings': {name: {split: {'mAP': v} for split, v in zip(('half_A', 'half_B', 'all'), values)}
                          for name, values in (('reference', ref), ('x', x), ('y', y))}}


def test_adopted_only_when_every_out_of_fold_gain_is_positive():
    good = [seed((.20, .20, .20), (.22, .23, .225), (.21, .21, .21)) for _ in range(3)]
    adopted, folds = decide(good)
    assert adopted == 'x' and [f[2] for f in folds] == ['x', 'x']
    mixed = good[:2] + [seed((.20, .20, .20), (.22, .19, .205), (.21, .21, .21))]  # x loses on half B for one seed
    assert decide(mixed)[0] == 'reference'


def test_only_seeds_0_1_2_of_one_configuration_can_adopt():
    e = {'data': HOLDOUT, 'decodings': {'reference': {}, 'x': {}}}
    args = lambda s, **kw: {'seed': s, 'out': f'runs/H_s{s}', 'lr': 1e-4, **kw}  # noqa: E731
    assert complete([(args(s), e) for s in (0, 1, 2)])
    assert not complete([(args(0), e)])  # a single-seed screen
    assert not complete([(args(0), e), (args(1), e), (args(2, lr=2e-4), e)])  # another configuration's seed
    assert not complete([(args(0), e), (args(1), e), (args(1), e)])


@pytest.mark.skipif(not (ROOT / HOLDOUT).exists(), reason='holdout not built')
def test_a_run_trained_on_the_holdout_images_is_refused():
    check_disjoint({'data': HOLDOUT}, ROOT / HOLDOUT)  # local train vs holdout: disjoint
    with pytest.raises(SystemExit, match='refused'):
        check_disjoint({'data': 'dataset/repro'}, ROOT / HOLDOUT)  # all of train includes the holdout


@pytest.mark.skipif(not (ROOT / 'workspace/sarmtl/runs/tm_s2').exists(), reason='sarmtl runs absent')
def test_the_bar_is_terramind_moelora_as_in_sarmtl_results():
    bars = ta_bar(local_reference(HOLDOUT))
    assert bars['TerraMind + MoE-LoRA'][0] == pytest.approx(9.9, abs=0.05)  # RESULTS.md §1.3: +9.9 ± 2.4
    assert bars['ViT + LoRA'][0] < bars['TerraMind + MoE-LoRA'][0]
