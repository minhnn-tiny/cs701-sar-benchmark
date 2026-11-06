"""compare.py: Δm on toy inputs, per seed against the mean of the reference seeds, mean ± sample std, and the JD
keep rule (a paired gain on all 3 seeds)."""
import math

from compare import REFERENCE, add_delta_m, delta_m, jd_rule, mean_std


def test_delta_m_is_the_mean_relative_change_in_percent():
    assert delta_m(0.9331, 0.2613, 0.9331, 0.2613) == 0
    # +10 % relative F1 and -10 % relative mAP cancel; +10 % on both is +10
    assert math.isclose(delta_m(0.99, 0.18, 0.9, 0.2), 0, abs_tol=1e-9)
    assert math.isclose(delta_m(0.99, 0.22, 0.9, 0.2), 10)
    # the leaderboard's example: full fine-tuning on test (95.8 / 31.8 vs 93.99 / 25.41) is about +13.5
    assert math.isclose(delta_m(0.958, 0.318, 0.9399, 0.2541), 13.5, abs_tol=0.1)


def test_delta_m_per_seed_against_the_mean_of_the_reference_seeds():
    runs = [{'config': REFERENCE, 'scores': {'macro_f1': f1, 'mAP': m}} for f1, m in [(0.92, 0.25), (0.94, 0.27)]]
    runs.append({'config': 'design', 'scores': {'macro_f1': 0.93, 'mAP': 0.286}})
    ref_f1, ref_mAP = add_delta_m(runs)
    assert math.isclose(ref_f1, 0.93) and math.isclose(ref_mAP, 0.26)
    assert math.isclose(runs[0]['delta_m'] + runs[1]['delta_m'], 0, abs_tol=1e-9)  # the reference averages to 0
    assert math.isclose(runs[2]['delta_m'], 5)  # F1 unchanged, mAP +10 % relative


def test_mean_std_is_the_sample_std():
    mean, std = mean_std([1.0, 2.0, 3.0])
    assert mean == 2 and math.isclose(std, 1)
    assert mean_std([4.0]) == (4.0, None)


def test_jd_is_kept_only_with_a_gain_on_all_3_seeds():
    def run(config, off, on):
        return {'config': config, 'scores': {'mAP': off}, 'scores_jd': {'mAP': on}}
    runs = [run('a', 0.20, 0.21), run('a', 0.22, 0.23), run('a', 0.21, 0.215),  # gains on every seed: kept
            run('b', 0.20, 0.25), run('b', 0.22, 0.24), run('b', 0.21, 0.20),  # one loss: not kept
            run('c', 0.20, 0.21), run('c', 0.22, 0.23)]  # 2 seeds only: not kept
    rule = jd_rule(runs)
    assert rule['a']['keep'] and not rule['b']['keep'] and not rule['c']['keep']
    assert [round(g, 3) for g in rule['b']['gains']] == [0.05, 0.02, -0.01]


def test_the_crossfit_adopts_a_decoding_only_if_both_folds_choose_it_and_gain_out_of_fold():
    from crossfit import decide

    def seed(ref, post):  # mAP per half (A, B) for the reference and posterior decodings; JD product = reference
        e = {d: {h: {'mAP': v} for h, v in zip(('half_A', 'half_B'), vals)}
             for d, vals in (('reference', ref), ('jd', ref), ('jd_posterior', post))}
        return e
    assert decide([seed((0.10, 0.12), (0.11, 0.13))] * 3)[0] == 'jd_posterior'  # better on both halves
    assert decide([seed((0.10, 0.12), (0.11, 0.11))] * 3)[0] == 'reference'  # better on A only: B's fold disagrees
    adopted, folds = decide([seed((0.10, 0.12), (0.11, 0.13))] * 3)
    assert [[round(g, 3) for g in f[3]] for f in folds] == [[0.01] * 3] * 2  # per seed, on the half not chosen on
    # A positive mean gain is not enough: the pre-registered rule needs a gain on every seed (cand_detect.md §2, step 4)
    mixed = [seed((0.10, 0.12), (0.11, 0.13))] * 2 + [seed((0.10, 0.12), (0.11, 0.11))]
    assert decide(mixed)[0] == 'reference'
