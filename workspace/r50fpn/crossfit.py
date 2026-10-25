"""The decoding choice of R50-FPN-MT (README.md, fixed before any result), next to the TA's configurations.
- Choice: a 2-fold cross-fit over the holdout halves (halves.json) of evaluate.py's decode_eval.json, one per seed. In
  each fold, the decoding with the best seed-mean mAP on one half is chosen and scored against the reference decoding
  on the other half, per seed. If every out-of-fold gain (2 folds x 3 seeds) is positive, the selection procedure
  carries to images it did not see, and it is applied to the whole holdout: the decoding with the best seed-mean mAP
  there is adopted (the gate validates the procedure, not that winner on its own); otherwise the reference decoding is
  kept. Only seeds 0, 1 and 2 of one configuration can adopt a decoding.
- Context: the TA's in-budget configurations trained on the same holdout (workspace/sarmtl's runs), each at its best
  decoding. CLAUDE.md's submission rule (25 Oct) doesn't gate the upload on them; the posting bar is the val score.
Example (from the repo root): uv run python workspace/r50fpn/crossfit.py"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / 'workspace' / 'sarmtl')]
from evaluate import REFERENCE_RUNS, delta_m, local_reference  # noqa: E402  workspace/r50fpn/evaluate.py

FOLDS = (('half_A', 'half_B'), ('half_B', 'half_A'))  # (choose on, score on)
HOLDOUT = 'dataset/holdout_sarmtl'
TA_CONFIGS = {'ViT + LoRA': 'ref_s*/vit_pretrained_lora', 'TerraMind + MoE-LoRA': 'tm_s*/terramind_pretrained_moelora'}


def decide(evals):
    """evals: the decode_eval.json dicts of one configuration's seeds. Returns (adopted decoding, folds), folds =
    [(choose on, score on, chosen decoding, out-of-fold mAP gains vs the reference decoding, one per seed)]."""
    names = list(evals[0]['decodings'])
    mean = lambda split: {d: statistics.mean(e['decodings'][d][split]['mAP'] for e in evals) for d in names}  # noqa
    folds = []
    for pick, score in FOLDS:
        on_pick = mean(pick)
        chosen = max(on_pick, key=on_pick.get)
        gains = [e['decodings'][chosen][score]['mAP'] - e['decodings']['reference'][score]['mAP'] for e in evals]
        folds.append((pick, score, chosen, gains))
    on_all = mean('all')
    adopted = max(on_all, key=on_all.get) if all(g > 0 for f in folds for g in f[3]) else 'reference'
    return adopted, folds


def complete(runs):
    """runs: [(training args, decode_eval.json)] of one configuration. True only for exactly seeds 0, 1 and 2 with the
    same training args apart from seed and out, scored on the same holdout with the same decodings: the rule's input."""
    rest = [({k: v for k, v in a.items() if k not in ('seed', 'out')}, e['data'], list(e['decodings'])) for a, e in runs]
    return sorted(a['seed'] for a, _ in runs) == [0, 1, 2] and all(r == rest[0] for r in rest)


def ta_bar(ref):
    """{TA configuration: (best seed-mean holdout Δm over its decodings, that decoding)} from workspace/sarmtl's runs:
    metrics.json (reference decoding) and jd_eval.json (its JD decodings)."""
    bars = {}
    for name, pattern in TA_CONFIGS.items():
        dirs = sorted(REFERENCE_RUNS.glob(pattern))
        assert len(dirs) == 3, f'expected 3 seeds of {name}'
        per = defaultdict(list)
        for d in dirs:
            m = json.loads((d / 'metrics.json').read_text())
            assert m['args']['data'] == HOLDOUT, f'{d} was not trained on {HOLDOUT}'
            scores = {'reference': m['val']}
            if (d / 'jd_eval.json').exists():
                scores.update({k: v['all'] for k, v in json.loads((d / 'jd_eval.json').read_text()).items()
                               if k != 'reference' and isinstance(v, dict) and 'all' in v})
            for k, s in scores.items():
                per[k].append(delta_m(s['macro_f1'], s['mAP'], ref))
        means = {k: statistics.mean(v) for k, v in per.items() if len(v) == 3}
        best = max(means, key=means.get)
        bars[name] = (means[best], best)
    return bars


def fmt(values):
    return f'{statistics.mean(values):+.1f} ± {statistics.stdev(values):.1f}' if len(values) > 1 else f'{values[0]:+.1f}'


def main():
    ref = local_reference(HOLDOUT)
    runs = defaultdict(list)  # configuration (run name without _s<seed>) -> (training args, decode_eval.json) per seed
    for path in sorted((HERE / 'runs').glob('*/r50fpn/decode_eval.json')):
        name = path.parts[-3]
        if not name.startswith('smoke'):
            args = json.loads((path.parent / 'metrics.json').read_text())['args']
            runs[re.sub(r'_s\d+$', '', name)].append((args, json.loads(path.read_text())))
    bars = ta_bar(ref)
    print(f'Local Δm reference (ViT + LoRA on {HOLDOUT}, 3 seeds): macro-F1 {100 * ref[0]:.2f}, mAP {100 * ref[1]:.2f}')
    print('TA configurations on this holdout (context, not an upload gate): '
          + '; '.join(f'{k} {v:+.1f} ({d})' for k, (v, d) in bars.items()))
    for config, seeds in runs.items():
        evals = [e for _, e in seeds]
        dm = lambda d: fmt([e['decodings'][d]['all']['delta_m'] for e in evals])  # noqa: E731
        print(f"\n{config} (seeds {', '.join(str(a['seed']) for a, _ in seeds)})")
        if not complete(seeds):  # e.g. a single-seed screen, compared on the reference decoding only
            print(f"  incomplete, no decoding adopted (the rule needs seeds 0, 1 and 2 of one configuration); holdout "
                  f"Δm reference decoding {dm('reference')}")
            continue
        adopted, folds = decide(evals)
        for pick, score, chosen, gains in folds:
            print(f"  choose on {pick}, score on {score}: {chosen}, out-of-fold mAP gain per seed "
                  + ', '.join(f'{100 * g:+.2f}' for g in gains))
        print(f"  adopted: {adopted}; holdout Δm reference decoding {dm('reference')}, as submitted {dm(adopted)}")


if __name__ == '__main__':
    main()
