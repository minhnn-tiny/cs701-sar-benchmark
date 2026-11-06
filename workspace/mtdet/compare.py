"""The results table: every finished run (runs/*/metrics.json) on ND (the holdout), LF and FIT, then per configuration
(same args but the seed) the seed mean ± std of ND Δm under the training-time decoding and under the decoding with the
best seed-mean ND Δm (decode_eval.json, only decodings scored for every seed); all-train runs are listed, never
selected on. FitDet's runs on the same holdout are
listed for context. Δm against the leaderboard's val reference (93.31 / 26.13).
Example: uv run python workspace/mtdet/compare.py [--per-class]"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

from train import LATER  # args added after runs had started: a run without one ran at its default

HERE = Path(__file__).resolve().parent
NOT_CONFIG = {'seed', 'out', 'workers', 'eval_every', 'evals', 'limit', 'max_steps'}


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def runs(root):
    for path in sorted(root.glob('*/metrics.json')):
        if path.parent.name != 'smoke':
            yield path.parent, json.loads(path.read_text())


def row(name, m):
    f = m['final']
    nd, lf, fit = f.get('nd', {}), f.get('lf', {}), f.get('fit', {})
    get = lambda d, k: f'{100 * d[k]:6.2f}' if k in d else '     -'  # noqa: E731
    return (f"{name:14s} {m['args'].get('arch', 'r50fpn-mt'):11s} {m['args']['seed']:>2} {m['args']['epochs']:>3} "
            f"{m['train_hours']:5.2f} | {get(nd, 'macro_f1')} {get(nd, 'mAP')} {nd.get('delta_m', float('nan')):+7.2f} | "
            f"{get(lf, 'mAP')} {get(fit, 'mAP')}")


def per_class(name, m):
    """AP per class on ND and FIT (training decoding)."""
    for split in ('nd', 'fit'):
        d = m['final'].get(split)
        if d:
            print(f"  {name:22s} {split:3s} " + ' '.join(f"{k[:5]} {100 * v:5.1f}" for k, v in d['per_class_AP'].items()))


def main():
    import sys
    show_classes = '--per-class' in sys.argv
    print('run            arch        seed ep  GPU h | ND F1  ND mAP  ND Δm   | LF mAP FIT mAP   (training decoding)')
    configs = defaultdict(list)
    for run, m in runs(HERE / 'runs'):
        print(row(run.name, m) + ('' if m.get('held_out', True) else '   all-train: ND/LF in-train, not selected on'))
        if show_classes:
            per_class(run.name, m)
        if not m.get('held_out', True):
            continue
        key = json.dumps({k: v for k, v in {**LATER, **m['args']}.items() if k not in NOT_CONFIG}, sort_keys=True,
                         default=str)
        configs[key].append((run, m))
    fitdet = HERE.parent / 'fitdet' / 'runs'
    for run, m in runs(fitdet):
        if 'final' in m and 'nd' in m['final']:
            m['final']['nd'].setdefault('delta_m', 50 * ((100 * m['final']['nd']['macro_f1'] - 93.31) / 93.31 +
                                                        (100 * m['final']['nd']['mAP'] - 26.13) / 26.13))
            print(row('fitdet/' + run.name, m))
            if show_classes:
                per_class('fitdet/' + run.name, m)

    print('\nconfiguration: seeds | ND Δm (training decoding) | best decoding: ND Δm, ND mAP (seed mean ± std)')
    for key, members in configs.items():
        dm = mean_std([m['final']['nd']['delta_m'] for _, m in members])
        evals = [json.loads((r / 'decode_eval.json').read_text()) for r, _ in members if (r / 'decode_eval.json').exists()]
        best = ''
        if evals and len(evals) == len(members):
            common = set.intersection(*(set(e) for e in evals))
            name = max(common, key=lambda d: statistics.mean(e[d]['delta_m'] for e in evals))
            bdm, bmap = mean_std([e[name]['delta_m'] for e in evals]), mean_std([100 * e[name]['mAP'] for e in evals])
            best = f'{name}: {bdm[0]:+.2f} ± {bdm[1]:.2f}, mAP {bmap[0]:.2f} ± {bmap[1]:.2f}'
        names = ','.join(r.name for r, _ in members)
        print(f'{names}: {len(members)} | {dm[0]:+.2f} ± {dm[1]:.2f} | {best}\n    {key}')


if __name__ == '__main__':
    main()
