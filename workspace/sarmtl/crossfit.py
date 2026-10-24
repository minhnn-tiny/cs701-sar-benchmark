"""The 2-fold cross-fit of DESIGN.md §4 rule 3, as pre-registered in cand_detect.md §2 (step 4). A decoding other than
the pre-registered pair (reference, JD product) is adopted for a configuration only if both folds choose it, by the
3-seed mean mAP on one half of the holdout, and in both folds it beats the reference decoding on the other half on
every seed. Reads the jd_eval.json files of evaluate.py; compare.py uses decide(). Example (from the repo root):
uv run python workspace/sarmtl/crossfit.py"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECODINGS = ['reference', 'jd', 'jd_posterior']
FOLDS = (('half_A', 'half_B'), ('half_B', 'half_A'))  # (choose on, score on)


def decide(evals):
    """evals: the jd_eval.json dicts of one configuration's seeds. Returns (adopted decoding or 'reference', folds),
    folds = [(choose on, score on, chosen decoding, out-of-fold mAP gains vs the reference decoding, one per seed)]."""
    folds = []
    for pick, score in FOLDS:
        mean = {d: statistics.mean(e[d][pick]['mAP'] for e in evals) for d in DECODINGS}
        chosen = max(mean, key=mean.get)
        gains = [e[chosen][score]['mAP'] - e['reference'][score]['mAP'] for e in evals]
        folds.append((pick, score, chosen, gains))
    chosen = {f[2] for f in folds}
    adopted = chosen.pop() if len(chosen) == 1 and all(g > 0 for f in folds for g in f[3]) else 'reference'
    return adopted, folds


def main():
    runs = defaultdict(list)  # config -> the jd_eval.json of each seed
    for path in sorted((ROOT / 'workspace/sarmtl/runs').glob('*/*/jd_eval.json')):
        if not path.parts[-3].startswith('smoke'):
            runs[path.parent.name].append(json.loads(path.read_text()))
    print('| config | seeds | fold (choose on → score on) | chosen | out-of-fold mAP gain vs reference per seed '
          '(points) | mean | adopted |')
    print('|---|---:|---|---|---|---:|---|')
    for config, evals in runs.items():
        adopted, folds = decide(evals)
        for pick, score, chosen, gains in folds:
            print(f"| {config} | {len(evals)} | {pick} → {score} | {chosen} | {', '.join(f'{100 * g:+.2f}' for g in gains)}"
                  f' | {100 * statistics.mean(gains):+.2f} | {adopted} |')


if __name__ == '__main__':
    main()
