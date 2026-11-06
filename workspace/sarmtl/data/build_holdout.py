"""Draw the holdout and audit it until the audit finds no crossing link (the split is then leak-free with respect to
every detector in near_dups.py and audit_split.py), then run the loader check. Deterministic.

Guards (lead's rules): at most 5 rounds; every round logs the crossing links found and the largest group per class
(analysis/build_holdout_log.json). If the loop has not converged by round 5, or a class falls below 8 % holdout, it
stops with the numbers (exit code 1) instead of loosening any threshold.
Usage (repo root, after extract_features.py and near_dups.py): uv run python workspace/sarmtl/data/build_holdout.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / 'dataset' / 'holdout_sarmtl'
ANALYSIS = OUT / 'analysis'
MAX_ROUNDS, MIN_SHARE = 5, 0.08


def run(script, *args, log):
    out = subprocess.run([sys.executable, str(HERE / script), *args], capture_output=True, text=True, check=True).stdout
    (ANALYSIS / log).write_text(out)
    return out


rounds = []
for it in range(1, MAX_ROUNDS + 1):
    run('make_holdout.py', '--fraction', '0.1', '--seed', '0', log='make_holdout_output.txt')
    split = json.loads((OUT / 'split.json').read_text())
    out = run('audit_split.py', '--emit-links', log=f'audit_round{it}.txt')
    new = int(re.search(r'NEW_AUDIT_LINKS (\d+)', out).group(1))
    crossing = re.search(r'thumbnail peak >= 0.3: (\d+) \((\d+) checked now\).*?: (\d+); passing the link rule[^:]*: (\d+)',
                         out)
    per_class = split['counts']['per_class']
    share = {c: v['holdout_images'] / (v['holdout_images'] + v['train_images'] + v['buffer_images'])
             for c, v in per_class.items()}
    rounds.append(dict(round=it, new_crossing_links=new, crossing_candidates=int(crossing.group(1)),
                       candidates_checked_now=int(crossing.group(2)), crossing_report_level=int(crossing.group(3)),
                       crossing_link_rule=int(crossing.group(4)), largest_group_per_class=split['largest_group_per_class'],
                       holdout_share_per_class={c: round(v, 4) for c, v in share.items()},
                       buffer_ids=split['buffer_ids']))
    (ANALYSIS / 'build_holdout_log.json').write_text(json.dumps(rounds, indent=1))
    print(f'round {it}: {new} new crossing links (crossing pairs passing the link rule: {crossing.group(4)}); '
          f'largest group per class {split["largest_group_per_class"]}; min holdout share '
          f'{min(share.values()):.3f} ({min(share, key=share.get)})', flush=True)
    low = {c: round(v, 4) for c, v in share.items() if v < MIN_SHARE}
    if low:
        sys.exit(f'STOP: classes below {MIN_SHARE:.0%} holdout: {low}')
    if new == 0:
        (ANALYSIS / 'audit_split.txt').write_text(out)
        break
else:
    sys.exit(f'STOP: not converged after {MAX_ROUNDS} rounds; see {ANALYSIS / "build_holdout_log.json"}')
print(run('verify_holdout.py', log='verify_holdout.txt'))
