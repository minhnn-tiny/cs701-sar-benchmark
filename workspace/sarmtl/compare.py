"""Results tables from every metrics.json: per configuration, the mean ± sample std over seeds of accuracy, macro-F1,
mAP, AP50 and Δm. Δm is computed per seed against the local reference (ViT + LoRA trained on the same holdout, mean
of its seeds, reference decoding), then averaged, as the TA's tables are. Also the JD keep rule and the submission
configuration C* (DESIGN.md §4), parameters, GPU hours, peak memory, per-class AP and the metrics.json behind every
row. JD scores come from metrics.json (train.py runs) or from jd_eval.json (evaluate.py, e.g. on root train.py runs).
Example (from the repo root): uv run python workspace/sarmtl/compare.py [--out workspace/sarmtl/runs/tables.md]"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import crossfit

ROOT = Path(__file__).resolve().parents[2]  # the repo root
REFERENCE = 'vit_pretrained_lora'
CLASSES = ['aircraft', 'airport', 'bridge', 'car', 'harbor', 'oil_tank', 'playground', 'ship', 'wind_turbine']
TINY = ['wind_turbine', 'bridge', 'oil_tank', 'playground']  # ≥ 92 % COCO-small (DATA.md §0)


def delta_m(f1, mAP, ref_f1, ref_mAP):
    """Δm in %: the mean relative change of macro-F1 and mAP against the reference (Maninis et al., CVPR 2019)."""
    return 100 * ((f1 - ref_f1) / ref_f1 + (mAP - ref_mAP) / ref_mAP) / 2


def mean_std(values):
    """Mean and sample standard deviation (None for a single value)."""
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else None)


def fmt(values, scale=100, sign=False):
    """'mean ± std' of values (fractions shown in % with scale=100)."""
    mean, std = mean_std([v * scale for v in values])
    text = f'{mean:+.1f}' if sign else f'{mean:.1f}'
    return text + (f' ± {std:.1f}' if std is not None else ' (1 seed)')


def load_runs(roots):
    """One dict per metrics.json under roots: config, data dir name, seed, final scores (reference decoding and JD)
    and costs."""
    runs = []
    for root in roots:
        for path in sorted(Path(root).rglob('metrics.json')):
            m = json.loads(path.read_text())
            a = m['args']
            jd_eval = json.loads((path.parent / 'jd_eval.json').read_text()) \
                if (path.parent / 'jd_eval.json').exists() else None  # evaluate.py: all decodings, halves
            jd = m.get('val_jd') or (jd_eval['jd']['all'] if jd_eval else None)
            runs.append({
                'path': path.resolve().relative_to(ROOT) if path.resolve().is_relative_to(ROOT) else path,
                'config': m.get('config') or f"{a['backbone']}_{a['init']}_{a['adapt']}",
                'data': Path(a['data']).name, 'seed': a['seed'],
                'complete': a.get('limit') is None and a['epochs'] == 24,  # else a smoke test or partial run
                'scores': m['val'],  # the last-epoch scores on the data's val split (the holdout), reference decoding
                'scores_jd': jd,  # the same checkpoint with JD, if scored
                'jd_eval': jd_eval,
                'parameters': m['parameters'], 'trainable': m['trainable_parameters'],
                'train_hours': m['train_hours'], 'peak_gb': m['peak_gpu_memory_gb'],
                'ms_per_image': m.get('inference_ms_per_image'),
            })
    return runs


def by_config(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[run['config']].append(run)
    return dict(sorted(groups.items(), key=lambda kv: (kv[0] != REFERENCE, kv[0])))


def add_delta_m(runs, key='scores', out='delta_m'):
    """Set run[out] from run[key] against the mean of the reference's runs among `runs` (reference decoding);
    returns the reference (F1, mAP)."""
    ref = [r['scores'] for r in runs if r['config'] == REFERENCE]
    if not ref:
        return None
    ref_f1, ref_mAP = (statistics.mean(s[k] for s in ref) for k in ('macro_f1', 'mAP'))
    for r in runs:
        if r.get(key) is not None:
            r[out] = delta_m(r[key]['macro_f1'], r[key]['mAP'], ref_f1, ref_mAP)
    return ref_f1, ref_mAP


def jd_rule(runs):
    """DESIGN.md §4, rule 3: per configuration, the paired mAP gains (JD − reference decoding, same checkpoint) and
    whether JD is kept: a gain > 0 on all 3 seeds."""
    rule = {}
    for config, rs in by_config(runs).items():
        gains = [r['scores_jd']['mAP'] - r['scores']['mAP'] for r in rs if r['scores_jd'] is not None]
        rule[config] = {'gains': gains, 'keep': len(gains) == len(rs) >= 3 and min(gains) > 0}
    return rule


def choose_decodings(runs, rule):
    """Per configuration: the decoding adopted by the 2-fold cross-fit (crossfit.py; needs every seed's jd_eval.json),
    else JD if it passed the keep rule, else the reference decoding. Returns {config: (decoding, cross-fit folds)}."""
    choice = {}
    for config, rs in by_config(runs).items():
        evals = [r['jd_eval'] for r in rs]
        adopted, folds = crossfit.decide(evals) if len(rs) >= 3 and all(evals) else ('reference', [])
        if adopted == 'reference' and rule[config]['keep']:
            adopted = 'jd'
        choice[config] = (adopted, folds)
    return choice


def as_submitted(runs, choice):
    """Each run scored as it would be submitted: with its configuration's decoding."""
    for r in runs:
        decoding = choice[r['config']][0]
        r['decoding'] = decoding
        r['submitted'] = {'reference': r['scores'], 'jd': r['scores_jd']}.get(decoding) or \
            r['jd_eval'][decoding]['all']


def results_table(runs, key='scores', dm='delta_m', with_delta_m=True, decoding=False):
    head = ['config', 'seeds'] + (['decoding'] if decoding else []) + ['accuracy', 'macro-F1', 'mAP', 'AP50'] + \
           (['Δm'] if with_delta_m else []) + ['total params', 'trainable params', 'GPU h per run', 'GPU h',
                                               'peak GiB', 'eval ms/image']
    left = 3 if decoding else 2  # text columns; the numbers are right-aligned
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * left + '---:|' * (len(head) - left)]
    for config, rs in by_config(runs).items():
        cells = [config, ','.join(str(r['seed']) for r in sorted(rs, key=lambda r: r['seed']))]
        if decoding:
            cells.append(rs[0]['decoding'])
        cells += [fmt([r[key][k] for r in rs]) for k in ('accuracy', 'macro_f1', 'mAP', 'AP50')]
        if with_delta_m:
            cells.append(fmt([r[dm] for r in rs], scale=1, sign=True) if all(dm in r for r in rs) else '—')
        ms = [r['ms_per_image'] for r in rs if r['ms_per_image'] is not None]
        cells += [f"{rs[0]['parameters'] / 1e6:.2f} M", f"{rs[0]['trainable'] / 1e6:.2f} M",
                  f"{statistics.mean(r['train_hours'] for r in rs):.2f}", f"{sum(r['train_hours'] for r in rs):.2f}",
                  f"{max(r['peak_gb'] for r in rs):.1f}", f'{statistics.mean(ms):.1f}' if ms else '—']
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def per_class_table(runs, key='scores'):
    lines = ['| config | ' + ' | '.join(CLASSES) + ' | tiny-4 mean |', '|---|' + '---:|' * (len(CLASSES) + 1)]
    for config, rs in by_config(runs).items():
        ap = {c: statistics.mean(r[key]['per_class_AP'][c] for r in rs) for c in CLASSES}
        cells = [f'{100 * ap[c]:.1f}' for c in CLASSES] + [f'{100 * statistics.mean(ap[c] for c in TINY):.1f}']
        lines.append(f'| {config} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def queue_hours(queue_dir):
    """Wall-clock GPU hours of the jobs run.sh ran (claimed jobs with a start and an end)."""
    hours = 0.0
    for job in Path(queue_dir).glob('*/start'):
        end = job.parent / 'end'
        if end.exists():
            hours += (int(end.read_text()) - int(job.read_text())) / 3600
    return hours


def report(runs, holdout, queue_dir):
    out = []
    local = [r for r in runs if r['data'] == holdout and r['complete']]
    ref = add_delta_m(local)
    out.append(f'## Local holdout ({holdout}), last-epoch model, %, mean ± sample std over seeds\n')
    if ref:
        out.append(f'Δm reference: {REFERENCE} on {holdout}, reference decoding, mean of its seeds: macro-F1 '
                   f'{100 * ref[0]:.2f}, mAP {100 * ref[1]:.2f}.\n')
    else:
        out.append(f'No complete {REFERENCE} run on {holdout} yet: Δm cannot be computed.\n')
    out += ['### Reference decoding (JD off: ablation A2 for every configuration)\n',
            results_table(local, with_delta_m=ref is not None), '']

    rule = jd_rule(local)
    out += ['### JD keep rule (DESIGN.md §4, rule 3): paired mAP gain, JD − reference decoding, same checkpoint\n',
            '| config | gains per seed (mAP points) | keep JD |', '|---|---|---|']
    out += [f"| {c} | {', '.join(f'{100 * g:+.2f}' for g in v['gains']) or '—'} | {'yes' if v['keep'] else 'no'} |"
            for c, v in rule.items()] + ['']
    choice = choose_decodings(local, rule)
    out += ['### 2-fold cross-fit of the decodings (rule 3; crossfit.py): chosen on one half by the 3-seed mean mAP, '
            'scored on the other, adopted only if both folds choose it and it gains on every seed\n',
            '| config | choose on → score on | chosen | out-of-fold mAP gain per seed (points) | mean | adopted decoding |',
            '|---|---|---|---|---:|---|']
    for config, (adopted, folds) in choice.items():
        out += [f"| {config} | {pick} → {score} | {chosen} | {', '.join(f'{100 * g:+.2f}' for g in gains)} | "
                f'{100 * statistics.mean(gains):+.2f} | {adopted} |' for pick, score, chosen, gains in folds] \
            or [f'| {config} | — (needs the jd_eval.json of all 3 seeds) | | | | {adopted} |']
    out.append('')
    if ref:
        as_submitted(local, choice)
        add_delta_m(local, key='submitted', out='delta_m_submitted')
        out += ['### As submitted: each configuration with its own decoding (Δm against the reference decoding of '
                f'{REFERENCE})\n', results_table(local, key='submitted', dm='delta_m_submitted', decoding=True), '']
        eligible = {c: statistics.mean(r['delta_m_submitted'] for r in rs)
                    for c, rs in by_config(local).items() if len(rs) >= 3}
        if eligible:
            best = max(eligible, key=eligible.get)
            out.append('**C\\* (DESIGN.md §4, rule 4)**: ' + ', '.join(f'{c} {v:+.2f}' for c, v in eligible.items())
                       + f' → **{best}**\n')
        out += ['### Per-class AP (%), as submitted, mean over seeds\n', per_class_table(local, 'submitted'), '']
    out += ['### Per-class AP (%), reference decoding, mean over seeds\n', per_class_table(local), '']

    for data in sorted({r['data'] for r in runs if r['complete']} - {holdout}):
        other = [r for r in runs if r['data'] == data and r['complete']]
        out += [f'## Other runs on {data} (its val split is not the local holdout: no Δm)\n',
                results_table(other, with_delta_m=False), '']
    partial = [r for r in runs if not r['complete']]
    if partial:
        out += ['## Smoke tests and partial runs (GPU hours only)\n', '| run | GPU h |', '|---|---:|']
        out += [f"| {r['path'].parent} | {r['train_hours']:.3f} |" for r in partial] + ['']
    out += ['## Sources (metrics.json per configuration, complete runs)\n']
    for data in sorted({r['data'] for r in runs if r['complete']}):
        for config, rs in by_config([r for r in runs if r['data'] == data and r['complete']]).items():
            out.append(f'- {data} / {config}: ' + ', '.join(f"`{r['path']}`" for r in rs))
    out += ['', f"Total training GPU hours (train_hours of every metrics.json above): "
                f"{sum(r['train_hours'] for r in runs):.2f}"]
    if Path(queue_dir).exists():
        out.append(f'Wall-clock GPU hours of the jobs run by run.sh: {queue_hours(queue_dir):.2f}')
    return '\n'.join(out) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+',
                   default=[ROOT / 'workspace/sarmtl/runs', ROOT / 'workspace/baseline/runs'])
    p.add_argument('--holdout', default='holdout_sarmtl', help='name of the holdout data dir')
    p.add_argument('--queue', type=Path, default=ROOT / 'workspace/sarmtl/runs/queue')
    p.add_argument('--out', type=Path, help='also write the tables to this file')
    args = p.parse_args()
    text = report(load_runs(args.runs), args.holdout, args.queue)
    print(text, end='')
    if args.out:
        args.out.write_text(text)


if __name__ == '__main__':
    main()
