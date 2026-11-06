"""Markdown tables of trained runs on the holdout: accuracy, macro-F1, mAP, AP50, parameters, hours, and Δm per
seed against the local ViT + LoRA reference, as mean ± sample std over seeds (the TA's convention: Δm per seed
against the mean of the reference's seeds, then averaged). Reads every metrics.json under the given paths, from
DetailLoRA runs and from reference runs of the top-level train.py; fusion_val.json (evaluate.py) adds ablation A2.
Runs are grouped by configuration, training data, epochs and --limit, so smoke runs never mix with real ones.
Example (repo root): uv run python workspace/detail_lora/compare.py workspace/detail_lora/runs"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CLASSES = ['aircraft', 'airport', 'bridge', 'car', 'harbor', 'oil_tank', 'playground', 'ship', 'wind_turbine']
NAMES = {'detaillora': 'DetailLoRA (primary)', 'detaillora_nodetail_refanchors': 'A1: DetailLoRA − detail − anchors',
         'detail_terramind_moelora_noaug': 'TerraMind + MoE-LoRA + detail + anchors'}


def delta_m(f1, map_, ref):
    """Δm in % of a run (macro-F1, mAP as fractions) against ref = (F1, mAP)."""
    return 100 * 0.5 * ((f1 - ref[0]) / ref[0] + (map_ - ref[1]) / ref[1])


def load(paths):
    """Every run under `paths`: its metrics.json, plus fusion_val.json when evaluate.py wrote one."""
    runs = []
    for path in paths:
        for file in sorted([path] if path.is_file() else path.rglob('metrics.json')):
            run = json.loads(file.read_text())
            fusion_file = file.parent / 'fusion_val.json'
            if 'val_by_fusion' not in run and fusion_file.exists():
                run['val_by_fusion'] = {k: v for k, v in json.loads(fusion_file.read_text()).items() if k != 'data'}
            run['path'] = str(file.parent)
            runs.append(run)
    return runs


def config(run):
    """(label, data, epochs, limit): runs with the same key are seeds of one configuration."""
    a = run['args']
    if run.get('design') == 'detail_lora':
        label = NAMES.get(run['config'], run['config'])
    else:
        label = f"{a['backbone']} + {a['adapt']}" + (' (scratch)' if a['init'] == 'scratch' else '')
    return label, Path(str(a['data'])).name, a['epochs'], a.get('limit')


def is_reference(run):
    a = run['args']
    return run.get('design') is None and (a['backbone'], a['init'], a['adapt']) == ('vit', 'pretrained', 'lora')


def order(key, reference_first=False, run=None):
    """Sort groups by data, epochs, --limit (None = all images), then reference first, then label."""
    label, data, epochs, limit = key
    return data, epochs, limit or 0, reference_first and not is_reference(run), label


def data_label(key):
    """'holdout', plus the epochs and --limit when they differ from a real run."""
    label, data, epochs, limit = key
    return data + (f', {epochs} ep' if epochs != 24 else '') + (f', limit {limit}' if limit else '')


def mean_std(values, digits=1):
    mean = round(statistics.mean(values), digits) + 0.0  # + 0.0: no "-0.0"
    if len(values) == 1:
        return f'{mean:.{digits}f}'
    return f'{mean:.{digits}f} ± {statistics.stdev(values):.{digits}f}'


def tables(runs, ref=None):
    """The markdown report. ref: (F1, mAP) as fractions to use instead of the local reference runs."""
    groups = defaultdict(list)
    for run in runs:
        groups[config(run)].append(run)
    refs = {}  # (data, epochs, limit) -> mean (F1, mAP) of the local ViT + LoRA seeds
    for key, members in groups.items():
        if is_reference(members[0]):
            refs[key[1:]] = (statistics.mean(r['val']['macro_f1'] for r in members),
                             statistics.mean(r['val']['mAP'] for r in members))
    out = []
    for (label, data, epochs, limit), members in groups.items():
        seeds = [r['args']['seed'] for r in members]
        if len(set(seeds)) < len(seeds):
            out.append(f'> warning: {label} on {data} has repeated seeds {seeds}: {[r["path"] for r in members]}')

    def reference_for(key):
        return ref or refs.get(key[1:])

    out += ['| config | data | seeds | total (M) | trainable (M) | accuracy | macro-F1 | mAP | AP50 | Δm | train h | '
            'peak GB | ms/img (b16 / b1) |', '|' + '---|' * 2 + '---:|' * 11]
    for key, members in sorted(groups.items(), key=lambda kv: order(kv[0], True, kv[1][0])):
        label, data, epochs, limit = key
        val = [r['val'] for r in members]
        r0 = members[0]
        reference = reference_for(key)
        dm = mean_std([delta_m(v['macro_f1'], v['mAP'], reference) for v in val]) if reference else '—'
        timing = [r['inference_ms_per_image'] for r in members if 'inference_ms_per_image' in r]
        ms = f"{statistics.mean(t['batch16'] for t in timing):.1f} / {statistics.mean(t['batch1'] for t in timing):.1f}" \
            if timing else '—'
        out.append(f"| {label} | {data_label(key)} | {len(members)} | {r0['parameters'] / 1e6:.2f} | "
                   f"{r0['trainable_parameters'] / 1e6:.2f} | "
                   + ' | '.join(mean_std([100 * v[k] for v in val]) for k in ('accuracy', 'macro_f1', 'mAP', 'AP50'))
                   + f" | {dm} | {statistics.mean(r['train_hours'] for r in members):.2f} | "
                   f"{statistics.mean(r['peak_gpu_memory_gb'] for r in members):.1f} | {ms} |")
    missing = sorted({data_label(key) for key in groups if not reference_for(key)})
    if missing:
        out.append(f'\nNo local ViT + LoRA reference for: {"; ".join(missing)}. Δm shown as —. Train it on the '
                   'same data (README step 3) or pass --ref F1 MAP.')
    elif ref:
        out.append(f'\nΔm against --ref macro-F1 {100 * ref[0]:.2f}, mAP {100 * ref[1]:.2f}.')

    # Ablation A2: the same checkpoints under every detection scoring
    scored = [(key, members) for key, members in groups.items() if all('val_by_fusion' in r for r in members)]
    if scored:
        out += ['', 'Ablation A2, detection scoring (same checkpoints; Δm against the unchanged reference):', '',
                '| config | data | scoring | macro-F1 | mAP | AP50 | Δm |', '|---|---|---|---:|---:|---:|---:|']
        for key, members in sorted(scored, key=lambda kv: order(kv[0])):
            reference = reference_for(key)
            for fusion in ('none', 'hard', 'soft'):
                val = [r['val_by_fusion'][fusion] for r in members]
                dm = mean_std([delta_m(v['macro_f1'], v['mAP'], reference) for v in val]) if reference else '—'
                out.append(f'| {key[0]} | {data_label(key)} | {fusion} | '
                           + ' | '.join(mean_std([100 * v[k] for v in val]) for k in ('macro_f1', 'mAP', 'AP50'))
                           + f' | {dm} |')

    # Per-class AP: where the mAP changes come from
    out += ['', 'Per-class AP (%), mean over seeds, as reported:', '',
            '| config | data | ' + ' | '.join(CLASSES) + ' |', '|---|---|' + '---:|' * len(CLASSES)]
    for key, members in sorted(groups.items(), key=lambda kv: order(kv[0])):
        aps = [[100 * r['val']['per_class_AP'][c] for r in members] for c in CLASSES]  # -100: class absent
        out.append(f'| {key[0]} | {data_label(key)} | '
                   + ' | '.join(f'{statistics.mean(ap):.1f}' if min(ap) >= 0 else '—' for ap in aps) + ' |')

    hours = sum(r['train_hours'] for r in runs)
    out += ['', f'{len(runs)} runs, {hours:.2f} GPU hours of training in total (train_hours: epoch loop with '
                'validation; excludes model loading and the final scoring).']
    return '\n'.join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('paths', type=Path, nargs='+', help='run directories or folders of runs (searched recursively)')
    p.add_argument('--ref', type=float, nargs=2, metavar=('F1', 'MAP'),
                   help='reference macro-F1 and mAP in %% instead of the local ViT + LoRA runs, e.g. the '
                        'leaderboard val reference 93.31 26.13 (only meaningful for Codabench scores)')
    args = p.parse_args(argv)
    runs = load(args.paths)
    if not runs:
        sys.exit(f'no metrics.json under {[str(p) for p in args.paths]}')
    print(tables(runs, tuple(v / 100 for v in args.ref) if args.ref else None))


if __name__ == '__main__':
    main()
