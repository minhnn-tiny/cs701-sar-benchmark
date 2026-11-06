"""compare.py: Δm per seed against the local reference, mean ± std, grouping."""
import json
import statistics

import pytest

from compare import delta_m, load, tables


def write_run(folder, seed, f1, map_, design=False, data='dataset/holdout', limit=None):
    folder.mkdir(parents=True)
    args = {'data': data, 'epochs': 24, 'limit': limit, 'seed': seed}
    if design:
        run = {'design': 'detail_lora', 'config': 'detaillora', 'inference_ms_per_image': {'batch16': 9.0, 'batch1': 30.0}}
        args.update(rank=32)
    else:
        run = {}
        args.update(backbone='vit', init='pretrained', adapt='lora')
    val = {'accuracy': f1, 'macro_f1': f1, 'mAP': map_, 'AP50': 2 * map_, 'per_class_AP': {
        c: map_ for c in ['aircraft', 'airport', 'bridge', 'car', 'harbor', 'oil_tank', 'playground', 'ship', 'wind_turbine']}}
    run.update(args=args, parameters=111e6, trainable_parameters=26e6, train_hours=1.0, peak_gpu_memory_gb=15.0, val=val)
    if design:
        run['val_by_fusion'] = {f: val for f in ('none', 'hard', 'soft')}
    (folder / 'metrics.json').write_text(json.dumps(run))


def test_delta_m_matches_the_definition():
    assert delta_m(0.9331, 0.2613, (0.9331, 0.2613)) == 0
    assert delta_m(0.9331 * 1.02, 0.2613 * 1.10, (0.9331, 0.2613)) == pytest.approx(6.0)


def test_table_reports_delta_m_per_seed_against_the_local_reference(tmp_path):
    ref = [(0.93, 0.26), (0.94, 0.25), (0.92, 0.27)]
    new = [(0.95, 0.30), (0.96, 0.31), (0.94, 0.29)]
    for s, (f1, m) in enumerate(ref):
        write_run(tmp_path / f'ref_seed{s}' / 'vit_pretrained_lora', s, f1, m)
    for s, (f1, m) in enumerate(new):
        write_run(tmp_path / f'seed{s}' / 'detaillora', s, f1, m, design=True)
    write_run(tmp_path / 'smoke' / 'detaillora', 0, 0.1, 0.01, design=True, limit=64)  # its own group, no reference
    out = tables(load([tmp_path]))
    base = (statistics.mean(f for f, _ in ref), statistics.mean(m for _, m in ref))
    dms = [delta_m(f, m, base) for f, m in new]
    expected = f'{statistics.mean(dms):.1f} ± {statistics.stdev(dms):.1f}'
    row = next(line for line in out.splitlines() if line.startswith('| DetailLoRA (primary) | holdout | 3'))
    assert expected in row
    ref_row = next(line for line in out.splitlines() if line.startswith('| vit + lora | holdout | 3'))
    assert ' 0.0 ± ' in ref_row  # the reference against its own mean
    assert 'limit 64' in out and 'No local ViT + LoRA reference' in out
    assert 'Ablation A2' in out and '7 runs, 7.00 GPU hours' in out
