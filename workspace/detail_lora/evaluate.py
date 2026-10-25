"""Ablation A2: score trained runs on the holdout under every detection scoring (none, hard, soft), without
retraining. Works on reference runs of the top-level train.py (ViT + LoRA, TerraMind + MoE-LoRA) and on DetailLoRA
runs (whose train.py already does this). Writes <run>/fusion_val.json, which compare.py reads.
Example (repo root): CUDA_VISIBLE_DEVICES=1 uv run python workspace/detail_lora/evaluate.py workspace/detail_lora/runs/ref_seed*/*/"""
import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from common import ROOT, evaluate_fusions, load_run, reference_train
from sarbench.data import SARMultiTask, collate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', type=Path, nargs='+', help='run directories (metrics.json, model.pt)')
    p.add_argument('--data', type=Path, default=ROOT / 'dataset' / 'holdout')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    loader = DataLoader(SARMultiTask(args.data, 'val'), args.batch_size, num_workers=args.workers,
                        collate_fn=collate)
    for run in args.runs:
        model, metrics = load_run(run)
        trained_on = Path(metrics['args']['data']).resolve()
        if trained_on != args.data.resolve():
            print(f'skip {run}: trained on {trained_on}, not {args.data.resolve()} (its val split is not a holdout)')
            continue
        results = evaluate_fusions(model, loader, args.data / 'val' / 'instances.json')
        (run / 'fusion_val.json').write_text(json.dumps({'data': str(args.data), **results}, indent=2))
        for fusion, m in results.items():
            print(f'{run} scoring {fusion}: {reference_train.headline(m)}')


if __name__ == '__main__':
    main()
