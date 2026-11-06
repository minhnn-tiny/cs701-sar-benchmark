"""The decoding sweep: score a finished run on its holdout (<data>/val, ND) under every decoding of its architecture
(model.decodings()) and write <run>/decode_eval.json. The decoding of a configuration is chosen on its 3-seed mean ND Δm
(compare.py). Refuses a run that trained on any ND image.
Example: workspace/gpu_run.sh --min-free 8000 -- uv run python workspace/mtdet/evaluate.py workspace/mtdet/runs/Y_s0"""
import argparse
import json
from pathlib import Path

import torch

from common import ROOT, evaluate, make_loader, split_images
from train import build_model, parse_args as train_args


def load_run(run):
    """The run's model with its scored weights (model.pt), on the GPU in eval mode, and its training args."""
    args = json.loads((run / 'metrics.json').read_text())['args']
    model = build_model(train_args(['--arch', args['arch'], '--cls-head', args.get('cls_head', 'p5'), '--out', str(run)]),
                        coco=False)
    model.load_state_dict(torch.load(run / 'model.pt', map_location='cpu'))
    return model.cuda().eval(), args


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    model, run_args = load_run(args.run)
    data = Path(run_args['data'])
    overlap = split_images(ROOT / data, 'train') & split_images(ROOT / data, 'val')
    if overlap:
        raise SystemExit(f'refused: {data}/train and {data}/val share {len(overlap)} images (e.g. dataset/repro)')
    loader = make_loader(ROOT / data, 'val', batch_size=args.batch_size, workers=args.workers)
    results = {}
    for name, cfg in model.decodings().items():
        model.apply_decoding(cfg)
        metrics, _ = evaluate(model, loader, ROOT / data / 'val' / 'instances.json')
        results[name] = {'decoding': cfg, **{k: v for k, v in metrics.items() if k != 'confusion_matrix'}}
        print(f"{name:24s} macro_f1 {metrics['macro_f1']:.4f} mAP {metrics['mAP']:.4f} AP50 {metrics['AP50']:.4f} "
              f"delta_m {metrics['delta_m']:+.2f}", flush=True)
    (args.run / 'decode_eval.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
