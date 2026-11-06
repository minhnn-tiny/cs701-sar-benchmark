"""Model soup (Wortsman et al., ICML 2022): the weights of several finished runs of one recipe, averaged into ONE model.
Inference stays one forward pass of one model, like any single run (no ensemble, no test-time augmentation). Averaged
weights no longer match their batch-norm running statistics, so those are recomputed on clean training images.
  soup.py RUN [RUN ...] --out DIR [--bn-batches 200]   (RUN: a run directory, or a weights file such as swa_012.pt;
  --bn-batches 0: keep the averaged running statistics)
Seeds trained apart do NOT soup: three seeds of one recipe averaged scored holdout -35 to -47 (31 Oct), as they end
in different basins. Weights from one trajectory do (SWA: train.py --init-from RUN/model.pt --swa N).
DIR gets model.pt and metrics.json (the first run's, with a 'soup' record), so evaluate.py and predict.py take it like
a run directory.
Example: workspace/gpu_run.sh --min-free 10000 -- uv run python workspace/mtdet/soup.py workspace/mtdet/runs/YSA450_s{0,1,2}
         --out workspace/mtdet/runs/soup_YSA450"""
import argparse
import json
from pathlib import Path

import torch

from common import ROOT, make_loader, to_unit_rgb
from evaluate import load_run


def average(states):
    """The mean of state dicts with the same keys; integer buffers (num_batches_tracked) are taken from the first."""
    return {k: (sum(s[k].float() for s in states) / len(states)).to(v.dtype) if v.is_floating_point() else v
            for k, v in states[0].items()}


@torch.no_grad()
def recalibrate_bn(model, loader, batches):
    """Batch-norm running statistics recomputed as plain averages over `batches` training batches."""
    norms = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # cumulative average
    model.train()
    for i, (images, _, _) in enumerate(loader):
        if i == batches:
            break
        with torch.autocast('cuda', dtype=torch.bfloat16):
            model.det(to_unit_rgb(images.cuda(non_blocking=True)))
        model.features()  # drops the feature the classification hook kept (the classifier head has no batch norm)
    model.eval()
    return len(norms)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', type=Path, nargs='+')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bn-batches', type=int, default=200)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    dirs = [r if r.is_dir() else r.parent for r in args.runs]  # a run directory (its model.pt) or a weights file in one
    metrics = [json.loads((d / 'metrics.json').read_text()) for d in dirs]
    for key in ('arch', 'cls_head', 'data'):
        assert len({m['args'].get(key) for m in metrics}) == 1, f'the runs differ in {key}: not one recipe'
    model, run_args = load_run(dirs[0])
    soup = average([torch.load(r / 'model.pt' if r.is_dir() else r, map_location='cpu') for r in args.runs])
    model.load_state_dict(soup)
    # a shuffled pass over training images (train=True: seeded order and flips only, no SAR noise), so every class is seen;
    # --bn-batches 0 keeps the ingredients' averaged running statistics instead
    n = recalibrate_bn(model, make_loader(ROOT / run_args['data'], 'train', train=True, batch_size=16,
                                          workers=args.workers), args.bn_batches) if args.bn_batches else 0
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / 'model.pt')
    record = {**metrics[0], 'args': {**metrics[0]['args'], 'out': str(args.out)},
              'soup': {'runs': [str(r) for r in args.runs], 'bn_batches': args.bn_batches, 'bn_layers': n}}
    (args.out / 'metrics.json').write_text(json.dumps(record, indent=1, default=str))
    print(f'wrote {args.out}: the mean of {len(args.runs)} runs, {n} batch-norm layers recomputed on '
          f'{args.bn_batches} training batches')


if __name__ == '__main__':
    main()
