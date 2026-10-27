"""Refit a finished run's image classification head on its frozen trunk; the detector and its weights stay as they are.
Question (27 Oct): YOLO's classifier misses crops of training scenes that FitDet's gets right (ND images with a copy or
crop partner in local train: 96.5-99.1 % accurate against FitDet's 98.8-100 %, mostly oil tanks called playground with
p > 0.9), and zoom, mosaic and the pyramid head each raised ND F1 by ~1.1 in joint training while costing mAP. Does a
head refit on frozen features, from zoomed views of the training images or from the P3-P5 pyramid, classify better
without touching detection?
- default: features of the run's train images (PASSES passes with flips, PASSES with flips + Zoom 0.5), ND and LF; one
  head per VARIANTS entry, LayerNorm + Linear like the joint head; macro-F1 and accuracy on ND and LF -> probe.json.
- --write VARIANT --out DIR: fit that variant and save DIR (metrics.json with its cls_head, model.pt with the new head
  and the run's other weights) for predict.py.
Example: workspace/gpu_run.sh --min-free 8000 -- uv run python workspace/mtdet/probe.py workspace/mtdet/runs/Y250_s0"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from common import ROOT, classification_metrics, eval_dirs, make_loader, split_images, to_unit_rgb
from evaluate import load_run
from ultra_model import _keep_output

VARIANTS = {'p5_flips': ('p5', 0.0), 'p5_zoom': ('p5', 0.5), 'pyramid_flips': ('pyramid', 0.0),
            'pyramid_zoom': ('pyramid', 0.5)}  # name: (features, Zoom spread of the training views)
PASSES = 4


@torch.no_grad()
def features(model, loader, passes=1):
    """({'p5': [N, C] layer-10 means, 'pyramid': [N, 2 x (C3 + C4 + C5)] P3-P5 means and maxima}, labels [N])."""
    pyramid = model.det.model[-1].f
    feats, labels = {'p5': [], 'pyramid': []}, []
    for p in range(passes):
        if passes > 1:
            loader.sampler.epoch = p
        for images, y, _ in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model.det(to_unit_rgb(images.cuda(non_blocking=True)))
            feats['p5'].append(model.features().cpu())
            pooled = []
            for i in pyramid:
                kept = model.det.model[i].kept.float()
                del model.det.model[i].kept
                pooled += [kept.mean(dim=(2, 3)), kept.amax(dim=(2, 3))]
            feats['pyramid'].append(torch.cat(pooled, 1).cpu())  # the order of UltraMultiTask(cls_head='pyramid')
            labels.append(y)
    return {k: torch.cat(v) for k, v in feats.items()}, torch.cat(labels)


def fit(x, y, num_classes=9, epochs=20, lr=1e-3, weight_decay=0.05, seed=0):
    """A LayerNorm + Linear head (the joint head's shape), fit with AdamW on cached features."""
    torch.manual_seed(seed)
    head = nn.Sequential(nn.LayerNorm(x.shape[1]), nn.Linear(x.shape[1], num_classes)).to(x.device)
    opt = torch.optim.AdamW(head.parameters(), lr, weight_decay=weight_decay)
    order = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(x), generator=order).split(256):
            loss = nn.functional.cross_entropy(head(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head.eval()


@torch.no_grad()
def score(head, x, y):
    m = classification_metrics(y.tolist(), head(x).argmax(1).tolist())
    return {'macro_f1': 100 * m['macro_f1'], 'accuracy': 100 * m['accuracy']}


def with_head(state, head):
    """The run's state dict with its classification head replaced."""
    return {**{k: v for k, v in state.items() if not k.startswith('cls_head.')},
            **{f'cls_head.{k}': v.cpu() for k, v in head.state_dict().items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    p.add_argument('--write', choices=VARIANTS, help='fit this variant and save it as a run directory (--out)')
    p.add_argument('--out', type=Path)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    assert not args.write or args.out, '--write needs --out'
    model, run_args = load_run(args.run)
    assert run_args.get('cls_head', 'p5') == 'p5', 'probe.py refits the default head of a p5 run'
    for i in model.det.model[-1].f:
        model.det.model[i].register_forward_hook(_keep_output)
    data = ROOT / run_args['data']
    variants = {args.write: VARIANTS[args.write]} if args.write else VARIANTS
    train = {}
    for zoom in sorted({z for _, z in variants.values()}):
        loader = make_loader(data, 'train', train=True, zoom=zoom, batch_size=32, workers=args.workers,
                             seed=run_args['seed'])
        train[zoom] = features(model, loader, PASSES)
    heads = {name: fit(train[zoom][0][kind].cuda(), train[zoom][1].cuda()) for name, (kind, zoom) in variants.items()}
    if args.write:
        kind = VARIANTS[args.write][0]
        args.out.mkdir(parents=True, exist_ok=True)
        metrics = json.loads((args.run / 'metrics.json').read_text())
        metrics['args'] = {**metrics['args'], 'cls_head': kind, 'out': str(args.out)}
        metrics['probe'] = {'from': str(args.run), 'variant': args.write, 'passes': PASSES}
        torch.save(with_head(torch.load(args.run / 'model.pt', map_location='cpu'), heads[args.write]),
                   args.out / 'model.pt')
        (args.out / 'metrics.json').write_text(json.dumps(metrics, indent=1))
        print(f'wrote {args.out}: {args.write} head on the detector of {args.run}')
        return
    results, seen = {'passes': PASSES}, split_images(data, 'train')
    for name, d in eval_dirs(data, ['nd', 'lf']).items():
        if seen & split_images(d.parent, d.name):
            print(f"{name}: skipped (its images are in the run's train split)")
            continue
        feats, y = features(model, make_loader(d.parent, d.name, batch_size=32, workers=args.workers))
        rows = {'joint': score(model.cls_head, feats['p5'].cuda(), y.cuda())}  # the run's own head, for reference
        rows.update({v: score(heads[v], feats[kind].cuda(), y.cuda()) for v, (kind, _) in variants.items()})
        for v, r in rows.items():
            results.setdefault(v, {})[name] = r
            print(f"{name} {v:14s} F1 {r['macro_f1']:.2f} accuracy {r['accuracy']:.2f}")
    (args.run / 'probe.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
