"""Failure analysis of finished runs on ND (the holdout): per-class AP and COCO small / medium / large AP; the same by the
near-duplicate partner each ND image has in local train; with two runs, a paired bootstrap of their Δm difference.
- Partner (holdout_sarmtl's links, near_dup_links.csv + audit_links.csv, from the ND image to a local-train image),
  strongest kind first: copy (the same image, maybe flipped or turned), crop (a shifted or overlapping tile: shift,
  nshift, overlap, audit), revisit (another acquisition: revisit, georevisit), related, none. Then by orientation:
  same (a partner with k = 0, the only orientation training without flips shows), turned (partners only flipped or
  turned), unknown (only overlap links, which have no k), none.
- Bootstrap: resamples near-duplicate units (connected components of the links among ND images), so the images of one
  scene move together. Scoring and bootstrap follow FitDet's workspace/fitdet/analyze.py.
--dump (GPU, through gpu_run.sh) first saves each run's per-image ND outputs under one decoding (default: the run's best
in decode_eval.json) to <run>/outputs_nd.json; the rest is CPU only.
Examples (from the repo root):
  workspace/gpu_run.sh --min-free 8000 -- uv run python workspace/mtdet/analyze.py --dump workspace/mtdet/runs/Y_s0
  uv run python workspace/mtdet/analyze.py workspace/mtdet/runs/Y_s0 workspace/mtdet/runs/YP2_s0 --bootstrap 200"""
import argparse
import contextlib
import csv
import io
import json
import random
from collections import defaultdict
from pathlib import Path

from common import HOLDOUT, ROOT, classification_metrics, delta_m, split_images

LINKS = [ROOT / 'dataset/holdout_sarmtl/analysis/near_dup_links.csv', ROOT / 'workspace/sarmtl/data/audit_links.csv']
KIND = {'copy': 'copy', 'shift': 'crop', 'nshift': 'crop', 'overlap': 'crop', 'audit': 'crop', 'revisit': 'revisit',
        'georevisit': 'revisit', 'related': 'related'}
ORDER = ['copy', 'crop', 'revisit', 'related', 'none']


def dump(run, decoding=None):
    """The run's ND outputs (labels, class probabilities, detections in original pixels) under one decoding."""
    import torch
    from common import make_loader, to_original_xywh
    from evaluate import load_run
    model, args = load_run(run)
    data = ROOT / args['data']
    if split_images(data, 'train') & split_images(data, 'val'):
        raise SystemExit(f'refused: {run} trained on its ND images')
    if decoding is None:
        sweep = json.loads((run / 'decode_eval.json').read_text())
        decoding = max(sweep, key=lambda name: sweep[name]['delta_m'])
    model.apply_decoding(model.decodings()[decoding])
    images, detections = [], []
    with torch.no_grad():
        for x, y, targets in make_loader(data, 'val'):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, outputs, _ = model(x.cuda(non_blocking=True))
            for label, probs, t, out in zip(y.tolist(), logits.float().softmax(1).tolist(), targets, outputs):
                images.append({'image_id': t['image_id'], 'label': label, 'probs': probs})
                xywh = to_original_xywh(out['boxes'].float().cpu(), t['orig_size']).tolist()
                detections += [{'image_id': t['image_id'], 'category_id': int(c), 'bbox': b, 'score': float(s)}
                               for b, c, s in zip(xywh, out['labels'].tolist(), out['scores'].tolist())]
    out = {'split': str(data / 'val'), 'decoding': decoding, 'images': images, 'detections': detections}
    (run / 'outputs_nd.json').write_text(json.dumps(out))


def coco(gt, dets, ids):
    """pycocotools on GT dict gt, restricted to image ids: (stats in %, per-class AP in %)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO()
        g.dataset = gt
        g.createIndex()
        ev = COCOeval(g, g.loadRes([dict(d) for d in dets]) if dets else COCO(), 'bbox')
        ev.params.imgIds = ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    prec = ev.eval['precision'][:, :, :, 0, -1]
    return [100 * s for s in ev.stats], {g.cats[c]['name']: 100 * float(prec[:, :, k][prec[:, :, k] > -1].mean())
                                         if (prec[:, :, k] > -1).any() else float('nan')
                                         for k, c in enumerate(ev.params.catIds)}


def scores(out, gt, ids):
    """(true labels, predicted labels, COCO stats %, per-class AP %) on the multiset of image ids (repeats
    re-indexed)."""
    by_id = {im['image_id']: im for im in out['images']}
    image, anns, dets = {im['id']: im for im in gt['images']}, defaultdict(list), defaultdict(list)
    for a in gt['annotations']:
        anns[a['image_id']].append(a)
    for d in out['detections']:
        dets[d['image_id']].append(d)
    g = {'images': [], 'annotations': [], 'categories': gt['categories']}
    flat, labels, preds = [], [], []
    for new, old in enumerate(ids, 1):
        g['images'].append({**image[old], 'id': new})
        g['annotations'] += [{**a, 'id': len(g['annotations']) + k + 1, 'image_id': new} for k, a in enumerate(anns[old])]
        flat += [{**d, 'image_id': new} for d in dets[old]]
        probs = by_id[old]['probs']
        labels.append(by_id[old]['label'])
        preds.append(probs.index(max(probs)))
    return (labels, preds, *coco(g, flat, list(range(1, len(ids) + 1))))


def links(keep):
    """(a, b, kind, k) of every link with both ends in keep; k is the dihedral transform ('' for overlap links)."""
    for path in LINKS:
        with open(path) as f:
            yield from ((int(r['a']), int(r['b']), r['kind'], r['k']) for r in csv.DictReader(f)
                        if int(r['a']) in keep and int(r['b']) in keep)


def partner_kinds(nd, train):
    """{ND image id: its strongest partner kind in local train}."""
    best = {i: 'none' for i in nd}
    for a, b, kind, _ in links(nd | train):
        for x, y in ((a, b), (b, a)):
            if x in nd and y in train and ORDER.index(KIND[kind]) < ORDER.index(best[x]):
                best[x] = KIND[kind]
    return best


def orientations(nd, train):
    """{ND image id: same / turned / unknown / none}, over its partners of every kind in local train."""
    ks = defaultdict(set)
    for a, b, _, k in links(nd | train):
        for x, y in ((a, b), (b, a)):
            if x in nd and y in train:
                ks[x].add(k)
    return {i: 'none' if not ks[i] else 'same' if '0' in ks[i] else 'unknown' if ks[i] == {''} else 'turned'
            for i in nd}


def units(ids):
    """Connected components of the links among ids."""
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b, _, _ in links(set(ids)):
        parent[find(a)] = find(b)
    groups = defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    return list(groups.values())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', type=Path, nargs='+')
    p.add_argument('--dump', action='store_true', help="first save each run's ND outputs (GPU)")
    p.add_argument('--decoding', help="the decoding --dump uses (default: the run's best in decode_eval.json)")
    p.add_argument('--bootstrap', type=int, default=0, help='paired bootstrap draws of runs[1] - runs[0] (0: none)')
    args = p.parse_args()
    if args.dump:
        for run in args.runs:
            dump(run, args.decoding)
    outs = [json.loads((run / 'outputs_nd.json').read_text()) for run in args.runs]
    gt = json.loads((HOLDOUT / 'val' / 'instances.json').read_text())
    ids = [im['image_id'] for im in outs[0]['images']]
    with open(HOLDOUT / 'train' / 'labels.csv') as f:
        train = {int(r['image_id']) for r in csv.DictReader(f)}
    groups = [(partner_kinds(set(ids), train), ORDER), (orientations(set(ids), train), ['same', 'turned', 'unknown', 'none'])]
    for run, out in zip(args.runs, outs):
        labels, preds, stats, per_class = scores(out, gt, ids)
        f1 = 100 * classification_metrics(labels, preds)['macro_f1']
        print(f"{run.name} ({out['decoding']}): F1 {f1:.2f} mAP {stats[0]:.2f} Δm {delta_m(f1 / 100, stats[0] / 100):+.2f}"
              f" | AP50 {stats[1]:.1f} AP75 {stats[2]:.1f} APs {stats[3]:.1f} APm {stats[4]:.1f} APl {stats[5]:.1f}")
        print('  all      ' + ' '.join(f'{name[:5]} {ap:5.1f}' for name, ap in per_class.items()))
        for group, order in groups:  # by partner kind, then orientation: accuracy (a subset lacks classes, so no
            for k in order:            # macro-F1) and AP
                sub = [i for i in ids if group[i] == k]
                if not sub:
                    continue
                labels, preds, stats, per_class = scores(out, gt, sub)
                print(f'  {k:8s} {len(sub):3d} images: accuracy {100 * sum(map(int.__eq__, labels, preds)) / len(sub):5.1f}'
                      f' mAP {stats[0]:5.1f} AP50 {stats[1]:5.1f} | ' + ' '.join(f'{name[:5]} {ap:5.1f}'
                                                                             for name, ap in per_class.items()))
    if args.bootstrap and len(outs) >= 2:
        assert ids == [im['image_id'] for im in outs[1]['images']], 'the two runs scored different images'
        groups, rng, diffs = units(ids), random.Random(0), []

        def dm(out, draw):
            labels, preds, stats, _ = scores(out, gt, draw)
            return delta_m(classification_metrics(labels, preds)['macro_f1'], stats[0] / 100)
        for _ in range(args.bootstrap):
            draw = [i for g in rng.choices(groups, k=len(groups)) for i in g]
            diffs.append(dm(outs[1], draw) - dm(outs[0], draw))
        diffs.sort()
        lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
        print(f'paired unit bootstrap ({len(groups)} units, {args.bootstrap} draws): Δm({args.runs[1].name}) - '
              f'Δm({args.runs[0].name}) mean {sum(diffs) / len(diffs):+.2f}, 95% interval [{lo:+.2f}, {hi:+.2f}]')


if __name__ == '__main__':
    main()
