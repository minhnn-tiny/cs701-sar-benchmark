"""Build dataset/holdout/: a fixed, disjoint, class-stratified, duplicate-aware 10 % holdout of train.

Run from the repo root: uv run python workspace/detail_lora/make_holdout.py   (about 1 min)

How the holdout is drawn (the report states this):
1. Near-duplicate groups. Train holds the same scene several times (other acquisition or processing, other speckle,
   a few pixels of shift): every image is read as gray, resized to 64 x 64 (PIL bilinear) and standardised; two
   images whose Pearson correlation is > 0.85 are linked, and the connected components are the groups
   (1,011 groups of 2-8 images, 2,594 images). Each group goes wholly to one side, so the holdout has no copy of a
   training scene. Singletons are groups of one.
2. Strata: (class, image size) of each group (its majority label, and its first image's size: 512, 256, 224 or
   other), so the holdout keeps train's class and source mix.
3. In every stratum (sorted), the groups (sorted by smallest image id) are shuffled with random.Random(seed), one
   generator for the whole draw, and taken in that order while they fit: until round(fraction x images in the
   stratum) images are in the holdout.
The groups, the holdout ids and the per-class counts are saved in <out>/split.json, the record of the split.

Layout (both train.py and workspace/detail_lora/train.py read it with --data dataset/holdout):
  <out>/train/{labels.csv, instances.json, images -> ../../train/images}   the other 90 %
  <out>/val/{labels.csv, instances.json, images -> ../../train/images}     the holdout
  <out>/test -> val                                                         train.py also scores a test split
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def thumbnail(path):
    """64 x 64 gray thumbnail, zero mean and unit norm (so a dot product is a Pearson correlation)."""
    x = np.asarray(Image.open(path).convert('L').resize((64, 64), Image.BILINEAR), dtype=np.float64).ravel()
    x = x - x.mean()
    return x / max(np.linalg.norm(x), 1e-9)


def duplicate_groups(paths, threshold=0.85, processes=32):
    """Indices of `paths` grouped by near-duplicate links (correlation of thumbnails > threshold)."""
    with Pool(processes) as pool:
        thumbs = torch.from_numpy(np.stack(pool.map(thumbnail, paths, chunksize=32)))
    corr = thumbs @ thumbs.T
    corr.fill_diagonal_(0)
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in (corr > threshold).nonzero().tolist():
        parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(paths)):
        groups[find(i)].append(i)
    return sorted(sorted(g) for g in groups.values())


def size_bucket(image):
    """Image size as a proxy of the source dataset."""
    w, h = image['width'], image['height']
    return f'{w}' if w == h and w in (512, 256, 224) else 'other'


def draw_holdout(groups, labels, buckets, fraction=0.1, seed=0):
    """groups: lists of image ids; labels, buckets: id -> class, id -> size bucket.
    Returns the set of holdout ids; see the module docstring for the procedure."""
    strata = defaultdict(list)
    for group in sorted(sorted(g) for g in groups):
        label = Counter(labels[i] for i in group).most_common(1)[0][0]  # ties: the first one met
        strata[(label, buckets[group[0]])].append(group)
    rng, holdout = random.Random(seed), set()
    for key in sorted(strata):
        members = strata[key]
        target = round(fraction * sum(len(g) for g in members))
        rng.shuffle(members)
        taken = 0
        for group in members:
            if taken + len(group) <= target:
                holdout.update(group)
                taken += len(group)
    return holdout


def write_split(src, dst, ids):
    """<dst>/labels.csv and <dst>/instances.json restricted to `ids`; images linked to the train images."""
    dst.mkdir(parents=True, exist_ok=True)
    with open(src / 'labels.csv') as f:
        rows = [r for r in csv.DictReader(f) if int(r['image_id']) in ids]
    with open(dst / 'labels.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_id', 'file_name', 'label'])
        writer.writeheader()
        writer.writerows(rows)
    coco = json.loads((src / 'instances.json').read_text())
    coco['images'] = [img for img in coco['images'] if img['id'] in ids]
    coco['annotations'] = [a for a in coco['annotations'] if a['image_id'] in ids]
    (dst / 'instances.json').write_text(json.dumps(coco))
    link = dst / 'images'
    if not link.is_symlink():
        link.symlink_to('../../train/images')
    return len(rows), len(coco['annotations'])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=ROOT / 'dataset')
    p.add_argument('--out', type=Path, help='default: <data>/holdout')
    p.add_argument('--fraction', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threshold', type=float, default=0.85, help='thumbnail correlation that links two images')
    args = p.parse_args()
    out = args.out or args.data / 'holdout'
    src = args.data / 'train'

    with open(src / 'labels.csv') as f:
        rows = list(csv.DictReader(f))
    images = {img['id']: img for img in json.loads((src / 'instances.json').read_text())['images']}
    ids = [int(r['image_id']) for r in rows]
    labels = {int(r['image_id']): int(r['label']) for r in rows}
    buckets = {i: size_bucket(images[i]) for i in ids}

    groups = [[ids[k] for k in g] for g in duplicate_groups([src / r['file_name'] for r in rows], args.threshold)]
    holdout = draw_holdout(groups, labels, buckets, args.fraction, args.seed)
    train = set(ids) - holdout

    # Checks: disjoint, complete, no duplicate group split across the two sides
    assert not holdout & train and holdout | train == set(ids)
    assert all(set(g) <= holdout or set(g) <= train for g in groups)

    n_train = write_split(src, out / 'train', train)
    n_val = write_split(src, out / 'val', holdout)
    test = out / 'test'
    if not test.is_symlink():
        test.symlink_to('val')

    names = [c['name'] for c in json.loads((args.data / 'classes.json').read_text())]
    per_class = {names[c]: {'train': sum(labels[i] == c for i in train), 'holdout': sum(labels[i] == c for i in holdout)}
                 for c in range(len(names))}
    record = {
        'method': 'duplicate-aware, (class, image size)-stratified random draw; see make_holdout.py',
        'fraction': args.fraction, 'seed': args.seed, 'threshold': args.threshold,
        'source_sha256': {f: sha256(src / f) for f in ('labels.csv', 'instances.json')},
        'images': {'train': n_train[0], 'holdout': n_val[0]}, 'boxes': {'train': n_train[1], 'holdout': n_val[1]},
        'per_class': per_class,
        'duplicate_groups': [g for g in groups if len(g) > 1],
        'holdout_ids': sorted(holdout),
    }
    (out / 'split.json').write_text(json.dumps(record))
    print(f'{out}: train {n_train[0]} images / {n_train[1]} boxes, holdout {n_val[0]} images / {n_val[1]} boxes, '
          f'{len(record["duplicate_groups"])} duplicate groups kept whole')
    for name, n in per_class.items():
        print(f'  {name:13s} train {n["train"]:5d}  holdout {n["holdout"]:4d}  '
              f'({100 * n["holdout"] / (n["train"] + n["holdout"]):.1f} %)')


if __name__ == '__main__':
    main()
