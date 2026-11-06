"""Build dataset/repro/ so that train.py runs unchanged on all of train, as the reference runs did.
train: all of dataset/train. val and test: the same per-class sample of train (--fraction), because the real val
and test labels are withheld and train.py scores both; those scores measure the fit on training images, not
generalisation. Judge a model by its val submission on Codabench (predict.py). Files are linked, not copied."""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[2] / 'dataset')
p.add_argument('--fraction', type=float, default=0.05, help='of every class of train, used as val and test')
p.add_argument('--seed', type=int, default=0)
args = p.parse_args()

out = args.data / 'repro'
(out / 'val').mkdir(parents=True, exist_ok=True)
for link, target in [(out / 'train', '../train'), (out / 'test', 'val'), (out / 'val' / 'images', '../../train/images')]:
    if not link.is_symlink():
        link.symlink_to(target)

with open(args.data / 'train' / 'labels.csv') as f:
    rows = list(csv.DictReader(f))
by_class = defaultdict(list)
for row in rows:
    by_class[row['label']].append(row)
rng = random.Random(args.seed)
sample = sorted((row for label in sorted(by_class) for row in
                 rng.sample(by_class[label], round(args.fraction * len(by_class[label])))),
                key=lambda row: int(row['image_id']))
ids = {int(row['image_id']) for row in sample}

with open(out / 'val' / 'labels.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['image_id', 'file_name', 'label'])
    writer.writeheader()
    writer.writerows(sample)
coco = json.loads((args.data / 'train' / 'instances.json').read_text())
coco['images'] = [img for img in coco['images'] if img['id'] in ids]
coco['annotations'] = [ann for ann in coco['annotations'] if ann['image_id'] in ids]
(out / 'val' / 'instances.json').write_text(json.dumps(coco))
print(f'{out}: train {len(rows)} images, val = test = {len(sample)} train images, {len(coco["annotations"])} boxes')
