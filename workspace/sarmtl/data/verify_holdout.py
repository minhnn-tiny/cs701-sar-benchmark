"""Check on the CPU that the unchanged sarbench.data.SARMultiTask loads every split of dataset/holdout_sarmtl/:
length, a few items in detail, then one full pass over every item (every image decodes, every box survives the
resize, labels are consistent), and the COCO scoring of root train.py on val/test (ground truth as predictions -> mAP
1). No GPU, no training.
Usage (repo root): uv run python workspace/sarmtl/data/verify_holdout.py [--workers 8]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sarbench.data import SARMultiTask, collate  # noqa: E402
from sarbench.metrics import detection_metrics  # noqa: E402

DATA = ROOT / 'dataset' / 'holdout_sarmtl'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    torch.manual_seed(0)
    for split in ('train', 'val', 'test'):
        ds = SARMultiTask(DATA, split, train=split == 'train')
        with open(DATA / split / 'labels.csv') as f:
            n_rows = sum(1 for _ in csv.DictReader(f))
        n_boxes = len(json.loads((DATA / split / 'instances.json').read_text())['annotations'])
        assert len(ds) == n_rows, (split, len(ds), n_rows)
        for i in (0, len(ds) // 2, len(ds) - 1):
            image, label, target = ds[i]
            print(f'{split}[{i}]: image {tuple(image.shape)} {image.dtype} mean {image.mean():+.3f}, label {label}, '
                  f'{len(target["boxes"])} boxes, labels {sorted(set(target["labels"].tolist()))}, '
                  f'image_id {target["image_id"]}, orig_size {target["orig_size"]}')
        seen, boxes = 0, 0
        loader = DataLoader(ds, batch_size=32, num_workers=args.workers, collate_fn=collate)
        for images, labels, targets in loader:
            assert images.shape[1:] == (1, 512, 512)
            for y, t in zip(labels.tolist(), targets):
                b = t['boxes']
                assert len(b) >= 1 and (t['labels'] == y + 1).all()
                assert (b[:, 0] >= -1e-3).all() and (b[:, 1] >= -1e-3).all() and (b[:, 2] <= 512 + 1e-3).all() \
                    and (b[:, 3] <= 512 + 1e-3).all() and (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()
                boxes += len(b)
            seen += len(images)
        assert seen == len(ds) and boxes == n_boxes, (split, seen, boxes, n_boxes)
        print(f'{split}: {len(ds)} images, {boxes} boxes, all items load (full pass)')
        if split != 'train':  # the COCO path of root train.py: the ground truth scored as predictions gives mAP 1
            gt = json.loads((DATA / split / 'instances.json').read_text())
            preds = [{'image_id': a['image_id'], 'category_id': a['category_id'], 'bbox': a['bbox'], 'score': 1.0}
                     for a in gt['annotations']]
            m = detection_metrics(DATA / split / 'instances.json', preds, [img['id'] for img in gt['images']])
            assert abs(m['mAP'] - 1) < 1e-6 and abs(m['AP50'] - 1) < 1e-6, m
            print(f'{split}: sarbench.metrics on the ground truth as predictions: mAP {m["mAP"]:.4f}, AP50 {m["AP50"]:.4f}')


if __name__ == '__main__':
    main()
