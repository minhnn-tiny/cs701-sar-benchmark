"""Two box facts for the detection candidates (train split, boxes at the 512 x 512 input), CPU only:
1. localisation tolerance: which share of boxes fails an IoU threshold after a d-px error in both x and y;
2. per class: image sizes and boxes per image.
Run from the repo root:  uv run python workspace/sarmtl/scratch/cv-detect/box_facts.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def main():
    coco = json.loads((ROOT / 'dataset/train/instances.json').read_text())
    names = {c['id']: c['name'] for c in coco['categories']}
    images = {im['id']: im for im in coco['images']}
    side = np.array([np.sqrt(a['bbox'][2] * 512 / images[a['image_id']]['width'] *
                             a['bbox'][3] * 512 / images[a['image_id']]['height']) for a in coco['annotations']])
    print('1. a square box of side s shifted by d px in x and y has IoU (s-d)^2 / (2 s^2 - (s-d)^2) with itself')
    for d in (1, 2):
        for t in (0.5, 0.75, 0.95):
            s_max = d / (1 - np.sqrt(2 * t / (1 + t)))
            print(f'   {d} px error fails IoU {t}: side < {s_max:5.1f} px at 512 -> {np.mean(side < s_max):5.1%} of boxes')

    print('2. per class: images, the 3 commonest original sizes, boxes per image (mean / median / max, share >= 10)')
    per_image = defaultdict(list)
    for a in coco['annotations']:
        per_image[a['image_id']].append(a)
    by_class = defaultdict(list)
    for image_id, anns in per_image.items():
        by_class[names[anns[0]['category_id']]].append(image_id)
    for name in sorted(by_class):
        ids = by_class[name]
        sizes = Counter(f"{images[i]['width']}x{images[i]['height']}" for i in ids).most_common(3)
        n = np.array([len(per_image[i]) for i in ids])
        print(f'   {name:13s} {len(ids):5d}  ' + ', '.join(f'{s} {c / len(ids):.0%}' for s, c in sizes).ljust(38) +
              f' {n.mean():4.1f} / {np.median(n):2.0f} / {n.max():2d}, {np.mean(n >= 10):5.1%}')


if __name__ == '__main__':
    main()
