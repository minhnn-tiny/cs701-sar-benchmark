"""Anchor and RoI-level coverage of the train boxes at the 512 x 512 input (CPU only, no model).

For every train box (resized to 512 x 512 like sarbench/data.py) and an anchor configuration:
  - the best IoU over all anchors, and the number of anchors at IoU >= 0.7 / 0.5 / 0.3,
    i.e. how the RPN matcher (fg 0.7, bg 0.3, allow_low_quality_matches=True) sees that box;
  - the RoIAlign level torchvision's MultiScaleRoIAlign gives a box of that size (LevelMapper, canonical 224 / level 4);
  - the box size in feature cells of the stride-4 and stride-16 maps.
Run from the repo root:  uv run python workspace/sarmtl/scratch/cv-detect/anchors.py [--configs ref,alt]
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parents[4]
CONFIGS = {
    # the reference (sarbench/model.py): 2 scales per octave, 3 ratios, 6 anchors per location
    'ref': (((8, 11), (16, 23), (32, 45), (64, 91), (128, 181)), (0.5, 1.0, 2.0)),
    # same octaves, one more scale per octave below each (3 scales x 3 ratios = 9 per location)
    'ref+s4': (((5.7, 8, 11), (16, 23, 32), (32, 45, 64), (64, 91, 128), (128, 181, 256)), (0.5, 1.0, 2.0)),
    # shift every level down by half an octave
    'half-down': (((5.7, 8), (11, 16), (23, 32), (45, 64), (91, 128)), (0.5, 1.0, 2.0)),
    # wider ratios for elongated boxes (bridges, ships)
    'ratios5': (((8, 11), (16, 23), (32, 45), (64, 91), (128, 181)), (0.33, 0.5, 1.0, 2.0, 3.0)),
    # 1:4 and 4:1 for airports (20 % beyond 1:4) and ships (data-eng, DATA.md section 8)
    'ratios1:4': (((8, 11), (16, 23), (32, 45), (64, 91), (128, 181)), (0.25, 0.5, 1.0, 2.0, 4.0)),
}
SIZE_BINS = [0, 4, 8, 12, 16, 24, 32, 64, 128, 1e9]


def load_boxes():
    """Every train box as XYXY at the 512 x 512 input, with its class name and its image id."""
    coco = json.loads((ROOT / 'dataset/train/instances.json').read_text())
    names = {c['id']: c['name'] for c in coco['categories']}
    images = {im['id']: im for im in coco['images']}
    per_image = defaultdict(list)
    for a in coco['annotations']:
        im = images[a['image_id']]
        sx, sy = 512 / im['width'], 512 / im['height']
        x, y, w, h = a['bbox']
        per_image[a['image_id']].append(([x * sx, y * sy, (x + w) * sx, (y + h) * sy], names[a['category_id']],
                                         (w, h)))
    return per_image


def anchors_for(config):
    sizes, ratios = config
    gen = AnchorGenerator(sizes, (ratios,) * len(sizes))
    feats = [torch.zeros(1, 1, 512 // s, 512 // s) for s in (4, 8, 16, 32, 64)]
    anchors = gen(ImageList(torch.zeros(1, 1, 512, 512), [(512, 512)]), feats)[0]
    per_level = [f.shape[-1] ** 2 * len(sizes[0]) * len(ratios) for f in feats]
    level = torch.cat([torch.full((n,), k) for k, n in enumerate(per_level)])
    return anchors, level


def roi_level(s):
    """torchvision LevelMapper: floor(4 + log2(s / 224)), clamped to [2, 5] (strides 4..32)."""
    return int(min(max(math.floor(4 + math.log2(max(s, 1e-6) / 224) + 1e-6), 2), 5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', default=','.join(CONFIGS))
    args = p.parse_args()
    torch.set_num_threads(16)
    per_image = load_boxes()
    boxes = [b for v in per_image.values() for b in v]
    xyxy = np.array([b[0] for b in boxes])
    s512 = np.sqrt((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]))
    s_orig = np.array([math.sqrt(b[2][0] * b[2][1]) for b in boxes])
    cls = np.array([b[1] for b in boxes])
    print(f'{len(boxes)} boxes in {len(per_image)} images')
    print('sqrt(area) at the 512 input: percentiles 5/10/25/50/75/90/95/99 =',
          ' / '.join(f'{v:.1f}' for v in np.percentile(s512, [5, 10, 25, 50, 75, 90, 95, 99])))
    for t in (4, 8, 12, 16, 32, 64, 112):
        print(f'  share of boxes with sqrt(area) < {t:3d} px at 512: {np.mean(s512 < t):6.1%}'
              f'   (in original pixels: {np.mean(s_orig < t):6.1%})')
    print('COCO area groups (original pixels): small < 32^2: {:.1%}, medium: {:.1%}, large >= 96^2: {:.1%}'.format(
        np.mean(s_orig ** 2 < 32 ** 2), np.mean((s_orig ** 2 >= 32 ** 2) & (s_orig ** 2 < 96 ** 2)),
        np.mean(s_orig ** 2 >= 96 ** 2)))
    lv = np.array([roi_level(s) for s in s512])
    print('RoIAlign level of the GT boxes (proxy for proposals):',
          ', '.join(f'stride {2 ** k}: {np.mean(lv == k):.1%}' for k in (2, 3, 4, 5)))
    print('\nper class at 512: n, median sqrt(area), share < 16 px, < 8 px, median w/h ratio (long/short)')
    for c in sorted(set(cls)):
        m = cls == c
        wh = np.stack([xyxy[m, 2] - xyxy[m, 0], xyxy[m, 3] - xyxy[m, 1]], 1)
        elong = np.maximum(wh[:, 0], wh[:, 1]) / np.maximum(np.minimum(wh[:, 0], wh[:, 1]), 1e-6)
        print(f'  {c:13s} {m.sum():6d}  {np.median(s512[m]):6.1f}  {np.mean(s512[m] < 16):6.1%}  '
              f'{np.mean(s512[m] < 8):6.1%}  {np.median(elong):4.2f}  (p90 {np.percentile(elong, 90):.2f})')

    cls_order = np.array([b[1] for v in per_image.values() for b in v])  # same order as the loop below
    for name in args.configs.split(','):
        anchors, level = anchors_for(CONFIGS[name])
        best, best_level, n07, n05, n03 = [], [], [], [], []
        for v in per_image.values():
            gt = torch.tensor([b[0] for b in v], dtype=torch.float32)
            iou = box_iou(gt, anchors)  # [G, A]
            b, i = iou.max(dim=1)
            best += b.tolist()
            best_level += level[i].tolist()
            n07 += (iou >= 0.7).sum(1).tolist()
            n05 += (iou >= 0.5).sum(1).tolist()
            n03 += (iou >= 0.3).sum(1).tolist()
        best, n07, n05, n03 = map(np.array, (best, n07, n05, n03))
        print(f'\n== anchors {name}: sizes {CONFIGS[name][0]}, ratios {CONFIGS[name][1]}, {len(anchors)} anchors')
        print(f'  best IoU: median {np.median(best):.2f}; share of boxes whose best anchor IoU < 0.7: '
              f'{np.mean(best < 0.7):.1%}, < 0.5: {np.mean(best < 0.5):.1%}, < 0.3 (bg): {np.mean(best < 0.3):.1%}')
        print('  size bin (512 px) | boxes | best IoU p10/median | #anchors IoU>=0.7 (mean) | >=0.5 | >=0.3 | '
              'share best<0.5')
        for lo, hi in zip(SIZE_BINS[:-1], SIZE_BINS[1:]):
            m = (s512 >= lo) & (s512 < hi)
            if m.sum() == 0:
                continue
            print(f'  [{lo:4.0f},{hi if hi < 1e8 else math.inf:5.0f}) | {m.sum():5d} ({m.mean():5.1%}) | '
                  f'{np.percentile(best[m], 10):.2f}/{np.median(best[m]):.2f} | {n07[m].mean():6.2f} | '
                  f'{n05[m].mean():6.2f} | {n03[m].mean():6.2f} | {np.mean(best[m] < 0.5):5.1%}')
        bl = np.array(best_level)
        print('  level of the best anchor:', ', '.join(f'stride {4 * 2 ** k}: {np.mean(bl == k):.1%}' for k in range(5)))
        print('  per class: median best IoU / share best < 0.5 / mean #anchors IoU >= 0.5:  ' + '; '.join(
            f'{c} {np.median(best[cls_order == c]):.2f}/{np.mean(best[cls_order == c] < 0.5):.0%}/'
            f'{n05[cls_order == c].mean():.1f}' for c in sorted(set(cls_order))))


if __name__ == '__main__':
    main()
