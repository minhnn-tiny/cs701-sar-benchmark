"""Statistics of the train split that drive the design (written up in docs/DATA.md).
Run from the repo root: uv run python workspace/detail_lora/analyze_data.py  (about 1 min on 32 processes)
Writes docs/data_stats.json next to this script; prints a short summary."""
import csv
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou

from make_holdout import duplicate_groups

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'dataset' / 'train'
SIZE = 512
BASELINE_ANCHORS = ((8, 11), (16, 23), (32, 45), (64, 91), (128, 181))  # sarbench/model.py
PROPOSED_ANCHORS = tuple(tuple(round(2 ** (k / 2), 2) for k in (j, j + 1, j + 2)) for j in range(5, 15, 2))  # 5.66..181
STRIDES = (4, 8, 16, 32, 64)


def image_stats(path):
    """Intensity statistics of one image read as gray, as sarbench/data.py does."""
    x = np.asarray(Image.open(path).convert('L'), dtype=np.float32) / 255
    return {'mean': float(x.mean()), 'std': float(x.std()), 'p01': float(np.percentile(x, 1)),
            'p50': float(np.median(x)), 'p99': float(np.percentile(x, 99)),
            'zero': float((x == 0).mean()), 'sat': float((x == 1).mean())}


def anchors_512(sizes):
    """All anchors of an AnchorGenerator on a 512 x 512 input, as XYXY."""
    generator = AnchorGenerator(sizes, ((0.5, 1.0, 2.0),) * len(sizes))
    features = [torch.zeros(1, 1, SIZE // s, SIZE // s) for s in STRIDES[:len(sizes)]]
    images = ImageList(torch.zeros(1, 1, SIZE, SIZE), [(SIZE, SIZE)])
    return generator(images, features)[0]


def pct(values, q):
    return float(np.percentile(values, q))


def main():
    names = [c['name'] for c in json.loads((ROOT / 'dataset' / 'classes.json').read_text())]
    rows = list(csv.DictReader(open(DATA / 'labels.csv')))
    coco = json.loads((DATA / 'instances.json').read_text())
    info = {img['id']: img for img in coco['images']}
    label = {int(r['image_id']): int(r['label']) for r in rows}
    out = {}

    # Classes, image sizes, file types
    out['images_per_class'] = {names[c]: n for c, n in sorted(Counter(label.values()).items())}
    size_key = lambda img: f"{img['width']}x{img['height']}" if img['width'] in (512, 256, 224) and \
        img['width'] == img['height'] else 'other'
    by_size = defaultdict(Counter)
    for i, c in label.items():
        by_size[names[c]][size_key(info[i])] += 1
    out['image_size_by_class'] = {k: dict(v) for k, v in by_size.items()}
    ext = defaultdict(Counter)
    for i, c in label.items():
        ext[names[c]][Path(info[i]['file_name']).suffix] += 1
    out['file_type_by_class'] = {k: dict(v) for k, v in ext.items()}

    # Boxes, at the 512 x 512 model input (non-square images are stretched, as in sarbench/data.py)
    boxes512, box_class, orig_area, per_image = [], [], [], Counter()
    for a in coco['annotations']:
        img = info[a['image_id']]
        x, y, w, h = a['bbox']
        sx, sy = SIZE / img['width'], SIZE / img['height']
        boxes512.append([x * sx, y * sy, (x + w) * sx, (y + h) * sy])
        box_class.append(a['category_id'] - 1)
        orig_area.append(w * h)
        per_image[a['image_id']] += 1
    b = torch.tensor(boxes512)
    box_class, orig_area = np.array(box_class), np.array(orig_area)
    side = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).sqrt().numpy()  # sqrt(area) at 512
    aspect = ((b[:, 2] - b[:, 0]) / (b[:, 3] - b[:, 1]).clamp(min=1e-6)).numpy()
    aspect = np.maximum(aspect, 1 / aspect)

    # Best IoU of every box with any anchor of the baseline generator (the RPN calls an anchor positive at
    # IoU >= 0.7; a box whose best anchor is below that is matched only through allow_low_quality_matches)
    ids = [a['image_id'] for a in coco['annotations']]
    best = torch.empty(len(b))
    for name, sizes in [('baseline', BASELINE_ANCHORS), ('proposed', PROPOSED_ANCHORS)]:
        anchors = anchors_512(sizes)
        for start in range(0, len(b), 2048):
            best[start:start + 2048] = box_iou(b[start:start + 2048], anchors).max(dim=1).values
        out[f'anchor_best_iou_{name}'] = {
            'sizes': sizes, 'all': {'median': float(best.median()), 'below_0.3': float((best < 0.3).float().mean()),
                    'below_0.5': float((best < 0.5).float().mean()), 'below_0.7': float((best < 0.7).float().mean())},
            'by_side_512': {f'{lo}-{hi}px': {'boxes': int(m.sum()), 'median': float(best[m].median()),
                                              'below_0.5': float((best[m] < 0.5).float().mean())}
                            for lo, hi in [(0, 4), (4, 6), (6, 8), (8, 12), (12, 16), (16, 32), (32, 1e9)]
                            for m in [torch.from_numpy((side >= lo) & (side < hi))] if m.any()}}
        if name == 'baseline':
            best_base = best.clone()

    out['boxes'] = {
        'total': int(len(b)), 'per_image_max': max(per_image.values()),
        'images_with_one_box': float(np.mean([n == 1 for n in per_image.values()])),
        'side_512': {q: pct(side, q) for q in (1, 10, 25, 50, 75, 90, 99)},
        'side_512_below': {f'{t}px': float((side < t).mean()) for t in (4, 6, 8, 12, 16, 32)},
        'aspect_above_3': float((aspect > 3).mean()),
        # COCO area buckets are measured in ORIGINAL pixels (what the leaderboard scores)
        'coco_area_buckets': {'small (<32^2)': float((orig_area < 32 ** 2).mean()),
                              'medium': float(((orig_area >= 32 ** 2) & (orig_area < 96 ** 2)).mean()),
                              'large (>=96^2)': float((orig_area >= 96 ** 2).mean())}}
    out['boxes_by_class'] = {}
    for c, name in enumerate(names):
        m = box_class == c
        imgs = [i for i, l in label.items() if l == c]
        out['boxes_by_class'][name] = {
            'boxes': int(m.sum()), 'boxes_per_image_mean': float(m.sum() / len(imgs)),
            'boxes_per_image_max': max(per_image[i] for i in imgs),
            'side_512_p10_p50_p90': [round(pct(side[m], q), 1) for q in (10, 50, 90)],
            'below_8px': float((side[m] < 8).mean()), 'below_16px': float((side[m] < 16).mean()),
            'aspect_above_3': float((aspect[m] > 3).mean()),
            'anchor_best_iou_below_0.5': float((best_base[torch.from_numpy(m)] < 0.5).float().mean()),
            'coco_small': float((orig_area[m] < 32 ** 2).mean())}

    # Intensity statistics and near-duplicates (dHash), in parallel
    paths = [DATA / info[int(r['image_id'])]['file_name'] for r in rows]
    with Pool(32) as pool:
        stats = pool.map(image_stats, paths, chunksize=32)
    stat = {k: np.array([s[k] for s in stats]) for k in ('mean', 'std', 'p01', 'p50', 'p99', 'zero', 'sat')}
    cls = np.array([int(r['label']) for r in rows])
    sizes = np.array([size_key(info[int(r['image_id'])]) for r in rows])
    out['intensity'] = {
        'image_mean_p5_p50_p95': [round(pct(stat['mean'], q), 3) for q in (5, 50, 95)],
        'image_std_p5_p50_p95': [round(pct(stat['std'], q), 3) for q in (5, 50, 95)],
        'image_p99_p5_p50_p95': [round(pct(stat['p99'], q), 3) for q in (5, 50, 95)],
        'images_with_>1%_saturated': float((stat['sat'] > 0.01).mean()),
        'images_with_>10%_zero': float((stat['zero'] > 0.10).mean()),
        'by_class_mean_std': {names[c]: [round(float(stat['mean'][cls == c].mean()), 3),
                                         round(float(stat['std'][cls == c].mean()), 3)] for c in range(9)},
        'by_size_mean_std': {s: [round(float(stat['mean'][sizes == s].mean()), 3),
                                 round(float(stat['std'][sizes == s].mean()), 3), int((sizes == s).sum())]
                             for s in sorted(set(sizes))}}
    groups = [g for g in duplicate_groups(paths) if len(g) > 1]
    out['near_duplicates'] = {
        'method': '64x64 gray thumbnails, Pearson correlation > 0.85, connected components (make_holdout.py)',
        'groups': len(groups), 'images_in_groups': sum(map(len, groups)), 'largest_group': max(map(len, groups)),
        'groups_with_mixed_labels': sum(len(set(cls[g])) > 1 for g in groups),
        'images_in_groups_by_class': {names[c]: int(sum((cls[g] == c).sum() for g in groups)) for c in range(9)}}

    (HERE / 'docs').mkdir(exist_ok=True)
    (HERE / 'docs' / 'data_stats.json').write_text(json.dumps(out, indent=1, default=str))
    json.dump(out, sys.stdout, indent=1, default=str)


if __name__ == '__main__':
    main()
