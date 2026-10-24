"""Per-image and per-box features of the TRAIN split only (no val/test pixels or metadata are read), for the
data analysis (analyze_train.py) and the near-duplicate search (near_dups.py). CPU only.

Writes to dataset/holdout_sarmtl/analysis/ (gitignored, recomputable):
  train_image_stats.csv  one row per image: size, format, PIL mode, channel equality, intensity statistics at the
                         original size and at the 512 x 512 model input, sharpness, speckle proxy, md5s
  train_box_stats.csv    one row per box: original and 512-input geometry, inside / local-background intensity
  train_thumbs.npz       thumbnails: whole image at 32 x 32 and 64 x 64 (any size; used by near_dups.py), and the
                         native-scale 1/8 thumbnail of every 512 x 512 image (64 x 64; not used by near_dups.py)
Usage (repo root): uv run python workspace/sarmtl/data/extract_features.py [--workers 12]
"""
import argparse
import csv
import hashlib
import io
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as TF

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'dataset' / 'train'
OUT = ROOT / 'dataset' / 'holdout_sarmtl' / 'analysis'


def image_features(job):
    image_id, file_name, label, boxes = job
    torch.set_num_threads(1)
    raw = (DATA / file_name).read_bytes()
    im = Image.open(io.BytesIO(raw))
    fmt, mode, (w, h) = im.format, im.mode, im.size
    chan_maxdiff = chan_meandiff = 0.0
    if mode not in ('L', 'I;16', 'I', 'F'):
        rgb = np.asarray(im.convert('RGB')).astype(np.int16)
        d = np.maximum(np.abs(rgb[..., 0] - rgb[..., 1]), np.abs(rgb[..., 1] - rgb[..., 2]))
        chan_maxdiff, chan_meandiff = float(d.max()), float(d.mean())
    g = im.convert('L')  # exactly as sarbench/data.py
    a = np.asarray(g)
    af = a.astype(np.float64)
    pct = np.percentile(a, [1, 5, 50, 95, 99])
    # the model input: torchvision v2 Resize on the uint8 tensor (bilinear, antialias), as in sarbench/data.py
    r = TF.resize(torch.from_numpy(a.copy())[None], [512, 512], antialias=True)[0].numpy()
    rf = r.astype(np.float64) / 255
    lap = cv2.Laplacian(r.astype(np.float32), cv2.CV_32F)
    # speckle proxy: coefficient of variation std/mean of 8 x 8 blocks at the original size (blocks with mean >= 8)
    bh, bw = h // 8, w // 8
    blocks = af[:bh * 8, :bw * 8].reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
    bm, bs = blocks.mean(1), blocks.std(1)
    keep = bm >= 8
    cv = bs[keep] / bm[keep] if keep.any() else np.array([np.nan])
    # boxes: inside mean / max and the local background (a ring of one box size around it, other boxes masked out)
    mask = np.zeros((h, w), bool)
    ib = []
    for (x, y, bw_, bh_) in boxes:
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = max(int(np.ceil(x + bw_)), x0 + 1), max(int(np.ceil(y + bh_)), y0 + 1)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        ib.append((x0, y0, x1, y1))
        mask[y0:y1, x0:x1] = True
    box_rows = []
    for (x0, y0, x1, y1), (x, y, bw_, bh_) in zip(ib, boxes):
        inside = af[y0:y1, x0:x1]
        m = max(int(round(max(bw_, bh_))), 2)
        X0, Y0, X1, Y1 = max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, w), min(y1 + m, h)
        ring = af[Y0:Y1, X0:X1][~mask[Y0:Y1, X0:X1]]
        box_rows.append((float(inside.mean()) if inside.size else np.nan, float(inside.max()) if inside.size else np.nan,
                         float(ring.mean()) if ring.size else np.nan, int(ring.size)))
    outside = af[~mask]
    t32 = np.asarray(g.resize((32, 32), Image.BOX), dtype=np.uint8)
    t64 = np.asarray(g.resize((64, 64), Image.BOX), dtype=np.uint8)
    nat = np.asarray(g.resize((w // 8, h // 8), Image.BOX), dtype=np.uint8) if (w, h) == (512, 512) else None
    row = dict(image_id=image_id, file_name=file_name, label=label, width=w, height=h, format=fmt, mode=mode,
               bytes=len(raw), chan_maxdiff=chan_maxdiff, chan_meandiff=chan_meandiff,
               mean=af.mean() / 255, std=af.std() / 255, p1=pct[0], p5=pct[1], p50=pct[2], p95=pct[3], p99=pct[4],
               frac0=float((a == 0).mean()), frac255=float((a == 255).mean()),
               sum512=rf.sum(), sumsq512=(rf ** 2).sum(), mean512=rf.mean(), std512=rf.std(),
               lapvar512=float(lap.var()), cv_median=float(np.median(cv)), cv_p10=float(np.percentile(cv, 10)),
               inbox_mean=float(af[mask].mean()) / 255, outbox_mean=float(outside.mean()) / 255 if outside.size else np.nan,
               box_area_frac=float(mask.mean()),
               md5_file=hashlib.md5(raw).hexdigest(),
               md5_pixels=hashlib.md5(a.tobytes() + str(a.shape).encode()).hexdigest())
    return row, box_rows, t32, t64, nat


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(DATA / 'labels.csv') as f:
        rows = list(csv.DictReader(f))
    coco = json.loads((DATA / 'instances.json').read_text())
    by_file = {img['file_name']: img for img in coco['images']}
    anns = {}
    for a in coco['annotations']:
        anns.setdefault(a['image_id'], []).append(a)
    jobs = []
    for r in rows:
        info = by_file[r['file_name']]
        assert info['id'] == int(r['image_id'])
        jobs.append((info['id'], r['file_name'], int(r['label']), [a['bbox'] for a in anns[info['id']]]))
    with Pool(args.workers) as pool:
        results = pool.map(image_features, jobs, chunksize=32)
    img_rows, box_rows, t32, t64, nat_ids, nat = [], [], [], [], [], []
    for (image_id, file_name, label, _), (row, brs, a32, a64, n) in zip(jobs, results):
        img_rows.append(row)
        t32.append(a32)
        t64.append(a64)
        if n is not None:
            nat_ids.append(image_id)
            nat.append(n)
        for a, (in_mean, in_max, ring_mean, ring_n) in zip(anns[image_id], brs):
            x, y, bw, bh = a['bbox']
            W, H = row['width'], row['height']
            box_rows.append(dict(ann_id=a['id'], image_id=image_id, label=label, category_id=a['category_id'],
                                 img_w=W, img_h=H, x=x, y=y, w=bw, h=bh, area=a['area'],
                                 w512=bw * 512 / W, h512=bh * 512 / H,
                                 touches_border=int(x <= 0.5 or y <= 0.5 or x + bw >= W - 0.5 or y + bh >= H - 0.5),
                                 in_mean=in_mean, in_max=in_max, ring_mean=ring_mean, ring_n=ring_n))
    for name, table in [('train_image_stats.csv', img_rows), ('train_box_stats.csv', box_rows)]:
        with open(OUT / name, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(table[0]))
            wr.writeheader()
            wr.writerows(table)
    np.savez_compressed(OUT / 'train_thumbs.npz', ids=np.array([j[0] for j in jobs]), t32=np.stack(t32),
                        t64=np.stack(t64), nat_ids=np.array(nat_ids), nat64=np.stack(nat))
    print(f'{len(img_rows)} images, {len(box_rows)} boxes -> {OUT}')


if __name__ == '__main__':
    main()
