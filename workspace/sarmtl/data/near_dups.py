"""Near-duplicate and overlap groups in the TRAIN split (no val/test data is read). CPU only.

Kinds of links between two train images A and B (B transformed by one of the 8 flips/rotations, or shifted):
  copy     the same acquisition: B is a flipped/rotated (or identical) copy of A, possibly re-encoded.
           Test: high-pass NCC (image minus its Gaussian blur, sigma 3 px: keeps speckle and edges) >= 0.5.
  revisit  the same site, another acquisition (other date or polarisation: different speckle and grey levels, the
           same boxes). Test: every box of A matches a box of B at IoU >= 0.9 (same count) and full NCC >= 0.3; a
           single-box pair also needs high-pass NCC >= 0.04 (unrelated pairs are <= 0.02), or high-pass NCC >= 0.15
           with a mean box match >= 0.5.
  related  weaker evidence, linked to be safe (a false link only costs split flexibility; a missed one leaks),
           same image size only: high-pass NCC >= 0.15, or high-pass NCC >= 0.1 with mean box match >= 0.3, or full NCC >= 0.7 with mean
           box match >= 0.3 (unrelated pairs: high-pass NCC <= 0.02 for 97 % of candidates).
  overlap  shifted tiles of one scene: >= 2 boxes of A reappear in B at one common offset (same size within 0.5 px),
           or one box of >= 400 px^2, and the overlapping pixels agree (NCC >= 0.9 on >= 5 % of the image).
  nshift   (stage 6) shifted overlaps found with the correlation normalised by the overlap energies (nshift.py,
           the critic's method), so partial overlaps score like full ones: 32 x 32 high-passed thumbnails, 8
           transforms, all pairs of the same class and size; candidates (NCC >= 0.5 on >= 20 %, or >= 0.85 on >= 10 %)
           are refined at 128 px and at full resolution and linked by the stage-5 rule.
  georevisit  the same site under a flip/rotation with other radiometry (found by the critic's review: these pairs
           have thumbnail NCC < 0.6 and their boxes are 1 px off, so neither candidate path above sees them):
           under one of the 8 transforms, >= 50 % of the boxes of A and of B (>= 1) match in position within 2.6 px
           and in size within 0.6 px (512^2; 4.6 / 1.1 px for smaller images), and the band-pass NCC of the 64 x 64
           thumbnails (minus their Gaussian blur, sigma 2 thumbnail px) under that transform is >= 0.15.
  shift    shifted tiles found from pixels alone (their boxes can differ: the same objects boxed with other sizes):
           for every pair of square images of the same class and size, the cross-correlation of the band-pass 64 x 64
           thumbnails over all shifts (no flip); every pair not yet in one group with a peak >= 0.35 is re-checked at
           full resolution (the best of the 5 highest peaks of the cross-correlation of the high-passed images, which
           finds the shift again) and linked if the overlap is >= 5 % of the image and its high-pass NCC >= 0.3 or
           its NCC >= 0.9. Random same-class pairs reach high-pass NCC >= 0.3 in 0-3 % of cases (null test, 120 pairs
           per class); the critic's and my visual checks of links at 0.3-0.5 show the same ships, turbine
           constellations and structures, re-processed or re-acquired. Rule 'same' = high-pass NCC >= 0.8 (the same
           acquisition), 'partial' = 0.3-0.8 (re-processed or re-acquired), 'ncc' = NCC >= 0.9 only.
           Flipped/rotated AND shifted copies (found by the audit): for every pair of the same class and square size,
           the circular cross-correlation of the band-pass thumbnails under the 7 non-identity transforms; pairs with
           a peak >= 0.28 whose thumbnail overlap NCC (best of the 4 unwrapped shifts) is >= 0.7 on >= 10 % are
           checked at full resolution under that transform and linked by the same rule (rules '*_transformed').
Candidates for copy/revisit: 32 x 32 thumbnail NCC >= 0.6 under the best of the 8 transforms, plus every pair with
identical box lists. Groups = connected components of all links; make_holdout.py keeps each group on one side.

Reads dataset/holdout_sarmtl/analysis/train_thumbs.npz (extract_features.py); writes
workspace/sarmtl/data/near_dup_groups.json (groups, summary, thresholds) and analysis/near_dup_links.csv (every link
with its scores).
Usage (repo root): uv run python workspace/sarmtl/data/near_dups.py [--workers 14]
"""
import argparse
import hashlib
import csv
import json
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'dataset' / 'train'
ANALYSIS = ROOT / 'dataset' / 'holdout_sarmtl' / 'analysis'
OUT = Path(__file__).resolve().parent / 'near_dup_groups.json'
NAMES = [c['name'] for c in json.loads((ROOT / 'dataset' / 'classes.json').read_text())]
TH = dict(thumb_ncc=0.6, copy_hpncc=0.5, revisit_box_iou=0.9, revisit_ncc=0.3, revisit_single_hpncc=0.04,
          scene_hpncc=0.15, scene_boxmatch=0.5, related_hpncc=0.15, related_hpncc_box=0.1, related_ncc_box=0.7,
          related_boxmatch=0.3, overlap_ncc=0.9, overlap_min_frac=0.05, overlap_big_box=400,
          geo_pos_512=2.6, geo_size_512=0.6, geo_pos_small=4.6, geo_size_small=1.1, geo_min_frac=0.5, bp_sigma=2.0,
          geo_bpncc=0.15, shift_peak=0.35, shift_min_frac=0.05, shift_hpncc=0.3, shift_ncc=0.9, shift_same_hpncc=0.8,
          tshift_peak=0.28, tshift_thumb_ncc=0.7, tshift_thumb_frac=0.1)

FILES, SIZES, BOXES = {}, {}, {}


def dihedral(x, k):
    """k = 0..3: rotate by k x 90 degrees counter-clockwise (np.rot90); k = 4..7: the same, then flip left-right."""
    y = np.rot90(x, k % 4, axes=(-2, -1))
    return y[..., ::-1] if k >= 4 else y


def dihedral_box(b, w, h, k):
    """XYXY box on a w x h image -> the same box after dihedral(image, k)."""
    pts = [(b[0], b[1]), (b[2], b[3])]
    for _ in range(k % 4):
        pts = [(y, w - x) for x, y in pts]
        w, h = h, w
    if k >= 4:
        pts = [(w - x, y) for x, y in pts]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_scores(a, b, k):
    """(mean best IoU, min over both directions; fraction of boxes matched at IoU >= 0.9, min over directions)."""
    w, h = SIZES[b]
    A = BOXES[a]
    B = [dihedral_box(x, w, h, k) for x in BOXES[b]]
    ab = [max(iou(p, q) for q in B) for p in A]
    ba = [max(iou(p, q) for q in A) for p in B]
    t = TH['revisit_box_iou']
    return min(np.mean(ab), np.mean(ba)), min(np.mean([v >= t for v in ab]), np.mean([v >= t for v in ba]))


def load(i):
    return np.asarray(Image.open(DATA / FILES[i]).convert('L'), dtype=np.float32)


def ncc(x, y):
    x, y = x - x.mean(), y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def highpass(x):
    return x - cv2.GaussianBlur(x, (0, 0), 3)


def verify_pair(job):
    """Full-resolution scores of A vs B under transform k (B resized to A's shape if the sizes differ)."""
    a, b, k = job
    A, B = load(a), np.ascontiguousarray(dihedral(load(b), k))
    same_shape = A.shape == B.shape
    if not same_shape:
        B = cv2.resize(B, A.shape[::-1], interpolation=cv2.INTER_AREA)
    bm, bx = box_scores(a, b, k)
    return dict(a=a, b=b, k=k, same_shape=same_shape, ncc=ncc(A, B), hpncc=ncc(highpass(A), highpass(B)),
                mad=float(np.abs(A - B).mean()), equal_frac=float((A == B).mean()), boxmatch=float(bm),
                boxexact=float(bx), n_a=len(BOXES[a]), n_b=len(BOXES[b]))


def verify_shift(job):
    """Pixel agreement of A and B where B is A shifted by (dx, dy): A[y, x] = B[y - dy, x - dx]."""
    a, b, dx, dy = job
    A, B = load(a), load(b)
    h, w = A.shape
    if B.shape != A.shape:
        return None
    xa0, xa1, ya0, ya1 = max(0, dx), min(w, w + dx), max(0, dy), min(h, h + dy)
    if xa1 - xa0 < 8 or ya1 - ya0 < 8:
        return None
    pa, pb = A[ya0:ya1, xa0:xa1], B[ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx]
    return dict(a=a, b=b, dx=dx, dy=dy, overlap_frac=pa.size / A.size, ncc=ncc(pa, pb),
                hpncc=ncc(highpass(pa), highpass(pb)), mad=float(np.abs(pa - pb).mean()))


def bandpass_thumbs(t64):
    """64 x 64 whole-image thumbnails minus their Gaussian blur (sigma 2 thumbnail px), zero-mean, unit-norm.
    The blur is isotropic with a mirror border, so it commutes with the 8 flips/rotations."""
    out = np.empty(t64.shape, np.float32)
    for n, t in enumerate(t64.astype(np.float32)):
        b = t - cv2.GaussianBlur(t, (0, 0), TH['bp_sigma'])
        b -= b.mean()
        out[n] = b / max(float(np.linalg.norm(b)), 1e-6)
    return out


def box_geometry_matches(boxes, sizes, label):
    """(a, b, k) -> (boxes of a matched, boxes of b matched) where dihedral(b, k) puts b's boxes onto a's boxes within
    the position/size tolerances, for images of the same class and size."""
    from scipy.spatial import cKDTree
    buckets = defaultdict(list)
    for i, bs in boxes.items():
        for j, b in enumerate(bs):
            buckets[(label[i], sizes[i])].append((i, j, b))
    geo = defaultdict(lambda: (set(), set()))
    for (_, (W, H)), entries in buckets.items():
        pt, st = ((TH['geo_pos_512'], TH['geo_size_512']) if (W, H) == (512, 512)
                  else (TH['geo_pos_small'], TH['geo_size_small']))
        scale = np.array([pt, pt, st, st])
        xywh = lambda b: [b[0], b[1], b[2] - b[0], b[3] - b[1]]
        tree = cKDTree(np.array([xywh(b) for _, _, b in entries]) / scale)
        for k in range(8):
            if k % 2 and W != H:
                continue
            q = np.array([xywh(dihedral_box(b, W, H, k)) for _, _, b in entries]) / scale
            for n, hits in enumerate(tree.query_ball_point(q, r=1.0, p=np.inf)):
                ib, jb = entries[n][0], entries[n][1]
                for h in hits:
                    ia, ja = entries[h][0], entries[h][1]
                    if ia < ib:
                        geo[(ia, ib, k)][0].add(ja)
                        geo[(ia, ib, k)][1].add(jb)
    return geo


def shift_candidates(bp, ids, label, sizes):
    """Same class, same square size, all pairs: cross-correlation of band-pass thumbnails over every shift (zero-padded
    FFT) -> (a, b, dx, dy, peak, overlap NCC, overlap fraction) for peaks >= shift_peak, with A[y, x] ~ B[y - dy, x - dx]
    in thumbnail pixels."""
    n, s = bp.shape[0], bp.shape[1]
    F = torch.fft.rfft2(torch.from_numpy(bp), s=(2 * s, 2 * s))
    buckets = defaultdict(list)
    for m, i in enumerate(ids):
        w, h = sizes[int(i)]
        if w == h:
            buckets[(label[int(i)], w)].append(m)
    out = []
    for members in buckets.values():
        members = np.array(members)
        for x, i in enumerate(members[:-1]):
            js = members[x + 1:]
            c = torch.fft.irfft2(F[i][None] * F[js].conj(), s=(2 * s, 2 * s)).reshape(len(js), -1)
            peak, arg = c.max(1)
            keep = (peak >= TH['shift_peak']).numpy()
            for j, v, a in zip(js[keep].tolist(), peak.numpy()[keep].tolist(), arg.numpy()[keep].tolist()):
                dy, dx = divmod(a, 2 * s)
                dy, dx = (dy - 2 * s if dy >= s else dy), (dx - 2 * s if dx >= s else dx)
                ya0, ya1, xa0, xa1 = max(0, dy), min(s, s + dy), max(0, dx), min(s, s + dx)
                if ya1 - ya0 < 4 or xa1 - xa0 < 4:
                    continue
                pa, pb = bp[i, ya0:ya1, xa0:xa1], bp[j, ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx]
                out.append((int(ids[i]), int(ids[j]), dx, dy, v, ncc(pa, pb), pa.size / (s * s)))
    return out


def transformed_shift_candidates(bp, ids, label, sizes, chunk=24):
    """Same class, same square size, all pairs, all 8 transforms of the first image: circular cross-correlation of
    the band-pass thumbnails -> {(a, b): (k, peak, overlap NCC, overlap fraction)} where dihedral(a, k) ~ b shifted,
    kept if peak >= tshift_peak and the thumbnail overlap NCC >= tshift_thumb_ncc on >= tshift_thumb_frac."""
    s = bp.shape[1]
    F0 = torch.fft.rfft2(torch.from_numpy(bp)).conj()
    buckets = defaultdict(list)
    for m, i in enumerate(ids):
        w, h = sizes[int(i)]
        if w == h:
            buckets[(label[int(i)], w)].append(m)
    out = {}
    for members in buckets.values():
        members = np.array(members)
        for k in range(1, 8):  # k = 0 is stage 5
            Fk = torch.fft.rfft2(torch.from_numpy(np.ascontiguousarray(dihedral(bp[members], k))))
            for c0 in range(0, len(members) - 1, chunk):
                ii = np.arange(c0, min(c0 + chunk, len(members)))
                js = np.arange(c0 + 1, len(members))
                c = torch.fft.irfft2(Fk[ii][:, None] * F0[members[js]][None], s=(s, s)).reshape(len(ii), len(js), -1)
                peak, arg = c.max(2)
                hit = (peak >= TH['tshift_peak']) & torch.from_numpy(js[None, :] > ii[:, None])
                for x, y in zip(*[t.tolist() for t in torch.nonzero(hit, as_tuple=True)]):
                    i, j = members[ii[x]], members[js[y]]
                    dyc, dxc = divmod(int(arg[x, y]), s)
                    A = np.ascontiguousarray(dihedral(bp[i], k))
                    best = None
                    for dy in (dyc, dyc - s):
                        for dx in (dxc, dxc - s):
                            ya0, ya1, xa0, xa1 = max(0, dy), min(s, s + dy), max(0, dx), min(s, s + dx)
                            if ya1 - ya0 < 4 or xa1 - xa0 < 4:
                                continue
                            v = ncc(A[ya0:ya1, xa0:xa1], bp[j][ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx])
                            fr = (ya1 - ya0) * (xa1 - xa0) / (s * s)
                            if fr >= TH['tshift_thumb_frac'] and (best is None or v > best[0]):
                                best = (v, fr)
                    if best and best[0] >= TH['tshift_thumb_ncc']:
                        key = (int(ids[i]), int(ids[j]))
                        if key not in out or best[0] > out[key][2]:
                            out[key] = (k, float(peak[x, y]), best[0], best[1])
    return out


def verify_shift_fft(job):
    """Full resolution: B transformed by k, then the best of the 5 highest peaks of the cross-correlation of the
    high-passed images, scored by NCC and high-pass NCC on the overlap."""
    a, b, k = job if len(job) == 3 else (*job, 0)
    A, B = load(a), np.ascontiguousarray(dihedral(load(b), k))
    if A.shape != B.shape:
        return None
    H, W = A.shape
    ha, hb = highpass(A), highpass(B)
    c = np.fft.irfft2(np.fft.rfft2(ha, (2 * H, 2 * W)) * np.conj(np.fft.rfft2(hb, (2 * H, 2 * W))), (2 * H, 2 * W))
    best = None
    for f in np.argpartition(c, c.size - 5, axis=None)[-5:]:
        dy, dx = np.unravel_index(f, c.shape)
        dy, dx = int(dy - 2 * H if dy >= H else dy), int(dx - 2 * W if dx >= W else dx)
        xa0, xa1, ya0, ya1 = max(0, dx), min(W, W + dx), max(0, dy), min(H, H + dy)
        if xa1 - xa0 < 16 or ya1 - ya0 < 16:
            continue
        pa, pb = A[ya0:ya1, xa0:xa1], B[ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx]
        r = dict(a=a, b=b, k=k, dx=dx, dy=dy, overlap_frac=pa.size / A.size, ncc=ncc(pa, pb),
                 hpncc=ncc(highpass(pa), highpass(pb)), mad=float(np.abs(pa - pb).mean()))
        if best is None or r['hpncc'] > best['hpncc']:
            best = r
    if best is not None:  # how widely the agreement is spread: 4 x 4 blocks of the overlap with high-pass NCC > 0.2
        dx, dy = best['dx'], best['dy']
        xa0, xa1, ya0, ya1 = max(0, dx), min(W, W + dx), max(0, dy), min(H, H + dy)
        pa, pb = ha[ya0:ya1, xa0:xa1], hb[ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx]
        h, w = pa.shape
        v = [ncc(pa[i * h // 4:(i + 1) * h // 4, j * w // 4:(j + 1) * w // 4],
                 pb[i * h // 4:(i + 1) * h // 4, j * w // 4:(j + 1) * w // 4]) for i in range(4) for j in range(4)]
        best['blocks_agree'] = float(np.mean([x > 0.2 for x in v]))
    return best


def init(files, sizes, boxes):
    FILES.update(files)
    SIZES.update(sizes)
    BOXES.update(boxes)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=14)
    args = p.parse_args()
    torch.set_num_threads(min(16, args.workers))
    coco = json.loads((DATA / 'instances.json').read_text())
    with open(DATA / 'labels.csv') as f:
        label = {int(r['image_id']): int(r['label']) for r in csv.DictReader(f)}
    files = {i['id']: i['file_name'] for i in coco['images']}
    sizes = {i['id']: (i['width'], i['height']) for i in coco['images']}
    boxes = defaultdict(list)
    for a in coco['annotations']:
        x, y, w, h = a['bbox']
        boxes[a['image_id']].append([x, y, x + w, y + h])
    boxes = dict(boxes)
    init(files, sizes, boxes)

    # --- Stage 1: candidate pairs (thumbnails under 8 transforms; identical box lists) ---
    z = np.load(ANALYSIS / 'train_thumbs.npz')
    ids = z['ids']
    t32 = torch.from_numpy(z['t32'].astype(np.float32))

    def unit(x):
        x = x.reshape(len(x), -1)
        x = x - x.mean(1, keepdim=True)
        return x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)

    U = unit(t32)
    best = torch.full((len(ids), len(ids)), -2.0)
    for k in range(8):
        best = torch.maximum(best, U @ unit(torch.from_numpy(dihedral(z['t32'], k).copy()).float()).T)
    best = torch.maximum(best, best.T)
    best.fill_diagonal_(-2.0)
    I, J = torch.nonzero(torch.triu(best, 1) >= TH['thumb_ncc'], as_tuple=True)
    cand = {(int(ids[i]), int(ids[j])) for i, j in zip(I.tolist(), J.tolist())}
    n_thumb = len(cand)
    sig = defaultdict(list)  # identical box lists (k = 0), rounded to 0.5 px
    for i, bs in boxes.items():
        sig[(sizes[i], tuple(sorted(tuple(round(v * 2) / 2 for v in b) for b in bs)))].append(i)
    n_sig = 0
    for members in sig.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                pair = tuple(sorted((members[x], members[y])))
                if pair not in cand:
                    cand.add(pair)
                    n_sig += 1
    print(f'stage 1: {n_thumb} pairs with thumbnail NCC >= {TH["thumb_ncc"]}, {n_sig} more with identical box lists')

    # best transform per candidate, from 64 x 64 thumbnails
    index = {int(v): n for n, v in enumerate(ids)}
    cand = sorted(cand)
    t64 = z['t64'].astype(np.float32)
    ia = np.array([index[a] for a, _ in cand])
    ib = np.array([index[b] for _, b in cand])
    Ua = unit(torch.from_numpy(t64[ia]))
    scores = torch.stack([(Ua * unit(torch.from_numpy(dihedral(t64[ib], k).copy()))).sum(1) for k in range(8)], 1)
    ks = scores.argmax(1).tolist()

    # --- Stage 2: full-resolution verification ---
    with Pool(args.workers, initializer=init, initargs=(files, sizes, boxes)) as pool:
        pairs = pool.map(verify_pair, [(a, b, k) for (a, b), k in zip(cand, ks)], chunksize=64)

    links = []
    for r in pairs:
        single = min(r['n_a'], r['n_b']) == 1
        if r['hpncc'] >= TH['copy_hpncc']:
            kind = 'copy'
        elif ((r['boxexact'] == 1.0 and r['n_a'] == r['n_b'] and r['ncc'] >= TH['revisit_ncc']
               and (not single or r['hpncc'] >= TH['revisit_single_hpncc']))
              or (r['hpncc'] >= TH['scene_hpncc'] and r['boxmatch'] >= TH['scene_boxmatch'])):
            kind = 'revisit'
        elif r['same_shape'] and (r['hpncc'] >= TH['related_hpncc']
              or (r['hpncc'] >= TH['related_hpncc_box'] and r['boxmatch'] >= TH['related_boxmatch'])
              or (r['ncc'] >= TH['related_ncc_box'] and r['boxmatch'] >= TH['related_boxmatch'])):
            kind = 'related'
        else:
            continue
        links.append(dict(kind=kind, **r))

    # --- Stage 3: shifted overlapping tiles, from boxes that reappear at one common offset ---
    by_key = defaultdict(list)
    for i, bs in boxes.items():
        for b in bs:
            by_key[(label[i], sizes[i], round((b[2] - b[0]) * 2), round((b[3] - b[1]) * 2))].append((i, b[0], b[1], b))
    votes = defaultdict(set)
    for entries in by_key.values():
        if len(entries) > 2000:  # very common tiny sizes: too unspecific to vote alone
            continue
        for x in range(len(entries)):
            for y in range(x + 1, len(entries)):
                (a, xa, ya, ba), (b, xb, yb, bb) = entries[x], entries[y]
                if a == b:
                    continue
                if a > b:
                    (a, xa, ya, ba), (b, xb, yb, bb) = (b, xb, yb, bb), (a, xa, ya, ba)
                dx, dy = round(xa - xb), round(ya - yb)  # A[y, x] = B[y - dy, x - dx]
                if (dx, dy) != (0, 0):
                    votes[(a, b, dx, dy)].add((tuple(ba), (ba[2] - ba[0]) * (ba[3] - ba[1])))
    shift_jobs = [key for key, v in votes.items()
                  if len(v) >= 2 or max(area for _, area in v) >= TH['overlap_big_box']]
    print(f'stage 3: {len(shift_jobs)} shifted candidate pairs from box votes')
    with Pool(args.workers, initializer=init, initargs=(files, sizes, boxes)) as pool:
        shifts = [r for r in pool.map(verify_shift, shift_jobs, chunksize=64) if r is not None]
    for r in shifts:
        if r['ncc'] >= TH['overlap_ncc'] and r['overlap_frac'] >= TH['overlap_min_frac']:
            links.append(dict(kind='overlap', **r))

    # --- Stage 4: revisits under a flip/rotation, from box geometry + band-pass thumbnails ---
    geo = box_geometry_matches(boxes, sizes, label)
    need = lambda n: max(1, int(np.ceil(TH['geo_min_frac'] * n)))
    geo_c = [(a, b, k, len(ma), len(mb)) for (a, b, k), (ma, mb) in geo.items()
             if len(ma) >= need(len(boxes[a])) and len(mb) >= need(len(boxes[b]))]
    bp = bandpass_thumbs(z['t64'])
    bpncc = {}
    for k in range(8):
        sel = [c for c in geo_c if c[2] == k]
        if sel:
            A = bp[[index[c[0]] for c in sel]]
            B = dihedral(bp[[index[c[1]] for c in sel]], k)
            for c, v in zip(sel, (A * B).sum((1, 2)).tolist()):
                bpncc[c[:3]] = v
    linked = {(l['a'], l['b']) for l in links} | {(l['b'], l['a']) for l in links}
    geo_jobs = {}
    for a, b, k, na, nb in geo_c:
        if bpncc[(a, b, k)] >= TH['geo_bpncc'] and (a, b) not in linked:
            if (a, b) not in geo_jobs or bpncc[(a, b, k)] > bpncc[(a, b, geo_jobs[(a, b)][2])]:
                geo_jobs[(a, b)] = (a, b, k)
    print(f'stage 4: {len(geo_c)} pairs whose boxes match under a flip/rotation, '
          f'{sum(v >= TH["geo_bpncc"] for v in bpncc.values())} of them with band-pass NCC >= {TH["geo_bpncc"]}, '
          f'{len(geo_jobs)} not linked yet', flush=True)
    with Pool(args.workers, initializer=init, initargs=(files, sizes, boxes)) as pool:
        for r in pool.map(verify_pair, sorted(geo_jobs.values()), chunksize=16):
            links.append(dict(kind='georevisit', bpncc=bpncc[(r['a'], r['b'], r['k'])], **r))

    # --- Stage 5: shifted tiles from pixels alone (boxes of the same objects may differ), and 5b: flipped + shifted ---
    # The thumbnail searches and the full-resolution checks are cached in analysis/ (they depend only on the images
    # and the thresholds, not on the other links), so a rerun with other link rules is fast.
    key = hashlib.sha256((ANALYSIS / 'train_thumbs.npz').read_bytes()).hexdigest()[:16] + json.dumps(
        {k: TH[k] for k in ('bp_sigma', 'shift_peak', 'tshift_peak', 'tshift_thumb_ncc', 'tshift_thumb_frac')})
    cache_file = ANALYSIS / 'cache_shift_search.json'
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    if cache.get('key') != key:
        cache = {'key': key, 'verified': {}}
        sc = shift_candidates(bp, ids, label, sizes)
        cache['shift'] = sorted({(a, b) for a, b, dx, dy, v, nc, fr in sc})
        cache['tshift'] = sorted((a, b, k) for (a, b), (k, v, nc, fr) in
                                 transformed_shift_candidates(bp, ids, label, sizes).items())
        cache_file.write_text(json.dumps(cache))
    comp = {i: i for i in files}  # components of the links so far: pairs already in one group need no check

    def root(i):
        while comp[i] != i:
            comp[i] = comp[comp[i]]
            i = comp[i]
        return i

    for l in links:
        comp[root(l['a'])] = root(l['b'])
    jobs = [(a, b, 0) for a, b in cache['shift'] if root(a) != root(b)]
    jobs += [(b, a, k) for a, b, k in cache['tshift'] if root(a) != root(b)]  # dihedral(a, k) ~ b shifted
    todo = [j for j in jobs if json.dumps(j) not in cache['verified']]
    print(f'stage 5: {len(cache["shift"])} unflipped and {len(cache["tshift"])} flipped/rotated shift candidates; '
          f'{len(jobs)} not in one group yet, {len(todo)} to check at full resolution', flush=True)
    with Pool(args.workers, initializer=init, initargs=(files, sizes, boxes)) as pool:
        for j, r in zip(todo, pool.map(verify_shift_fft, todo, chunksize=4)):
            cache['verified'][json.dumps(j)] = r
    cache_file.write_text(json.dumps(cache))
    for j in jobs:
        r = cache['verified'][json.dumps(j)]
        if not r or r['overlap_frac'] < TH['shift_min_frac']:
            continue
        rule = ('same' if r['hpncc'] >= TH['shift_same_hpncc'] else 'partial' if r['hpncc'] >= TH['shift_hpncc']
                else 'ncc' if r['ncc'] >= TH['shift_ncc'] else None)
        if rule:
            links.append(dict(kind='shift', rule=rule + ('_transformed' if r['k'] else ''), **r))

    # --- Stage 6: shifted overlaps with the normalised correlation (nshift.py, the critic's method), all pairs ---
    # Every square image against every image of its class and size, 8 transforms; the per-image partner lists and the
    # full-resolution checks are cached (analysis/cache_audit_npartners.json, cache_audit_nverify.json, shared with
    # audit_split.py), since they depend only on the images.
    import nshift
    srch = nshift.Search(nshift.hp32(z['t64']))
    np_file, nv_file = ANALYSIS / 'cache_audit_npartners.json', ANALYSIS / 'cache_audit_nverify.json'
    npartners = json.loads(np_file.read_text()) if np_file.exists() else {}
    nverified = json.loads(nv_file.read_text()) if nv_file.exists() else {}
    lab_arr = np.array([label[int(i)] for i in ids])
    sz_arr = np.array([sizes[int(i)] for i in ids])
    todo = [m for m in range(len(ids)) if str(int(ids[m])) not in npartners]
    print(f'stage 6: normalised shift search, {len(todo)} of {len(ids)} images not searched before', flush=True)
    for n_done, m in enumerate(todo):
        w, h = sz_arr[m]
        js = np.nonzero((lab_arr == lab_arr[m]) & (sz_arr[:, 0] == w) & (sz_arr[:, 1] == h))[0]
        js = js[js != m]
        found = []
        if w == h and len(js):
            for j, (k, pk, dx, dy, fr) in sorted(srch.partners(m, js).items()):
                if any(pk >= g0 and fr >= g1 for g0, g1 in nshift.GATES):
                    found.append([int(ids[j]), k, pk, dx, dy, fr])
        npartners[str(int(ids[m]))] = found
        if n_done % 1000 == 999:
            np_file.write_text(json.dumps(npartners))
    np_file.write_text(json.dumps(npartners))
    for l in links:
        comp[root(l['a'])] = root(l['b'])
    njobs = sorted({(j, int(b), k, dx, dy) for b, lst in npartners.items() for j, k, pk, dx, dy, fr in lst
                    if root(j) != root(int(b))})
    nnew = [jb for jb in njobs if json.dumps(jb) not in nverified]
    print(f'stage 6: {len(njobs)} candidate pairs not in one group yet, {len(nnew)} to check', flush=True)
    # nshift imports near_dups as a module (this file runs as __main__): initialise that module's globals in workers
    with Pool(args.workers, initializer=nshift.near_dups.init, initargs=(files, sizes, boxes)) as pool:
        for n_done, (jb, r) in enumerate(zip(nnew, pool.imap(nshift.refine_verify, nnew, chunksize=16))):
            nverified[json.dumps(jb)] = r
            if n_done % 20000 == 19999:
                nv_file.write_text(json.dumps(nverified))
    nv_file.write_text(json.dumps(nverified))
    for jb in njobs:
        r = nverified[json.dumps(jb)]
        if r and r[5] is not None and r[5] >= TH['shift_min_frac'] and (r[7] >= TH['shift_hpncc']
                                                                         or r[6] >= TH['shift_ncc']):
            links.append(dict(kind='nshift', rule='same' if r[7] >= TH['shift_same_hpncc'] else
                              'partial' if r[7] >= TH['shift_hpncc'] else 'ncc', a=r[0], b=r[1], k=r[2], dx=r[3],
                              dy=r[4], overlap_frac=r[5], ncc=r[6], hpncc=r[7], equal_frac=r[8], mad=r[9]))

    # --- Stage 6b: complete the links INSIDE components that can block a stratum (> 50 % of their class) ---
    # Stages 4-6 skip pairs already in one group, which is enough for grouping, but make_holdout.py's BFS ball +
    # buffer needs every neighbour of the ball: verify every candidate pair inside such components.
    for l in links:
        comp[root(l['a'])] = root(l['b'])
    members = defaultdict(list)
    for i in files:
        members[root(i)].append(i)
    n_class = Counter(label.values())
    giant_of = {i: r for r, m in members.items() if len(m) > 0.5 * n_class[label[m[0]]] for i in m}
    direct = {(l['a'], l['b']) for l in links} | {(l['b'], l['a']) for l in links}
    gjobs = sorted({(j, int(b), k, dx, dy) for b, lst in npartners.items() for j, k, pk, dx, dy, fr in lst
                    if int(b) in giant_of and giant_of.get(j) == giant_of[int(b)] and (j, int(b)) not in direct})
    gnew = [jb for jb in gjobs if json.dumps(jb) not in nverified]
    print(f'stage 6b: {len({r for r in giant_of.values()})} blocking component(s), {len(gjobs)} candidate pairs inside '
          f'them not directly linked, {len(gnew)} to check', flush=True)
    with Pool(args.workers, initializer=nshift.near_dups.init, initargs=(files, sizes, boxes)) as pool:
        for jb, r in zip(gnew, pool.imap(nshift.refine_verify, gnew, chunksize=16)):
            nverified[json.dumps(jb)] = r
    nv_file.write_text(json.dumps(nverified))
    added = set()
    for jb in gjobs:
        r = nverified[json.dumps(jb)]
        if (r and r[5] is not None and r[5] >= TH['shift_min_frac'] and (r[7] >= TH['shift_hpncc'] or r[6] >= TH['shift_ncc'])
                and (r[0], r[1]) not in added and (r[1], r[0]) not in added):
            added.add((r[0], r[1]))
            links.append(dict(kind='nshift', rule=('same' if r[7] >= TH['shift_same_hpncc'] else
                                                   'partial' if r[7] >= TH['shift_hpncc'] else 'ncc') + '_inside',
                              a=r[0], b=r[1], k=r[2], dx=r[3], dy=r[4], overlap_frac=r[5], ncc=r[6], hpncc=r[7],
                              equal_frac=r[8], mad=r[9]))
    print(f'stage 6b: {len(added)} links added inside blocking components', flush=True)

    # --- Groups: connected components (union-find) ---
    parent = {i: i for i in files}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for l in links:
        ra, rb = find(l['a']), find(l['b'])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    comps = defaultdict(list)
    for i in sorted(files):
        comps[find(i)].append(i)
    groups = sorted((m for m in comps.values() if len(m) > 1), key=lambda m: m[0])
    mixed = [g for g in groups if len({label[i] for i in g}) > 1]
    sizes_hist = defaultdict(int)
    for g in groups:
        sizes_hist[len(g)] += 1
    kinds = defaultdict(int)
    for l in links:
        kinds[l['kind']] += 1
    summary = dict(images=len(files), linked_images=sum(len(g) for g in groups), groups=len(groups),
                   singletons=len(files) - sum(len(g) for g in groups), largest_group=max(map(len, groups)),
                   links=dict(kinds), group_size_histogram=dict(sorted(sizes_hist.items())),
                   groups_with_mixed_labels=len(mixed))
    print(json.dumps(summary, indent=1))
    OUT.write_text(json.dumps(dict(description=__doc__.split('\n\n')[0], thresholds=TH, summary=summary,
                                   links_file=str((ANALYSIS / 'near_dup_links.csv').relative_to(ROOT)),
                                   groups=groups)))
    fields = ['kind', 'rule', 'a', 'b', 'k', 'dx', 'dy', 'same_shape', 'ncc', 'hpncc', 'bpncc', 'mad', 'equal_frac',
              'boxmatch', 'boxexact', 'n_a', 'n_b', 'overlap_frac', 'blocks_agree']
    with open(ANALYSIS / 'near_dup_links.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(links)
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
