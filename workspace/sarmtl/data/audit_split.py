"""Audit of the holdout split: every pair of train images with one image on each side, scored by three detectors
that do not depend on the link rules of near_dups.py. TRAIN only, CPU only.
  thumb   NCC of 32 x 32 thumbnails, max over the 8 flips/rotations
  bp      band-pass NCC of 64 x 64 thumbnails (minus Gaussian blur, sigma 2 thumbnail px), max over the 8 transforms
  geo     boxes that match under one transform (near_dups.box_geometry_matches tolerances, >= 50 % of both sides)
  shift   cross-correlation of band-pass 64 x 64 thumbnails over all shifts and the 8 transforms (same class and
          size); pairs with a peak >= 0.3 are re-checked at full resolution under that transform (best of the 5
          highest full cross-correlation peaks): reported if the overlap is >= 5 % and its high-pass NCC >= 0.2 or
          NCC >= 0.8 (looser than the 0.35 / 0.3 / 0.9 of near_dups.py stage 5, which also uses no transform)
  nshift  the critic's method (nshift.py): correlation normalised by the overlap energies, on 32 x 32 high-passed
          thumbnails, 8 transforms, so partial overlaps score like full ones; candidates are refined at 128 px and
          at full resolution around the thumbnail shift
Prints the null distribution of bp over all pairs, the crossing pairs above thresholds and the top ones.
With --emit-links, crossing pairs that pass the link rule of near_dups.py stage 5 (overlap >= 5 %, high-pass NCC
>= 0.3 or NCC >= 0.9) are appended to workspace/sarmtl/data/audit_links.csv; make_holdout.py adds them to the links, and
build_holdout.py repeats draw + audit until no new link is found. The per-image partner search (against all images of
the same class and size, not only the current local train) and the full-resolution checks are cached in analysis/.
Usage (repo root): uv run python workspace/sarmtl/data/audit_split.py [--emit-links]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiprocessing import Pool  # noqa: E402

import near_dups  # noqa: E402
import nshift  # noqa: E402
from near_dups import ANALYSIS, DATA, ROOT, TH, bandpass_thumbs, box_geometry_matches, dihedral  # noqa: E402


def verify_shift_k(job):
    """near_dups.verify_shift_fft after transforming B by k."""
    a, b, k = job
    A, B = near_dups.load(a), np.ascontiguousarray(dihedral(near_dups.load(b), k))
    if A.shape != B.shape:
        return None
    H, W = A.shape
    ha, hb = near_dups.highpass(A), near_dups.highpass(B)
    c = np.fft.irfft2(np.fft.rfft2(ha, (2 * H, 2 * W)) * np.conj(np.fft.rfft2(hb, (2 * H, 2 * W))), (2 * H, 2 * W))
    best = None
    for f in np.argpartition(c, c.size - 5, axis=None)[-5:]:
        dy, dx = np.unravel_index(f, c.shape)
        dy, dx = int(dy - 2 * H if dy >= H else dy), int(dx - 2 * W if dx >= W else dx)
        xa0, xa1, ya0, ya1 = max(0, dx), min(W, W + dx), max(0, dy), min(H, H + dy)
        if xa1 - xa0 < 16 or ya1 - ya0 < 16:
            continue
        pa, pb = A[ya0:ya1, xa0:xa1], B[ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx]
        r = (a, b, k, dx, dy, pa.size / A.size, near_dups.ncc(pa, pb),
             near_dups.ncc(near_dups.highpass(pa), near_dups.highpass(pb)))
        if best is None or r[7] > best[7]:
            best = r
    return best

NAMES = [c['name'] for c in json.loads((ROOT / 'dataset' / 'classes.json').read_text())]


def max_over_dihedral(X):
    """X [N, s, s] unit-norm -> [N, N] max over k of <X_i, dihedral(X_j, k)>."""
    A = torch.from_numpy(X.reshape(len(X), -1))
    best = torch.full((len(X), len(X)), -2.0)
    for k in range(8):
        best = torch.maximum(best, A @ torch.from_numpy(np.ascontiguousarray(dihedral(X, k))).reshape(len(X), -1).T)
    best = torch.maximum(best, best.T)
    best.fill_diagonal_(-2.0)
    return best.numpy()


def main():
    torch.set_num_threads(16)
    z = np.load(ANALYSIS / 'train_thumbs.npz')
    ids = z['ids']
    split = json.loads((ROOT / 'dataset' / 'holdout_sarmtl' / 'split.json').read_text())
    hold = np.isin(ids, split['holdout_ids'])
    buf = np.isin(ids, split.get('buffer_ids', []))
    train = ~hold & ~buf  # buffer images are on neither side
    print(f'holdout {hold.sum()}, local train {train.sum()}, buffer {buf.sum()} images')
    coco = json.loads((DATA / 'instances.json').read_text())
    sizes = {i['id']: (i['width'], i['height']) for i in coco['images']}
    with open(DATA / 'labels.csv') as f:
        label = {int(r['image_id']): int(r['label']) for r in csv.DictReader(f)}
    boxes = {}
    for a in coco['annotations']:
        x, y, w, h = a['bbox']
        boxes.setdefault(a['image_id'], []).append([x, y, x + w, y + h])
    groups = json.loads((Path(__file__).resolve().parent / 'near_dup_groups.json').read_text())['groups']
    group = {i: n for n, g in enumerate(groups) for i in g}

    t32 = z['t32'].astype(np.float32)
    t32 -= t32.mean((1, 2), keepdims=True)
    t32 /= np.maximum(np.linalg.norm(t32.reshape(len(t32), -1), axis=1), 1e-6)[:, None, None]
    thumb = max_over_dihedral(t32)
    bp = max_over_dihedral(bandpass_thumbs(z['t64']))
    iu = np.triu_indices(len(ids), 1)
    allbp = bp[iu]
    print('band-pass NCC over all %d pairs: p50 %.3f p95 %.3f p99 %.3f p99.9 %.3f p99.99 %.3f' %
          ((len(allbp),) + tuple(np.quantile(allbp, [0.5, 0.95, 0.99, 0.999, 0.9999]))))
    cross = (hold[:, None] & train[None, :]) | (train[:, None] & hold[None, :])
    same_cls = np.array([label[int(i)] for i in ids])
    same_cls = same_cls[:, None] == same_cls[None, :]
    for name, M in (('thumb', thumb), ('bp', bp)):
        for t in ((0.6, 0.7, 0.8, 0.9) if name == 'thumb' else (0.15, 0.2, 0.3, 0.4)):
            m = np.triu(cross & (M >= t), 1)
            r, c = np.nonzero(m)
            imgs = np.unique(np.concatenate([r, c]))
            n_pairs, n_hold, n_same = int(m.sum()), int(hold[imgs].sum()), int((m & same_cls).sum())
            print(f'crossing pairs with {name} >= {t}: {n_pairs} ({n_same} same class), holdout images involved: {n_hold}')
    index = {int(v): n for n, v in enumerate(ids)}
    geo = box_geometry_matches(boxes, sizes, label)
    need = lambda n: max(1, int(np.ceil(TH['geo_min_frac'] * n)))
    geo_cross = sorted({(a, b) for (a, b, k), (ma, mb) in geo.items()
                        if len(ma) >= need(len(boxes[a])) and len(mb) >= need(len(boxes[b]))
                        and cross[index[a], index[b]]})
    print(f'crossing pairs whose boxes match under a flip/rotation: {len(geo_cross)}')
    for a, b in geo_cross[:20]:
        print(f'   {a} {b} {NAMES[label[a]]} {sizes[a]} boxes {len(boxes[a])}/{len(boxes[b])} '
              f'bp {bp[index[a], index[b]]:.3f} thumb {thumb[index[a], index[b]]:.3f}')
    # shifted tiles under the 8 transforms: every holdout image against every image of its class and square size
    # (cached per image, so a re-drawn split only searches its new holdout images), then the crossing pairs
    bpt = bandpass_thumbs(z['t64'])
    s2 = 2 * bpt.shape[1]
    sizes_arr = np.array([sizes[int(i)] for i in ids])
    lab_arr = np.array([label[int(i)] for i in ids])
    pc_file, vc_file = ANALYSIS / 'cache_audit_partners.json', ANALYSIS / 'cache_audit_verify.json'
    partners = json.loads(pc_file.read_text()) if pc_file.exists() else {}
    verified = json.loads(vc_file.read_text()) if vc_file.exists() else {}
    todo = [i for i in np.nonzero(hold)[0] if str(int(ids[i])) not in partners]
    F_all = torch.fft.rfft2(torch.from_numpy(bpt), s=(s2, s2)).conj() if todo else None
    print(f'shift search: {int(hold.sum())} holdout images, {len(todo)} not searched before', flush=True)
    for i in todo:
        w, h = sizes_arr[i]
        found = []
        js = np.nonzero((lab_arr == lab_arr[i]) & (sizes_arr[:, 0] == w) & (sizes_arr[:, 1] == h))[0]
        js = js[js != i]
        if w == h and len(js):
            Fj = F_all[js]
            best = {}
            for k in range(8):
                Fi = torch.fft.rfft2(torch.from_numpy(np.ascontiguousarray(dihedral(bpt[i], k))), s=(s2, s2))
                peak = torch.fft.irfft2(Fi[None] * Fj, s=(s2, s2)).reshape(len(js), -1).max(1).values.numpy()
                for j, v in zip(js[peak >= 0.3], peak[peak >= 0.3]):
                    if int(ids[j]) not in best or v > best[int(ids[j])][1]:
                        best[int(ids[j])] = (k, float(v))
            found = [[j, k, v] for j, (k, v) in sorted(best.items())]
        partners[str(int(ids[i]))] = found
    pc_file.write_text(json.dumps(partners))
    hold_set = {int(i) for i in ids[hold]}
    train_set = {int(i) for i in ids[train]}
    jobs = sorted({(j, int(b), k) for b in hold_set for j, k, v in partners[str(b)] if j in train_set})
    new_jobs = [jb for jb in jobs if json.dumps(jb) not in verified]
    near_dups.init({i['id']: i['file_name'] for i in coco['images']}, sizes, boxes)
    with Pool(14, initializer=near_dups.init,
              initargs=({i['id']: i['file_name'] for i in coco['images']}, sizes, boxes)) as pool:
        for jb, r in zip(new_jobs, pool.map(verify_shift_k, new_jobs, chunksize=8)):
            verified[json.dumps(jb)] = r
    vc_file.write_text(json.dumps(verified))
    res = [verified[json.dumps(jb)] for jb in jobs if verified[json.dumps(jb)]]
    hits = [r for r in res if r[5] >= 0.05 and (r[7] >= 0.2 or r[6] >= 0.8)]
    links = [r for r in res if r[5] >= TH['shift_min_frac'] and (r[7] >= TH['shift_hpncc'] or r[6] >= TH['shift_ncc'])]
    print(f'crossing same-class pairs with a shifted/transformed thumbnail peak >= 0.3: {len(jobs)} '
          f'({len(new_jobs)} checked now); confirmed at full resolution (overlap >= 5 %, high-pass NCC >= 0.2 or '
          f'NCC >= 0.8): {len(hits)}; passing the link rule (high-pass NCC >= 0.3 or NCC >= 0.9): {len(links)}')
    for a, b, k, dx, dy, fr, nc, hp in sorted(hits, key=lambda r: -r[7])[:25]:
        print(f'   train {a} ~ holdout {b} {NAMES[label[a]]} k {k} shift ({dx}, {dy}) overlap {fr:.2f} '
              f'NCC {nc:.3f} high-pass NCC {hp:.3f}')
    if res:
        hp_all = np.array([r[7] for r in res])
        print('   full-resolution overlap high-pass NCC of all checked crossing pairs: p50 %.3f p99 %.3f max %.3f'
              % (np.quantile(hp_all, 0.5), np.quantile(hp_all, 0.99), hp_all.max()))
    # normalised shifted search (the critic's method: correlation / overlap energies, so partial overlaps score fully)
    srch = nshift.Search(nshift.hp32(z['t64']))
    np_file, nv_file = ANALYSIS / 'cache_audit_npartners.json', ANALYSIS / 'cache_audit_nverify.json'
    npartners = json.loads(np_file.read_text()) if np_file.exists() else {}
    nverified = json.loads(nv_file.read_text()) if nv_file.exists() else {}
    ntodo = [i for i in np.nonzero(hold)[0] if str(int(ids[i])) not in npartners]
    print(f'normalised shift search: {int(hold.sum())} holdout images, {len(ntodo)} not searched before', flush=True)
    for i in ntodo:
        w, h = sizes_arr[i]
        js = np.nonzero((lab_arr == lab_arr[i]) & (sizes_arr[:, 0] == w) & (sizes_arr[:, 1] == h))[0]
        js = js[js != i]
        found = []
        if w == h and len(js):
            for j, (k, p, dx, dy, fr) in sorted(srch.partners(i, js).items()):
                if any(p >= g0 and fr >= g1 for g0, g1 in nshift.GATES):
                    found.append([int(ids[j]), k, p, dx, dy, fr])
        npartners[str(int(ids[i]))] = found
    np_file.write_text(json.dumps(npartners))
    njobs = sorted({(j, b, k, dx, dy) for b in hold_set for j, k, p, dx, dy, fr in npartners[str(b)] if j in train_set})
    nnew = [jb for jb in njobs if json.dumps(jb) not in nverified]
    with Pool(14, initializer=near_dups.init,
              initargs=({i['id']: i['file_name'] for i in coco['images']}, sizes, boxes)) as pool:
        for jb, r in zip(nnew, pool.map(nshift.refine_verify, nnew, chunksize=16)):
            nverified[json.dumps(jb)] = r
    nv_file.write_text(json.dumps(nverified))
    nres = [nverified[json.dumps(jb)] for jb in njobs if nverified[json.dumps(jb)]]
    nfull = [r for r in nres if r[5] is not None]
    nlinks = [r for r in nfull if r[5] >= TH['shift_min_frac'] and (r[7] >= TH['shift_hpncc'] or r[6] >= TH['shift_ncc'])]
    print(f'normalised: crossing candidates {len(njobs)} ({len(nnew)} checked now); passed the 128 px check '
          f'{len(nfull)}; passing the link rule (overlap >= 5 %, high-pass NCC >= 0.3 or NCC >= 0.9): {len(nlinks)}')
    for r in sorted(nlinks, key=lambda r: -r[7])[:25]:
        print(f'   train {r[0]} ~ holdout {r[1]} {NAMES[label[r[0]]]} k {r[2]} shift ({r[3]}, {r[4]}) overlap {r[5]:.2f} '
              f'NCC {r[6]:.3f} high-pass NCC {r[7]:.3f} equal px {r[8]:.3f}')
    links = list(links) + [list(r[:8]) for r in nlinks]

    if '--emit-links' in sys.argv:
        out = Path(__file__).resolve().parent / 'audit_links.csv'  # tracked: make_holdout.py reproduces the split from it
        old = list(csv.DictReader(open(out))) if out.exists() else []
        seen = {(int(r['a']), int(r['b'])) for r in old}
        new_rows = []
        for r in links:
            if (r[0], r[1]) in seen:
                continue
            seen.add((r[0], r[1]))
            new_rows.append(dict(kind='audit', rule='same' if r[7] >= TH['shift_same_hpncc'] else
                                 'partial' if r[7] >= TH['shift_hpncc'] else 'ncc', a=r[0], b=r[1], k=r[2], dx=r[3],
                                 dy=r[4], overlap_frac=r[5], ncc=r[6], hpncc=r[7]))
        with open(out, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=['kind', 'rule', 'a', 'b', 'k', 'dx', 'dy', 'overlap_frac', 'ncc', 'hpncc'])
            wr.writeheader()
            wr.writerows(old + new_rows)
        print(f'NEW_AUDIT_LINKS {len(new_rows)} (total {len(old) + len(new_rows)}) -> {out}')

    m = np.triu(cross, 1)
    order = np.argsort(-np.where(m, bp, -2), axis=None)[:25]
    print('top crossing pairs by band-pass NCC:')
    for flat in order:
        i, j = np.unravel_index(flat, bp.shape)
        a, b = int(ids[i]), int(ids[j])
        print(f'   {a} {b} {NAMES[label[a]]}/{NAMES[label[b]]} {sizes[a]}/{sizes[b]} bp {bp[i, j]:.3f} '
              f'thumb {thumb[i, j]:.3f} boxes {len(boxes[a])}/{len(boxes[b])}')


if __name__ == '__main__':
    main()
