"""Buffer sweep, the lead's final procedure. The holdout is FROZEN (frozen_holdout.json). Every crossing detector runs
once over holdout x local train (same class and size, all 8 transforms, all shifts). Every local-train image of a
detected crossing pair moves into the buffer (workspace/sarmtl/data/sweep_buffer.json, which make_holdout.py
applies). Removing local-train images cannot create a crossing, so one sweep terminates. `--check` re-runs the
detectors on the swept split and exits 1 if any crossing remains. TRAIN only, CPU only.

Detectors:
  links    every near-duplicate link (analysis/near_dup_links.csv, audit_links.csv) between holdout and local train
  shift    the raw-peak band-pass search of audit_split.py (its cached partners and checks, computed on demand)
  nshift   the critic's normalised search: mean-subtracted 32 px high-passed thumbnails, NCC >= 0.5 on >= 10 %
           overlap; 64 px thumbnails for 5-10 % overlaps. Then a 128 px check (>= 0.3, as the critic) and a
           full-resolution confirmation by the link rule (overlap >= 5 %, high-pass NCC >= 0.3 or NCC >= 0.9); for
           the 5-10 % overlaps high-pass NCC >= 0.5, from that detector's null test (unrelated pairs never reach it).
  critic   the critic's verified pair list (regression test)
Usage (repo root): uv run python workspace/sarmtl/data/sweep_buffer.py [--check] [--sizes=32,64]
"""
import csv
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import near_dups  # noqa: E402
import nshift  # noqa: E402
from audit_split import verify_shift_k  # noqa: E402
from near_dups import ANALYSIS, DATA, ROOT, TH  # noqa: E402

OUT = ROOT / 'dataset' / 'holdout_sarmtl'
SWEEP = HERE / 'sweep_buffer.json'
NAMES = [c['name'] for c in json.loads((ROOT / 'dataset' / 'classes.json').read_text())]
CRITIC_PAIRS = [(4766, 1301), (2579, 3272), (2616, 4100), (5663, 4506), (5408, 7279), (3184, 5036), (8523, 1265),
                (1650, 680)]  # (holdout id, local-train id) verified by the critic at full resolution, 23-24 Oct
PRECHECK_128 = 0.3
# The 64 px search for 5-10 % overlaps needs a higher bar: on unrelated pairs (different groups, 80 per class) its
# best small-overlap match reaches high-pass NCC >= 0.3 in 10 % (wind turbine), 2.5 % (harbor), 1.2 % (aircraft) of
# cases, max 0.487, and never 0.5; the 32 px search (>= 10 % overlap) never reaches 0.3 on them (max 0.253).
SMALL_OVERLAP_HPNCC = 0.5


def link_rule(r):
    """(a, b, k, dx, dy, overlap, NCC, high-pass NCC, ...) -> True if it is a link (near_dups.py stage 4 rule)."""
    return (r is not None and r[5] is not None and r[5] >= TH['shift_min_frac']
            and (r[7] >= TH['shift_hpncc'] or r[6] >= TH['shift_ncc']))


def main():
    check = '--check' in sys.argv
    torch.set_num_threads(14)
    frozen = json.loads((HERE / 'frozen_holdout.json').read_text())
    split = json.loads((OUT / 'split.json').read_text())
    H = set(split['holdout_ids'])
    assert H == set(frozen['holdout_ids']), 'the holdout must be the frozen one'
    coco = json.loads((DATA / 'instances.json').read_text())
    files = {i['id']: i['file_name'] for i in coco['images']}
    sizes = {i['id']: (i['width'], i['height']) for i in coco['images']}
    with open(DATA / 'labels.csv') as f:
        label = {int(r['image_id']): int(r['label']) for r in csv.DictReader(f)}
    T = set(label) - H - set(split['buffer_ids'])  # local train of the split on disk
    print(f'{"check" if check else "sweep"}: holdout {len(H)}, local train {len(T)}, buffer {len(split["buffer_ids"])}',
          flush=True)
    hits = []  # (holdout id, train id, detector, details)

    # links of any kind
    for path in (ANALYSIS / 'near_dup_links.csv', HERE / 'audit_links.csv'):
        with open(path) as f:
            for r in csv.DictReader(f):
                a, b = int(r['a']), int(r['b'])
                if (a in H and b in T) or (b in H and a in T):
                    hits.append((a if a in H else b, b if a in H else a, 'link:' + r['kind'], {}))
    # the critic's verified pairs
    for h, t in CRITIC_PAIRS:
        if h in H and t in T:
            hits.append((h, t, 'critic', {}))

    z = np.load(ANALYSIS / 'train_thumbs.npz')
    ids = z['ids']
    index = {int(v): n for n, v in enumerate(ids)}
    lab_arr = np.array([label[int(i)] for i in ids])
    sz_arr = np.array([sizes[int(i)] for i in ids])
    T_mask = np.isin(ids, sorted(T))
    cache_file = ANALYSIS / 'cache_sweep.json'
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    cache.setdefault('verified', {})

    # raw-peak shift search (audit_split.py): cached partners of holdout images; check the crossing ones
    partners = json.loads((ANALYSIS / 'cache_audit_partners.json').read_text())
    jobs = sorted({(j, b, k) for b in H for j, k, v in partners.get(str(b), []) if j in T})
    missing = [b for b in H if str(b) not in partners]
    assert not missing, f'{len(missing)} holdout images lack raw-peak partners: run audit_split.py first'
    verified = json.loads((ANALYSIS / 'cache_audit_verify.json').read_text())
    todo = [jb for jb in jobs if json.dumps(jb) not in verified]
    with Pool(14, initializer=near_dups.init, initargs=(files, sizes, {})) as pool:
        for jb, r in zip(todo, pool.map(verify_shift_k, todo, chunksize=8)):
            verified[json.dumps(jb)] = r
    for jb in jobs:
        r = verified[json.dumps(jb)]
        if link_rule(r):
            hits.append((r[1], r[0], 'shift', dict(k=r[2], dx=r[3], dy=r[4], overlap=r[5], ncc=r[6], hpncc=r[7])))
    n_shift_jobs = len(jobs)

    # normalised search (the critic's spec): 32 px for >= 10 % overlap, 64 px for 5-10 %
    n_cand = {}
    sizes_run = [int(x) for x in next((a.split('=')[1] for a in sys.argv if a.startswith('--sizes=')), '32,64').split(',')]
    for size, lo, hi in [g for g in ((32, 0.10, 1.01), (64, 0.05, 0.10)) if g[0] in sizes_run]:
        key = f'partners{size}'
        cache.setdefault(key, {})
        srch = None
        todo_h = [b for b in sorted(H) if str(b) not in cache[key]]
        if todo_h:
            srch = nshift.Search(nshift.hp32(z['t64'], size, center=True), min_frac=lo, peak=0.5)
        for b in todo_h:
            i = index[b]
            w, h = sizes[b]
            js = np.nonzero((lab_arr == lab_arr[i]) & (sz_arr[:, 0] == w) & (sz_arr[:, 1] == h))[0]
            js = js[js != i]
            found = []
            if w == h and len(js):
                for j, (k, pk, dx, dy, fr) in sorted(srch.partners(i, js).items()):
                    if lo <= fr < hi:
                        found.append([int(ids[j]), k, pk, dx, dy, fr])
            cache[key][str(b)] = found
        cache_file.write_text(json.dumps(cache))
        njobs = sorted({(j, b, k, dx, dy, size) for b in H for j, k, pk, dx, dy, fr in cache[key][str(b)] if j in T})
        n_cand[size] = len(njobs)
        todo = [jb for jb in njobs if json.dumps(jb) not in cache['verified']]
        print(f'nshift{size}: {len(njobs)} crossing candidates, {len(todo)} to check', flush=True)
        nshift.PRECHECK = PRECHECK_128
        with Pool(14, initializer=near_dups.init, initargs=(files, sizes, {})) as pool:
            for jb, r in zip(todo, pool.imap(nshift.refine_verify, todo, chunksize=16)):
                cache['verified'][json.dumps(jb)] = r
        cache_file.write_text(json.dumps(cache))
        for jb in njobs:
            r = cache['verified'][json.dumps(jb)]
            if link_rule(r) and (size == 32 or r[7] >= SMALL_OVERLAP_HPNCC):
                hits.append((r[1], r[0], f'nshift{size}', dict(k=r[2], dx=r[3], dy=r[4], overlap=r[5], ncc=r[6],
                                                               hpncc=r[7], equal_px=r[8])))

    by_det = Counter(d for _, _, d, _ in hits)
    swept = sorted({t for _, t, _, _ in hits})
    print(f'crossing pairs found: {len(hits)} {dict(by_det)}; local-train images involved: {len(swept)}; '
          f'raw-peak crossing candidates checked: {n_shift_jobs}; normalised crossing candidates: {n_cand}', flush=True)
    per_class = Counter(NAMES[label[t]] for t in swept)
    n_train = Counter(NAMES[label[t]] for t in T)
    print('local-train images per class that the sweep moves to the buffer:',
          {c: f'{per_class[c]} of {n_train[c]} ({100 * per_class[c] / n_train[c]:.1f} %)' for c in NAMES}, flush=True)
    if check:
        if hits:
            print('CHECK FAILED: crossings remain after the sweep', hits[:10])
            sys.exit(1)
        print('CHECK OK: 0 crossing pairs under every detector')
        return
    SWEEP.write_text(json.dumps(dict(
        description=__doc__.split('\n\n')[0], frozen_val_labels_md5=frozen['val_labels_md5'],
        ids=swept, per_class=dict(per_class), detectors=dict(by_det),
        pairs=[[h, t, d, x] for h, t, d, x in sorted(hits, key=lambda x: (x[1], x[0], x[2]))]), indent=0))
    print(f'-> {SWEEP} ({len(swept)} local-train images)')


if __name__ == '__main__':
    main()
