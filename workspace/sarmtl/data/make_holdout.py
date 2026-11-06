"""Build dataset/holdout_sarmtl/: a fixed, leak-free local holdout of the TRAIN split, laid out so that the unchanged
root train.py reads it (python train.py --data dataset/holdout_sarmtl ...).

  train/  labels.csv, instances.json, images -> ../../train/images    local train (~90 % of train minus the buffer)
  val/    labels.csv, instances.json, images -> ../../train/images    the holdout (~10 % of every class)
  test -> val                                                          root train.py always scores a test split
  split.json                                                           how it was drawn, checksums, counts, ids
  halves.json                                                          {"A": ids, "B": ids}: group-aware halves of val

near_dups.py links train images that share pixels or objects (copies, revisits, overlapping tiles; see its docstring).
The holdout is drawn by units, the connected components of those links, stratified by class x image-size group
(512 / 256 / 224 / other). Each class gets round(fraction x its images), shared over its strata by largest remainder.
In each stratum, the units are taken while they fit the target, in a stable random order: every image has the key
sha256(f'{seed}-{image_id}'), a unit has the smallest key of its images. Adding a link then moves only the units it
merges, so the draw + audit loop of build_holdout.py converges. Where one giant component makes the target unreachable this way (shortfall > 1 image), the stratum
takes whole units except the giant first. The rest of its target is then a BFS ball of the giant's link graph (the
seed, out of up to 64 drawn with random.Random(f'{seed}-ball-{stratum}'), whose ball has the smallest buffer). The
ball's direct neighbours are a BUFFER, excluded from local train and from the holdout, so that no link of any kind
crosses holdout <-> local train. The full-train submission run still uses all 9,392 images. Rerunning gives the
identical split. No val/test data is read.

Usage (repo root): uv run python workspace/sarmtl/data/make_holdout.py [--fraction 0.1] [--seed 0]
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'dataset' / 'train'
OUT = ROOT / 'dataset' / 'holdout_sarmtl'
GROUPS = Path(__file__).resolve().parent / 'near_dup_groups.json'
LINKS = OUT / 'analysis' / 'near_dup_links.csv'
AUDIT_LINKS = Path(__file__).resolve().parent / 'audit_links.csv'  # crossing links found by audit_split.py --emit-links
FROZEN = Path(__file__).resolve().parent / 'frozen_holdout.json'  # the frozen holdout ids (lead's final procedure)
SWEEP = Path(__file__).resolve().parent / 'sweep_buffer.json'  # local-train images swept into the buffer
CRITIC_H3 = Path(__file__).resolve().parent / 'critic_h3_buffer.json'  # the critic's residual partners (lead's H3)
CRITIC_PAIRS = [(4766, 1301), (2579, 3272), (2616, 4100), (5663, 4506), (5408, 7279), (3184, 5036), (8523, 1265),
                (1650, 680)]  # (holdout, local-train) pairs the critic found leaking on 23-24 Oct: must not cross
NAMES = [c['name'] for c in json.loads((ROOT / 'dataset' / 'classes.json').read_text())]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def size_group(w, h):
    return {(512, 512): '512', (256, 256): '256', (224, 224): '224'}.get((w, h), 'other')


def largest_remainder(quotas, total):
    """Integers that sum to `total`, each the floor of its quota plus at most 1, by largest remainder."""
    base = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - base[k]), k))[:total - sum(base.values())]:
        base[k] += 1
    return base


def greedy(units, target):
    """Units taken in the given order while they fit the target."""
    taken, n = [], 0
    for u in units:
        if n + len(u) <= target:
            taken.append(u)
            n += len(u)
    return taken, n


def best_ball(members, r, adj, rng, tries=64):
    """BFS ball of exactly r images inside one component (neighbours in id order), from the seed whose ball has the
    fewest outside neighbours (the buffer); ties go to the smaller seed id."""
    inside = set(members)
    seeds = sorted(members) if len(members) <= tries else sorted(rng.sample(sorted(members), tries))
    best = None
    for s0 in seeds:
        order, seen, q = [s0], {s0}, deque([s0])
        while q and len(order) < r:
            for w in sorted(adj[q.popleft()]):
                if w in inside and w not in seen:
                    seen.add(w)
                    order.append(w)
                    q.append(w)
                    if len(order) == r:
                        break
        if len(order) < r:
            continue
        ball = set(order)
        buf = {w for v in ball for w in adj[v]} - ball
        if best is None or (len(buf), s0) < best[0]:
            best = ((len(buf), s0), ball, buf)
    return best


def halves(hold, adj, label, seed):
    """Group-aware halves of the holdout: units = connected components of the links among holdout images; per class,
    units shuffled (seed) then sorted by size (largest first, stable), each given to the half with fewer images."""
    comp, units = {}, []
    for i in sorted(hold):
        if i in comp:
            continue
        unit, q = [i], deque([i])
        comp[i] = len(units)
        while q:
            for w in adj[q.popleft()]:
                if w in hold and w not in comp:
                    comp[w] = len(units)
                    unit.append(w)
                    q.append(w)
        units.append(sorted(unit))
    rng = random.Random(f'{seed}-halves')
    A, B = set(), set()
    for c in range(len(NAMES)):
        us = [u for u in units if label[u[0]] == c]
        rng.shuffle(us)
        us.sort(key=len, reverse=True)
        na = nb = 0
        for u in us:
            if na <= nb:
                A.update(u)
                na += len(u)
            else:
                B.update(u)
                nb += len(u)
    return A, B, units


def representativeness(label, images, tr_ids, ho_ids):
    """Known gaps of the holdout. The near-black 256^2 ship chips (one look-alike-chained group, see near_dups.py) are
    counted per side from analysis/train_image_stats.csv (extract_features.py)."""
    stats = OUT / 'analysis' / 'train_image_stats.csv'
    if not stats.exists():
        return {}
    with open(stats) as f:
        dark = {int(r['image_id']) for r in csv.DictReader(f) if int(r['label']) == NAMES.index('ship')
                and (int(r['width']), int(r['height'])) == (256, 256) and r['mode'] != 'L' and float(r['mean']) < 0.05}
    ships = sum(1 for i in label if label[i] == NAMES.index('ship'))
    return dict(dark_ship_chips=dict(
        definition='ship, 256 x 256, 3-channel JPEG, mean grey < 0.05 (of 255 scaled to 1)', total=len(dark),
        share_of_ships=round(len(dark) / ships, 3), local_train=len(dark & tr_ids), holdout=len(dark & ho_ids),
        note='lead decision (a): leak-safe; the dark 256^2 ship source (~45 % of ships) is not represented in the '
             'holdout, so holdout ship AP covers the brighter ship chips only'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fraction', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    with open(SRC / 'labels.csv') as f:
        rows = list(csv.DictReader(f))
    coco = json.loads((SRC / 'instances.json').read_text())
    images = {img['id']: img for img in coco['images']}
    label = {int(r['image_id']): int(r['label']) for r in rows}
    assert len(rows) == len(images) == len(label) == 9392, 'expected the 9,392 train images'
    assert {r['file_name'] for r in rows} == {img['file_name'] for img in coco['images']}
    assert all(images[int(r['image_id'])]['file_name'] == r['file_name'] for r in rows)

    # Units (connected components of all near-duplicate links) and the link graph
    nd = json.loads(GROUPS.read_text())
    adj = defaultdict(set)
    with open(LINKS) as f:
        links = [(int(r['a']), int(r['b']), r['kind']) for r in csv.DictReader(f)]
    n_nd_links = len(links)
    if AUDIT_LINKS.exists():
        with open(AUDIT_LINKS) as f:
            links += [(int(r['a']), int(r['b']), r['kind']) for r in csv.DictReader(f)]
    for a, b, _ in links:
        adj[a].add(b)
        adj[b].add(a)
    grouped, seen = [], set()  # groups = connected components of all links (near_dups.py + audit)
    for i in sorted(adj):
        if i in seen:
            continue
        comp, q = [i], deque([i])
        seen.add(i)
        while q:
            for w in adj[q.popleft()]:
                if w not in seen:
                    seen.add(w)
                    comp.append(w)
                    q.append(w)
        grouped.append(sorted(comp))
    in_group = {i for g in grouped for i in g}
    assert {i for g in nd['groups'] for i in g} <= in_group, 'near_dups.py groups must be contained in the components'
    units = grouped + [[i] for i in sorted(label) if i not in in_group]
    stratum_of = {}
    for u in units:
        keys = {(label[i], size_group(images[i]['width'], images[i]['height'])) for i in u}
        assert len({k[0] for k in keys}) == 1, f'group {u[:5]} mixes classes'
        stratum_of[u[0]] = min(keys)  # (a group spanning size groups, if any, joins its smallest key)
    strata = defaultdict(list)
    for u in units:
        strata[stratum_of[u[0]]].append(u)

    # Targets: round(fraction x class) per class, split over the class's strata by largest remainder
    n_stratum = {s: sum(map(len, us)) for s, us in strata.items()}
    target = {}
    for c in range(len(NAMES)):
        quotas = {s: args.fraction * n for s, n in n_stratum.items() if s[0] == c}
        target.update(largest_remainder(quotas, round(args.fraction * sum(n_stratum[s] for s in quotas))))

    key = {i: hashlib.sha256(f'{args.seed}-{i}'.encode()).hexdigest() for i in label}
    holdout, buffer, blocks = set(), set(), []
    for s in sorted(strata):
        us = sorted(strata[s], key=lambda u: min(key[i] for i in u))
        taken, n = greedy(us, target[s])
        if target[s] - n > 1:  # a giant component blocks the target: whole units first, then a BFS ball
            giant = max(us, key=lambda u: (len(u), -u[0]))
            taken, n = greedy([u for u in us if u is not giant], target[s])
            ball_rng = random.Random(f'{args.seed}-ball-{NAMES[s[0]]}-{s[1]}')
            (nbuf, seed0), ball, buf = best_ball(giant, target[s] - n, adj, ball_rng)
            holdout.update(ball)
            buffer.update(buf)
            blocks.append(dict(stratum=f'{NAMES[s[0]]}|{s[1]}', group_size=len(giant), ball_seed=seed0,
                               ball=len(ball), buffer=len(buf), whole_units_images=n))
        for u in taken:
            holdout.update(u)
    ball_buffer = set(buffer)
    sweep = set(json.loads(SWEEP.read_text())['ids']) if SWEEP.exists() else set()
    h3 = json.loads(CRITIC_H3.read_text()) if CRITIC_H3.exists() else {'strong': [], 'wind_turbine': []}
    h3_ids = set(h3['strong']) | set(h3['wind_turbine'])
    assert not (sweep | h3_ids) & holdout, 'the sweep and the H3 list may only move local-train images'
    reasons = defaultdict(list)
    for i in ball_buffer:
        reasons[i].append('block')
    for i in sweep:
        reasons[i].append('sweep')
    for key in ('strong', 'wind_turbine'):
        for i in h3[key]:
            reasons[i].append(f'critic_h3_{key}')
    buffer |= sweep | h3_ids
    train_ids = sorted(set(label) - holdout - buffer)
    hold_ids = sorted(holdout)

    # Write the layout
    for split in ('train', 'val'):
        (OUT / split).mkdir(parents=True, exist_ok=True)
        link = OUT / split / 'images'
        if not link.is_symlink():
            link.symlink_to('../../train/images')
    if not (OUT / 'test').is_symlink():
        (OUT / 'test').symlink_to('val')
    by_id = {int(r['image_id']): r for r in rows}
    for split, ids in (('train', train_ids), ('val', hold_ids)):
        keep = set(ids)
        with open(OUT / split / 'labels.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['image_id', 'file_name', 'label'])
            w.writeheader()
            w.writerows(by_id[i] for i in ids)
        side = {'images': [img for img in coco['images'] if img['id'] in keep],
                'annotations': [a for a in coco['annotations'] if a['image_id'] in keep],
                'categories': coco['categories']}
        (OUT / split / 'instances.json').write_text(json.dumps(side))

    # Assertions, on the files as written
    side = {}
    for split in ('train', 'val'):
        with open(OUT / split / 'labels.csv') as f:
            r = list(csv.DictReader(f))
        side[split] = (r, json.loads((OUT / split / 'instances.json').read_text()))
    (tr_rows, tr_coco), (ho_rows, ho_coco) = side['train'], side['val']
    tr_ids, ho_ids = {int(r['image_id']) for r in tr_rows}, {int(r['image_id']) for r in ho_rows}
    tr_files, ho_files = {r['file_name'] for r in tr_rows}, {r['file_name'] for r in ho_rows}
    checks = []

    def check(cond, msg):
        assert cond, msg
        checks.append(f'OK  {msg}')

    if FROZEN.exists():
        check(ho_ids == set(json.loads(FROZEN.read_text())['holdout_ids']),
              'the drawn holdout equals the frozen holdout (frozen_holdout.json)')
    check(not sweep & ho_ids, f'the buffer sweep ({len(sweep)} images) removes local-train images only')
    check(not h3_ids & (tr_ids | ho_ids), f'regression: none of the critic\'s {len(h3_ids)} H3 partners is in local train '
          f'or holdout ({len(h3_ids - sweep - ball_buffer)} added by the H3 list itself)')
    check(not any((h in ho_ids and t in tr_ids) for h, t in CRITIC_PAIRS),
          f'regression: none of the critic\'s {len(CRITIC_PAIRS)} leaking pairs crosses holdout <-> local train')
    check(not tr_ids & ho_ids and not tr_files & ho_files, 'train and holdout are disjoint (image ids and file names)')
    check(not buffer & (tr_ids | ho_ids), f'the buffer ({len(buffer)} images) is in neither train nor holdout')
    check(tr_ids | ho_ids | buffer == set(label) and len(tr_ids) + len(ho_ids) + len(buffer) == 9392,
          'train + holdout + buffer = all 9,392 train images')
    check(len(tr_rows) == len(tr_ids) and len(ho_rows) == len(ho_ids), 'no image is listed twice')
    for name, rs, cc in (('train', tr_rows, tr_coco), ('holdout', ho_rows, ho_coco)):
        ids = {int(r['image_id']) for r in rs}
        check(all(r == by_id[int(r['image_id'])] for r in rs), f'{name}: labels.csv rows equal the source rows')
        check(sorted(cc['images'], key=lambda i: i['id']) == [images[i] for i in sorted(ids)],
              f'{name}: COCO images equal the source entries of its images')
        src_anns = sorted((a for a in coco['annotations'] if a['image_id'] in ids), key=lambda a: a['id'])
        check(sorted(cc['annotations'], key=lambda a: a['id']) == src_anns,
              f'{name}: boxes equal the boxes of dataset/train/instances.json for its images ({len(src_anns)})')
        check(cc['categories'] == coco['categories'], f'{name}: categories equal the source categories')
        check({int(r['label']) for r in rs} == set(range(len(NAMES))), f'{name}: every class is present')
    crossing = [(a, b, k) for a, b, k in links if (a in ho_ids and b in tr_ids) or (a in tr_ids and b in ho_ids)]
    check(not crossing, f'no link of any kind crosses holdout <-> local train ({len(links)} links checked)')
    split_groups = [g for g in grouped if set(g) & ho_ids and set(g) & tr_ids]
    n_blocked = sum(1 for g in grouped if set(g) & ball_buffer)
    check(all(set(g) & ball_buffer for g in split_groups) and n_blocked == len(blocks),
          f'only the {len(blocks)} buffered giant group(s) have images on both sides; the other '
          f'{len(grouped) - n_blocked} groups lie wholly on one side')
    n_box = Counter(a['image_id'] for a in coco['annotations'])
    n_ann = len(tr_coco['annotations']) + len(ho_coco['annotations'])
    check(n_ann + sum(n_box[i] for i in buffer) == len(coco['annotations']) == 32533,
          'train + holdout + buffer boxes = all 32,533 boxes')
    A, B, h_units = halves(ho_ids, adj, label, args.seed)
    check(A | B == ho_ids and not A & B and not any((a in A and b in B) or (a in B and b in A) for a, b, _ in links),
          f'halves A ({len(A)}) and B ({len(B)}) cover the holdout, are disjoint and no link joins them')

    # Counts per class
    cls_all = Counter(label.values())
    counts, lines = {}, [f'{"class":<13}{"train img":>10}{"holdout img":>12}{"holdout %":>10}{"buffer":>7}'
                         f'{"train box":>10}{"holdout box":>12}{"half A/B":>10}']
    for c, name in enumerate(NAMES):
        ti, hi = sum(label[i] == c for i in tr_ids), sum(label[i] == c for i in ho_ids)
        bi = sum(label[i] == c for i in buffer)
        tb = sum(n_box[i] for i in tr_ids if label[i] == c)
        hb = sum(n_box[i] for i in ho_ids if label[i] == c)
        ha, hb_ = sum(label[i] == c for i in A), sum(label[i] == c for i in B)
        counts[name] = dict(train_images=ti, holdout_images=hi, buffer_images=bi, train_boxes=tb, holdout_boxes=hb,
                            half_A=ha, half_B=hb_)
        lines.append(f'{name:<13}{ti:>10}{hi:>12}{100 * hi / cls_all[c]:>10.1f}{bi:>7}{tb:>10}{hb:>12}'
                     f'{f"{ha}/{hb_}":>10}')
    tb, hb = len(tr_coco['annotations']), len(ho_coco['annotations'])
    lines.append(f'{"all":<13}{len(tr_ids):>10}{len(ho_ids):>12}{100 * len(ho_ids) / 9392:>10.1f}{len(buffer):>7}'
                 f'{tb:>10}{hb:>12}{f"{len(A)}/{len(B)}":>10}')
    per_stratum = Counter((label[i], size_group(images[i]['width'], images[i]['height'])) for i in ho_ids)
    strata_counts = {f'{NAMES[s[0]]}|{s[1]}': dict(images=n_stratum[s], target=target[s], holdout=per_stratum[s])
                     for s in sorted(strata)}

    def n_units(ids):  # connected components of the links restricted to `ids`
        seen, n = set(), 0
        for i in ids:
            if i in seen:
                continue
            n += 1
            q = deque([i])
            seen.add(i)
            while q:
                for w in adj[q.popleft()]:
                    if w in ids and w not in seen:
                        seen.add(w)
                        q.append(w)
        return n

    units_count = {name: dict(holdout=n_units({i for i in ho_ids if label[i] == c}),
                              train=n_units({i for i in tr_ids if label[i] == c}))
                   for c, name in enumerate(NAMES)}
    units_count['all'] = dict(holdout=sum(v['holdout'] for v in units_count.values()),
                              train=sum(v['train'] for v in units_count.values()))
    split = dict(
        method='group-aware, leak-free stratified holdout of dataset/train (see make_holdout.py docstring): units = '
               'connected components of the near-duplicate links (near_dups.py); strata = class x image-size group; '
               'giant components that block a stratum target are split into a BFS ball (holdout) + its direct '
               'neighbours (buffer, excluded from local train) + the rest (local train)',
        command='uv run python workspace/sarmtl/data/extract_features.py && '
                'uv run python workspace/sarmtl/data/near_dups.py && '
                'uv run python workspace/sarmtl/data/build_holdout.py && '
                'uv run python workspace/sarmtl/data/sweep_buffer.py && '
                f'uv run python workspace/sarmtl/data/make_holdout.py --fraction {args.fraction} --seed {args.seed}',
        fraction=args.fraction, seed=args.seed,
        note='the buffer is excluded from LOCAL experiments only; the full-train submission run uses all 9,392 images',
        sources={str(pth.relative_to(ROOT) if pth.is_relative_to(ROOT) else pth): sha256(pth) for pth in
                 (SRC / 'labels.csv', SRC / 'instances.json', ROOT / 'dataset' / 'classes.json', GROUPS, LINKS)
                 + tuple(x for x in (AUDIT_LINKS, FROZEN, SWEEP, CRITIC_H3) if x.exists())},
        near_dup_summary=dict(nd['summary'], note='near_dups.py groups only, before the audit links; the final groups '
                              'and units used by this split are in "links"'),
        links=dict(near_dups=n_nd_links, audit=len(links) - n_nd_links, groups=len(grouped),
                   images_in_groups=len(in_group), units=len(grouped) + len(label) - len(in_group)),
        counts=dict(train_images=len(tr_ids), holdout_images=len(ho_ids), buffer_images=len(buffer),
                    train_boxes=tb, holdout_boxes=hb, per_class=counts, per_stratum=strata_counts,
                    independent_units=units_count,
                    halves=dict(A=len(A), B=len(B), holdout_units=len(h_units),
                                per_class={name: dict(A_images=sum(label[i] == c for i in A),
                                                      B_images=sum(label[i] == c for i in B),
                                                      A_units=n_units({i for i in A if label[i] == c}),
                                                      B_units=n_units({i for i in B if label[i] == c}))
                                           for c, name in enumerate(NAMES)})),
        largest_group_per_class={NAMES[c]: max([len(g) for g in grouped if label[g[0]] == c] or [1])
                                 for c in range(len(NAMES))},
        representativeness=representativeness(label, images, tr_ids, ho_ids),
        buffered_blocks=blocks,
        buffer_ball_ids=sorted(ball_buffer), buffer_sweep_ids=sorted(sweep), buffer_h3_ids=sorted(h3_ids),
        buffer_reasons={str(i): reasons[i] for i in sorted(buffer)},
        buffer_per_class={NAMES[c]: dict(total=sum(label[i] == c for i in buffer),
                                         block=sum(label[i] == c for i in ball_buffer),
                                         sweep=sum(label[i] == c for i in sweep),
                                         h3_added=sum(label[i] == c for i in h3_ids - sweep - ball_buffer))
                          for c in range(len(NAMES))},
        assertions=checks,
        holdout_ids=hold_ids,
        buffer_ids=sorted(buffer))
    (OUT / 'split.json').write_text(json.dumps(split, indent=1))
    (OUT / 'halves.json').write_text(json.dumps({'A': sorted(A), 'B': sorted(B)}))
    print('\n'.join(checks))
    for b in blocks:
        print(f'buffered block {b}')
    print('\n'.join(lines))
    print(f'-> {OUT} (split.json: {len(hold_ids)} holdout ids, {len(buffer)} buffer ids; halves.json)')


if __name__ == '__main__':
    main()
