"""The numbers behind workspace/sarmtl/docs/DATA.md, as markdown tables. TRAIN split only (no val/test data).
Reads dataset/holdout_sarmtl/analysis/*.csv (extract_features.py), near_dup_groups.json (near_dups.py) and the
holdout (make_holdout.py). Usage (repo root): uv run python workspace/sarmtl/data/analyze_train.py > out.md
"""
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AN = ROOT / 'dataset' / 'holdout_sarmtl' / 'analysis'
NAMES = [c['name'] for c in json.loads((ROOT / 'dataset' / 'classes.json').read_text())]


def size_group(w, h):
    return {(512, 512): '512', (256, 256): '256', (224, 224): '224'}.get((w, h), 'other')


def md(df, floatfmt=1):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: '' if pd.isna(v) else f'{v:.{floatfmt}f}')
    cols = [df.index.name or ''] + [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] + ['---:'] * len(df.columns)) + '|']
    lines += ['| ' + ' | '.join([str(i)] + [str(v) for v in row]) + ' |' for i, row in zip(df.index, df.values)]
    return '\n'.join(lines)


def main():
    im = pd.read_csv(AN / 'train_image_stats.csv')
    bx = pd.read_csv(AN / 'train_box_stats.csv')
    im['cls'] = im.label.map(dict(enumerate(NAMES)))
    bx['cls'] = bx.label.map(dict(enumerate(NAMES)))
    im['sg'] = [size_group(w, h) for w, h in zip(im.width, im.height)]
    bx['sg'] = [size_group(w, h) for w, h in zip(bx.img_w, bx.img_h)]
    im['src'] = im.format + '-' + im['mode']
    bx['s512'] = np.sqrt(bx.w512 * bx.h512)
    bx['so'] = np.sqrt(bx.w * bx.h)
    bx['ar'] = bx.w512 / bx.h512
    nbox = bx.groupby('image_id').size()
    im['nbox'] = im.image_id.map(nbox)
    order = NAMES + ['all']

    print('## 1. Classes, images and boxes\n')
    t = im.groupby('cls').agg(images=('image_id', 'size'), boxes=('nbox', 'sum'), mean_boxes=('nbox', 'mean'),
                              median_boxes=('nbox', 'median'), max_boxes=('nbox', 'max'))
    t.loc['all'] = [len(im), im.nbox.sum(), im.nbox.mean(), im.nbox.median(), im.nbox.max()]
    t['img_%'] = 100 * t.images / len(im)
    t['box_%'] = 100 * t.boxes / len(bx)
    t['single_box_%'] = [100 * (im[im.cls == c].nbox == 1).mean() if c != 'all' else 100 * (im.nbox == 1).mean()
                         for c in t.index]
    for c in ('images', 'boxes', 'max_boxes'):
        t[c] = t[c].astype(int)
    t['median_boxes'] = t['median_boxes'].astype(int)
    print(md(t.loc[order, ['images', 'img_%', 'boxes', 'box_%', 'mean_boxes', 'median_boxes', 'max_boxes',
                           'single_box_%']]))
    print('\nBoxes per image (all train):', dict(sorted(Counter(im.nbox).items())))

    print('\n## 2. Image sizes, file types and the resize to 512 x 512\n')
    t = pd.crosstab(im.cls, im.sg)
    t.loc['all'] = t.sum()
    f = pd.crosstab(im.cls, im.src)
    f.loc['all'] = f.sum()
    print(md(pd.concat([t, f], axis=1).loc[order]))
    g = im.groupby('sg').agg(images=('image_id', 'size'), lapvar512=('lapvar512', 'median'),
                             speckle_cv=('cv_median', 'median'), kB=('bytes', lambda s: s.mean() / 1024))
    print('\nSharpness at the model input (median variance of the Laplacian of the 512 x 512 input) and speckle by '
          'size group:\n')
    print(md(g))
    odd = im[im.sg == 'other']
    print(f'\nOdd sizes ({len(odd)} images, classes {sorted(set(odd.cls))}): '
          + ', '.join(f'{w}x{h}' for w, h in zip(odd.width, odd.height)))
    sx, sy = 512 / odd.width, 512 / odd.height
    print(f'stretch to 512 x 512: x {sx.min():.2f}-{sx.max():.2f}, y {sy.min():.2f}-{sy.max():.2f}, '
          f'anisotropy y/x {(sy / sx).min():.2f}-{(sy / sx).max():.2f}')

    print('\n## 3. Box sizes (sqrt(w x h)), at the 512 input and in original pixels\n')

    def box_tab(gr):
        return pd.Series({
            'boxes': len(gr), 'p10_512': gr.s512.quantile(.1), 'median_512': gr.s512.median(),
            'p90_512': gr.s512.quantile(.9), 'max_512': gr.s512.max(),
            '<8_%': 100 * (gr.s512 < 8).mean(), '<16_%': 100 * (gr.s512 < 16).mean(), '<32_%': 100 * (gr.s512 < 32).mean(),
            'median_orig': gr.so.median(), 'orig<8_%': 100 * (gr.so < 8).mean(), 'orig<16_%': 100 * (gr.so < 16).mean(),
            'COCO_small_%': 100 * (gr.area < 32 ** 2).mean(),
            'COCO_medium_%': 100 * ((gr.area >= 32 ** 2) & (gr.area < 96 ** 2)).mean(),
            'COCO_large_%': 100 * (gr.area >= 96 ** 2).mean()})

    t = bx.groupby('cls').apply(box_tab, include_groups=False)
    t.loc['all'] = box_tab(bx)
    t['boxes'] = t['boxes'].astype(int)
    print(md(t.loc[order]))
    t = bx.groupby('sg').apply(box_tab, include_groups=False)
    t['boxes'] = t['boxes'].astype(int)
    print('\nBy image-size group:\n')
    print(md(t))

    def shape_tab(gr):
        mn = np.minimum(gr.w512, gr.h512)
        return pd.Series({'aspect_outside_1:2_%': 100 * ((gr.ar < 0.5) | (gr.ar > 2)).mean(),
                          'aspect_outside_1:3_%': 100 * ((gr.ar < 1 / 3) | (gr.ar > 3)).mean(),
                          'aspect_outside_1:4_%': 100 * ((gr.ar < 1 / 4) | (gr.ar > 4)).mean(),
                          'short_side<4px_%': 100 * (mn < 4).mean(), 'short_side<8px_%': 100 * (mn < 8).mean(),
                          'touches_border_%': 100 * gr.touches_border.mean()})

    t = bx.groupby('cls').apply(shape_tab, include_groups=False)
    t.loc['all'] = shape_tab(bx)
    print('\nBox shapes at the 512 input (aspect = w/h; anchors are 1:2, 1:1, 2:1):\n')
    print(md(t.loc[order]))

    print('\n## 4. Intensities (grey levels 0-255 unless noted; mean/std as fractions of 255)\n')

    def int_tab(gr):
        return pd.Series({'images': len(gr), 'mean': gr['mean'].mean(), 'sd_of_image_means': gr['mean'].std(),
                          'std_within_image': gr['std'].mean(), 'p1': gr.p1.mean(), 'p50': gr.p50.mean(),
                          'p99': gr.p99.mean(), 'zero_px_%': 100 * gr.frac0.mean(), 'sat_255_px_%': 100 * gr.frac255.mean(),
                          'speckle_cv': gr.cv_median.median(), 'in_box_mean': gr.inbox_mean.mean(),
                          'out_box_mean': gr.outbox_mean.mean()})

    t = im.groupby('cls').apply(int_tab, include_groups=False)
    t.loc['all'] = int_tab(im)
    t['images'] = t['images'].astype(int)
    print(md(t.loc[order], 3))
    print('\nBy image-size group:\n')
    t = im.groupby('sg').apply(int_tab, include_groups=False)
    t['images'] = t['images'].astype(int)
    print(md(t, 3))
    print('\nBy class x size group x file type (the likely source):\n')
    t = im.groupby(['cls', 'sg', 'src']).apply(int_tab, include_groups=False)
    t['images'] = t['images'].astype(int)
    t.index = [f'{a} {b} {c}' for a, b, c in t.index]
    print(md(t[['images', 'mean', 'std_within_image', 'p50', 'p99', 'zero_px_%', 'speckle_cv']], 3))
    m512 = im.sum512.sum() / (len(im) * 512 * 512)
    s512 = np.sqrt(im.sumsq512.sum() / (len(im) * 512 * 512) - m512 ** 2)
    print(f'\nAll train pixels at the 512 input: mean {m512:.4f}, std {s512:.4f} (sarbench MEAN, STD = 0.2526, 0.2133)')
    rgb = im[im['mode'] != 'L']
    print(f'3-channel files: {len(rgb)}; with R = G = B exactly: {(rgb.chan_maxdiff == 0).sum()}; max channel '
          f'difference {rgb.chan_maxdiff.max():.0f} levels, mean {rgb.chan_meandiff.max():.4f} (worst image)')

    bx['contrast'] = bx.in_mean / bx.ring_mean.clip(lower=1)
    t = bx.groupby('cls').agg(median_in=('in_mean', 'median'), median_ring=('ring_mean', 'median'),
                              median_ratio=('contrast', 'median'),
                              darker_than_ring_pct=('contrast', lambda s: 100 * (s < 1).mean()),
                              ratio_below_1p5_pct=('contrast', lambda s: 100 * (s < 1.5).mean()))
    print('\nObject vs local background (mean grey inside the box vs a ring of one box size around it, other boxes '
          'masked):\n')
    print(md(t.loc[NAMES], 2))

    print('\n## 5. Near-duplicates (near_dups.py)\n')
    nd = json.loads((Path(__file__).resolve().parent / 'near_dup_groups.json').read_text())
    print('summary:', json.dumps(nd['summary']))
    links = pd.read_csv(AN / 'near_dup_links.csv', low_memory=False)
    audit = Path(__file__).resolve().parent / 'audit_links.csv'  # crossing links found by build_holdout.py's audit loop
    if audit.exists():
        links = pd.concat([links, pd.read_csv(audit)], ignore_index=True)
    lab = dict(zip(im.image_id, im.label))
    links['cls'] = links.a.map(lab).map(dict(enumerate(NAMES)))
    print(md(pd.crosstab(links.cls, links.kind).loc[NAMES]))
    c = links[links.kind == 'copy']
    print('\ncopy transforms (k: 0 identity, 2 rot180, 4 left-right flip, 6 up-down flip, 1/3/5/7 rot90 family):',
          dict(Counter(c.k.astype(int))))
    parent = {int(i): int(i) for i in im.image_id}  # groups = connected components of all links (as make_holdout.py)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(links.a.astype(int), links.b.astype(int)):
        parent[find(a)] = find(b)
    roots = Counter(find(int(i)) for i in im.image_id)
    grp = {int(i): find(int(i)) for i in im.image_id if roots[find(int(i))] > 1}
    im['group'] = [grp.get(int(i), -1) for i in im.image_id]
    sizes_g = sorted(Counter(grp.values()).values(), reverse=True)
    print(f'groups from all links: {len(sizes_g)}, images in groups {sum(sizes_g)}, largest {sizes_g[:5]}')
    rows = []
    for name in NAMES:
        s = im[im.cls == name]
        grouped = s[s.group >= 0]
        units = (s.group < 0).sum() + grouped.group.nunique()
        rows.append((name, len(s), len(grouped), 100 * len(grouped) / len(s), grouped.group.nunique(), units,
                     len(s) / units))
    t = pd.DataFrame(rows, columns=['class', 'images', 'in_groups', 'in_groups_%', 'groups', 'independent_units',
                                    'images_per_unit']).set_index('class')
    t.loc['all'] = [len(im), (im.group >= 0).sum(), 100 * (im.group >= 0).mean(), im[im.group >= 0].group.nunique(),
                    (im.group < 0).sum() + im[im.group >= 0].group.nunique(), np.nan]
    t.loc['all', 'images_per_unit'] = t.loc['all', 'images'] / t.loc['all', 'independent_units']
    for col in ('images', 'in_groups', 'groups', 'independent_units'):
        t[col] = t[col].astype(int)
    print('\n' + md(t))

    # leak of a naive split: per class, a random 10 % of images; share of them with a group mate left in train
    rng = random.Random(0)
    members = im.groupby('group').image_id.apply(set).to_dict()
    leak = Counter()
    n_hold = Counter()
    for _ in range(20):
        hold = set()
        for name in NAMES:
            ids = sorted(im[im.cls == name].image_id)
            hold.update(rng.sample(ids, round(0.1 * len(ids))))
        for i, g, name in zip(im.image_id, im.group, im.cls):
            if i in hold:
                n_hold[name] += 1
                if g >= 0 and members[g] - hold:
                    leak[name] += 1
    print('\nA naive class-stratified random 10 % holdout (20 draws): share of holdout images with a copy, revisit '
          'or overlapping tile in its train side: '
          + ', '.join(f'{n} {100 * leak[n] / n_hold[n]:.0f} %' for n in NAMES)
          + f'; all {100 * sum(leak.values()) / sum(n_hold.values()):.0f} %')

    print('\n## 6. Holdout vs local train (make_holdout.py)\n')
    split = json.loads((ROOT / 'dataset' / 'holdout_sarmtl' / 'split.json').read_text())
    hold, buf = set(split['holdout_ids']), set(split.get('buffer_ids', []))
    im['side'] = np.where(im.image_id.isin(hold), 'holdout', np.where(im.image_id.isin(buf), 'buffer', 'train'))
    bx['side'] = np.where(bx.image_id.isin(hold), 'holdout', np.where(bx.image_id.isin(buf), 'buffer', 'train'))
    print(f'buffer (neither side): {len(buf)} images', dict(Counter(im[im.side == "buffer"].cls)))

    def side_tab(s):
        b = bx[bx.side == s]
        i = im[im.side == s]
        return pd.Series({'images': len(i), 'boxes': len(b), 'boxes/img': len(b) / len(i),
                          '512_%': 100 * (i.sg == '512').mean(), '256_%': 100 * (i.sg == '256').mean(),
                          'box_median_512': b.s512.median(), '<8_%': 100 * (b.s512 < 8).mean(),
                          '<16_%': 100 * (b.s512 < 16).mean(), 'COCO_small_%': 100 * (b.area < 1024).mean(),
                          'mean_grey': i['mean'].mean(), 'in_groups_%': 100 * (i.group >= 0).mean()})

    print(md(pd.DataFrame({s: side_tab(s) for s in ('train', 'holdout')}).T, 2))
    u = split['counts']['independent_units']
    print('\nIndependent units (connected components of the links within each side):\n')
    print(md(pd.DataFrame({c: [split['counts']['per_class'][c]['train_images'], u[c]['train'],
                               split['counts']['per_class'][c]['holdout_images'], u[c]['holdout'],
                               split['counts']['per_class'][c]['buffer_images']] for c in NAMES},
                          index=['train_images', 'train_units', 'holdout_images', 'holdout_units', 'buffer']).T))


if __name__ == '__main__':
    main()
