"""The holdout is disjoint from its training part, class-stratified, and keeps near-duplicate groups whole."""
import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from make_holdout import draw_holdout

DATA = Path(__file__).resolve().parents[3] / 'dataset'
HOLDOUT = DATA / 'holdout'


def synthetic(n_classes=3, per_class=200, group_every=7):
    """Images 0..N-1, class = id % n_classes; every `group_every`-th image is grouped with the next of its class."""
    labels = {i: i % n_classes for i in range(n_classes * per_class)}
    buckets = {i: '512' if i % 5 else '256' for i in labels}
    grouped = {i for i in labels if i % group_every == 0 and i + n_classes in labels}
    groups = [[i, i + n_classes] for i in sorted(grouped)]
    taken = {j for g in groups for j in g}
    groups += [[i] for i in labels if i not in taken]
    return groups, labels, buckets


def test_draw_is_disjoint_stratified_and_keeps_groups_whole():
    groups, labels, buckets = synthetic()
    holdout = draw_holdout(groups, labels, buckets, fraction=0.1, seed=0)
    train = set(labels) - holdout
    assert holdout and not holdout & train
    assert all(set(g) <= holdout or set(g) <= train for g in groups)
    per_class = Counter(labels[i] for i in holdout)
    assert all(abs(per_class[c] - 20) <= 2 for c in range(3)), per_class


def test_draw_is_deterministic_per_seed():
    groups, labels, buckets = synthetic()
    assert draw_holdout(groups, labels, buckets, seed=0) == draw_holdout(groups, labels, buckets, seed=0)
    assert draw_holdout(groups, labels, buckets, seed=0) != draw_holdout(groups, labels, buckets, seed=1)


def read_ids(split):
    with open(HOLDOUT / split / 'labels.csv') as f:
        return {int(r['image_id']): int(r['label']) for r in csv.DictReader(f)}


@pytest.mark.skipif(not (HOLDOUT / 'split.json').exists(), reason='run make_holdout.py first')
def test_built_holdout_is_disjoint_complete_stratified_and_group_aware():
    train, holdout = read_ids('train'), read_ids('val')
    with open(DATA / 'train' / 'labels.csv') as f:
        everything = {int(r['image_id']): int(r['label']) for r in csv.DictReader(f)}
    assert not train.keys() & holdout.keys()  # disjoint
    assert {**train, **holdout} == everything  # complete, labels unchanged
    totals, held = Counter(everything.values()), Counter(holdout.values())
    for c, n in totals.items():  # class-stratified: 10 % of every class, within 2 images
        assert abs(held[c] - 0.1 * n) <= 2, (c, held[c], n)
    split = json.loads((HOLDOUT / 'split.json').read_text())
    assert set(split['holdout_ids']) == holdout.keys()
    for group in split['duplicate_groups']:  # no scene on both sides
        assert set(group) <= holdout.keys() or set(group) <= train.keys(), group
    for name, ids in (('train', train), ('val', holdout)):  # boxes match the images of each side
        coco = json.loads((HOLDOUT / name / 'instances.json').read_text())
        assert {img['id'] for img in coco['images']} == ids.keys()
        assert {a['image_id'] for a in coco['annotations']} == ids.keys()  # every image has >= 1 box
    assert (HOLDOUT / 'test').resolve() == (HOLDOUT / 'val').resolve()
