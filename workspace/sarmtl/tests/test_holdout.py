"""The local holdout dataset/holdout_sarmtl/ (workspace/sarmtl/data/, DATA.md §6): disjoint from local train and the
buffer, together all of train, every class held out at 10 %, boxes equal to the source, the halves a partition of the
holdout, and the critic's flagged overlap partners kept off local train."""
import csv
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DATA, HOLDOUT = ROOT / 'dataset', ROOT / 'dataset' / 'holdout_sarmtl'
pytestmark = pytest.mark.skipif(not HOLDOUT.exists(), reason='dataset/holdout_sarmtl is not built')
# The critic's last 33 overlap partners in local train (REVIEW.md §5, H3), and 2 of its 8 regression pairs (H2)
H3_PARTNERS = [160, 467, 998, 2980, 3127, 4890, 5867, 6424, 6551, 6752, 7084, 7472, 7696, 14, 272, 430, 779, 1230,
               1298, 1397, 1439, 2472, 3221, 4421, 5055, 5415, 5433, 5440, 6106, 6645, 7686, 7999, 8169]
H2_PAIRS = [(4766, 1301), (2579, 3272)]


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope='module')
def split():
    train = {int(r['image_id']): r for r in rows(HOLDOUT / 'train' / 'labels.csv')}
    holdout = {int(r['image_id']): r for r in rows(HOLDOUT / 'val' / 'labels.csv')}
    buffer = set(json.loads((HOLDOUT / 'split.json').read_text())['buffer_ids'])
    source = {int(r['image_id']): r for r in rows(DATA / 'train' / 'labels.csv')}
    return train, holdout, buffer, source


def test_holdout_local_train_and_buffer_are_disjoint_and_cover_train(split):
    train, holdout, buffer, source = split
    assert not (train.keys() & holdout.keys()) and not (train.keys() & buffer) and not (holdout.keys() & buffer)
    assert train.keys() | holdout.keys() | buffer == source.keys() and len(source) == 9392
    assert all(row == source[i] for part in (train, holdout) for i, row in part.items())
    assert (HOLDOUT / 'test').resolve() == (HOLDOUT / 'val').resolve()  # root train.py scores the holdout as test too


def test_every_class_is_held_out_at_about_10_percent(split):
    _, holdout, _, source = split
    per_class, held = Counter(r['label'] for r in source.values()), Counter(r['label'] for r in holdout.values())
    assert set(held) == set(per_class) and all(0.095 <= held[c] / per_class[c] <= 0.105 for c in per_class)


def test_boxes_of_each_side_equal_the_source_boxes(split):
    source = json.loads((DATA / 'train' / 'instances.json').read_text())['annotations']
    for side in ('train', 'val'):
        ids = {int(r['image_id']) for r in rows(HOLDOUT / side / 'labels.csv')}
        boxes = json.loads((HOLDOUT / side / 'instances.json').read_text())['annotations']
        key = lambda a: (a['image_id'], a['id'])  # noqa: E731
        assert sorted(boxes, key=key) == sorted((a for a in source if a['image_id'] in ids), key=key)


def test_the_halves_partition_the_holdout(split):
    halves = json.loads((HOLDOUT / 'halves.json').read_text())
    a, b = set(halves['A']), set(halves['B'])
    assert not (a & b) and a | b == split[1].keys()


def test_flagged_overlap_partners_are_not_in_local_train(split):
    train, holdout, _, _ = split
    assert not set(H3_PARTNERS) & train.keys()
    for x, y in H2_PAIRS:  # each pair on one side: never one in the holdout and the other in local train
        assert not ((x in holdout and y in train) or (y in holdout and x in train))
