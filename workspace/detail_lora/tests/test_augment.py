"""The SAR augmentations keep boxes on their objects and intensities in range."""
from pathlib import Path

import pytest
import torch

from detail_data import AugmentedSARMultiTask, gamma, transpose
from sarbench.data import MEAN, STD, SARMultiTask

HOLDOUT = Path(__file__).resolve().parents[3] / 'dataset' / 'holdout'


def test_transpose_moves_boxes_with_pixels():
    image = torch.zeros(1, 512, 512)
    image[0, 100:120, 300:340] = 1  # rows y 100-119, columns x 300-339
    box = torch.tensor([[300., 100., 340., 120.]])
    image_t, box_t = transpose(image, box)
    x1, y1, x2, y2 = box_t[0].int().tolist()
    assert image_t[0, y1:y2, x1:x2].min() == 1 and image_t.sum() == image.sum() == (y2 - y1) * (x2 - x1)


def test_gamma_is_monotone_and_keeps_the_range():
    x = torch.linspace(0, 1, 101)
    for g in (0.8, 1.0, 1.25):
        y = gamma(x, g)
        assert y[0] == 0 and y[-1] == 1 and (y.diff() >= 0).all()


@pytest.mark.skipif(not HOLDOUT.exists(), reason='run make_holdout.py first')
def test_augmented_items_match_the_reference_format():
    torch.manual_seed(0)
    plain = SARMultiTask(HOLDOUT, 'train', train=True, limit=8)
    augmented = AugmentedSARMultiTask(HOLDOUT, 'train', limit=8, p_transpose=1.0, p_gamma=1.0)
    off = AugmentedSARMultiTask(HOLDOUT, 'train', limit=8, augment=False)
    for i in range(8):
        image, label, target = augmented[i]
        ref_image, ref_label, ref_target = plain[i]
        assert image.shape == ref_image.shape == (1, 512, 512) and label == ref_label
        assert target['boxes'].shape == ref_target['boxes'].shape and (target['labels'] == ref_target['labels']).all()
        boxes = target['boxes']
        assert (boxes[:, 2:] > boxes[:, :2]).all() and boxes.min() >= 0 and boxes.max() <= 512
        lo, hi = (0 - MEAN) / STD, (1 - MEAN) / STD
        assert image.min() >= lo - 1e-4 and image.max() <= hi + 1e-4
        # augment=False is the reference pipeline: same values up to the random flips
        off_image = off[i][0]
        assert torch.allclose(off_image.flatten().sort().values, ref_image.flatten().sort().values, atol=1e-5)
