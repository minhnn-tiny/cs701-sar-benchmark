"""Training data of DetailLoRA: sarbench's SARMultiTask (resize to 512 x 512, random H/V flips) plus two SAR-aware
augmentations (docs/DATA.md §7). Evaluation uses SARMultiTask unchanged, so the input geometry is the reference's."""
import math
import sys
from pathlib import Path

import torch
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo root, for sarbench/
from sarbench.data import MEAN, STD, SARMultiTask  # noqa: E402


def transpose(image, boxes):
    """Swap the image axes: [C, H, W] -> [C, W, H]; XYXY boxes (x1, y1, x2, y2) -> (y1, x1, y2, x2). With the H/V
    flips this gives all 8 right-angle symmetries of the square input."""
    return image.transpose(-1, -2), boxes[:, [1, 0, 3, 2]]


def gamma(image, g):
    """Gamma on intensities in [0, 1]: monotone, keeps 0 and 1 (no-data borders and saturated scatterers)."""
    return image.clamp(0, 1) ** g


class AugmentedSARMultiTask(SARMultiTask):
    """SARMultiTask(train=True) + random transpose (p_transpose) and random gamma (p_gamma, log-uniform in
    gamma_range) on the [0, 1] image, then the reference normalisation. augment=False: the reference pipeline."""

    def __init__(self, root, split, size=512, limit=None, augment=True, p_transpose=0.5, p_gamma=0.5,
                 gamma_range=(0.8, 1.25)):
        super().__init__(root, split, train=True, size=size, limit=limit)
        # The reference transform without its final Normalize: the augmentations act on [0, 1] intensities
        self.transform = v2.Compose([t for t in self.transform.transforms if not isinstance(t, v2.Normalize)])
        self.augment, self.p_transpose, self.p_gamma = augment, p_transpose, p_gamma
        self.log_gamma = (math.log(gamma_range[0]), math.log(gamma_range[1]))

    def __getitem__(self, i):
        image, label, target = super().__getitem__(i)  # resized and flipped, intensities in [0, 1]
        image, boxes = image.as_subclass(torch.Tensor), target['boxes'].as_subclass(torch.Tensor)
        if self.augment:
            if torch.rand(()) < self.p_gamma:
                lo, hi = self.log_gamma
                image = gamma(image, math.exp(lo + (hi - lo) * torch.rand(()).item()))
            if torch.rand(()) < self.p_transpose:
                image, boxes = transpose(image, boxes)
        target['boxes'] = boxes
        return (image - MEAN) / STD, label, target
