"""train.py's 90-degree rotation augmentation keeps every box on its object."""
import importlib.util
from pathlib import Path

import torch

# workspace/r50fpn/train.py by path: the repo root, first on sys.path, has its own train.py
_spec = importlib.util.spec_from_file_location('r50fpn_train', Path(__file__).resolve().parents[1] / 'train.py')
r50fpn_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r50fpn_train)


def test_rot90_moves_each_box_with_its_pixels():
    image = torch.zeros(1, 8, 8)
    image[0, 1:3, 5:7] = 1  # rows 1-2 and columns 5-6, so the box is x 5..7, y 1..3
    box = torch.tensor([[5., 1., 7., 3.]])
    for k in range(4):
        rotated, b = r50fpn_train.rot90(image, box, k)
        rows, cols = rotated[0].nonzero(as_tuple=True)
        assert b.tolist() == [[cols.min().item(), rows.min().item(), cols.max().item() + 1, rows.max().item() + 1]]
