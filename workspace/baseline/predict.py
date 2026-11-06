"""Predict the unlabelled val or test images with a trained run and write its Codabench submission zip.
Example (from the repo root): uv run python workspace/baseline/predict.py <run_dir> --split val"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[2]  # the repo root, for sarbench/ and make_submission.py
sys.path.insert(0, str(ROOT))
from make_submission import write_submission
from sarbench.adapters import add_adapters
from sarbench.backbones import build_backbone
from sarbench.data import MEAN, STD
from sarbench.model import MultiTaskModel


class Images(Dataset):
    """The images of a COCO-style images.json, preprocessed as SARMultiTask does at evaluation."""

    def __init__(self, images, root, size=512):
        self.images, self.root = images, root
        self.transform = v2.Compose([v2.ToImage(), v2.Resize((size, size)), v2.ToDtype(torch.float32, scale=True),
                                     v2.Normalize([MEAN], [STD])])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(Image.open(self.root / self.images[i]['file_name']).convert('L'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path, help='a run directory of train.py (metrics.json, model.pt)')
    p.add_argument('--split', default='val', choices=['val', 'test'])
    p.add_argument('--data', type=Path, default=ROOT / 'dataset')
    p.add_argument('--images', type=Path, help='images.json to predict (default: <data>/<split>/images.json)')
    p.add_argument('--out', type=Path, help='default: <run>/submission_<split>.zip')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    images_json = args.images or args.data / args.split / 'images.json'
    out = args.out or args.run / f'submission_{args.split}.zip'

    # The run's model: its architecture from metrics.json, its weights from model.pt
    run = json.loads((args.run / 'metrics.json').read_text())['args']
    backbone = build_backbone(run['backbone'], pretrained=False)
    if run['adapt'] != 'full':
        add_adapters(backbone, run['adapt'])
    model = MultiTaskModel(backbone, task_routing=run['adapt'] == 'moelora')
    model.load_state_dict(torch.load(args.run / 'model.pt', map_location='cpu'))
    model.cuda().eval()

    images = json.loads(images_json.read_text())['images']
    loader = DataLoader(Images(images, images_json.parent), args.batch_size, num_workers=args.workers)
    labels, outputs, start = [], [], time.time()
    with torch.no_grad():
        for batch in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, detections, _ = model(batch.cuda())
            labels += logits.argmax(dim=1).tolist()
            outputs += [{k: v.cpu() for k, v in d.items()} for d in detections]
    seconds = time.time() - start

    # write_submission takes boxes XYXY on the model input and classes 0..8 (the detector's labels are 1..9)
    preds = {im['file_name']: (label, d['boxes'].tolist(), (d['labels'] - 1).tolist(), d['scores'].tolist())
             for im, label, d in zip(images, labels, outputs)}
    write_submission(images, preds, out=out)
    print(f'{len(images)} images, {1000 * seconds / len(images):.1f} ms per image (data loading included) -> {out}')


if __name__ == '__main__':
    main()
