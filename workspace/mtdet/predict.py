"""Predict the unlabelled val or test images with a finished run and write its leaderboard zip: make_submission's
write_submission, then compact() (boxes rounded to 0.01 px and clamped inside the image, scores to 6 digits,
compressed: 300 boxes per image stay far under 25 MB), then workspace/sarmtl/check_submission.py's checks.
Example: workspace/gpu_run.sh --min-free 8000 -- uv run python workspace/mtdet/predict.py workspace/mtdet/runs/FY_s0 \
             --split val --decoding o2o_a1_k300 --out workspace/mtdet/submissions/<name>.zip"""
import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

from common import MEAN, ROOT, STD
from check_submission import check, sample_format  # workspace/sarmtl's (on sys.path through common)
from evaluate import load_run
from make_submission import write_submission


class Images(torch.utils.data.Dataset):
    """The images of an images.json, preprocessed exactly as sarbench/data.py evaluates (gray, 512 x 512, SAR mean/std)."""

    def __init__(self, images, root):
        self.images, self.root = images, root
        self.transform = v2.Compose([v2.ToImage(), v2.Resize((512, 512)), v2.ToDtype(torch.float32, scale=True),
                                     v2.Normalize([MEAN], [STD])])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(Image.open(self.root / self.images[i]['file_name']).convert('L'))


def compact(path, images, digits=2, score_digits=6):
    """r50fpn's: rewrite the zip compressed, detection.json compact, box corners rounded (right and bottom edges
    clamped to the image), scores rounded; boxes that round to zero width or height are dropped."""
    size = {im['id']: (im['width'], im['height']) for im in images}
    with zipfile.ZipFile(path) as z:
        files = {name: z.read(name) for name in z.namelist()}
    kept = []
    for d in json.loads(files['detection.json']):
        width, height = size[d['image_id']]
        x, y, w, h = d['bbox']
        x1, y1 = round(x, digits), round(y, digits)
        x2, y2 = min(round(x + w, digits), width), min(round(y + h, digits), height)
        if x2 > x1 and y2 > y1:
            kept.append({**d, 'bbox': [x1, y1, round(x2 - x1, digits), round(y2 - y1, digits)],
                         'score': round(d['score'], score_digits)})
    files['detection.json'] = json.dumps(kept, separators=(',', ':')).encode()
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path, help='a finished run directory (metrics.json, model.pt)')
    p.add_argument('--split', default='val', choices=['val', 'test'])
    p.add_argument('--decoding', required=True, help="a name of the architecture's decoding grid (evaluate.py)")
    p.add_argument('--data', type=Path, default=ROOT / 'dataset')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    model, _ = load_run(args.run)
    model.apply_decoding(model.decodings()[args.decoding])
    images_json = args.data / args.split / 'images.json'
    images = json.loads(images_json.read_text())['images']
    loader = torch.utils.data.DataLoader(Images(images, images_json.parent), args.batch_size, num_workers=args.workers)
    labels, outputs, start = [], [], time.time()
    with torch.no_grad():
        for batch in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, detections, _ = model(batch.cuda())
            labels += logits.argmax(dim=1).tolist()
            outputs += [{k: v.float().cpu() for k, v in d.items()} for d in detections]
    seconds = time.time() - start
    # write_submission takes boxes XYXY on the model input and classes 0..8 (the detectors' labels are 1..9)
    preds = {im['file_name']: (label, d['boxes'].tolist(), (d['labels'] - 1).long().tolist(), d['scores'].tolist())
             for im, label, d in zip(images, labels, outputs)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(images, preds, out=args.out)
    compact(args.out, images)
    print(f'{len(images)} images, {1000 * seconds / len(images):.1f} ms per image (batch {args.batch_size}, data '
          f'loading included), decoding {args.decoding} -> {args.out}')
    sample = args.data / f'sample_submission_{args.split}.zip'
    header, keys = sample_format(sample if sample.exists() else args.data / 'sample_submission_val.zip')
    problems, summary = check(args.out, images, header, keys)
    for name, value in summary.items():
        print(f'  {name}: {value}')
    if problems:
        print(f'FAILED: {dict(problems)}')
        sys.exit(1)
    print('OK: the zip passes every check of check_submission.py')


if __name__ == '__main__':
    main()
