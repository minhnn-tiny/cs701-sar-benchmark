"""Predict the unlabelled val or test images with a trained run and write its Codabench submission zip, then check
the zip's format. Works on DetailLoRA runs and on reference runs of the top-level train.py.
Example (repo root): CUDA_VISIBLE_DEVICES=1 uv run python workspace/detail_lora/predict.py <run_dir> --split val"""
import argparse
import csv
import io
import json
import time
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import FUSIONS, ROOT, load_run, reference_predict
from make_submission import write_submission


def check_submission(path, images, sample=None):
    """Raise if the zip at `path` is not a valid submission for `images` (from images.json): exactly
    classification.csv (image_id,label; one row per image; labels 0-8) and detection.json (a list of
    {image_id, category_id 1-9, bbox [x, y, w, h] with w, h >= 0, score in [0, 1]} on known ids).
    `sample`: the organisers' sample zip, whose file names and CSV header must match."""
    ids = {im['id'] for im in images}
    with zipfile.ZipFile(path) as z:
        names = sorted(z.namelist())
        assert names == ['classification.csv', 'detection.json'], names
        rows = list(csv.DictReader(io.StringIO(z.read('classification.csv').decode())))
        header = z.read('classification.csv').decode().splitlines()[0]
        detections = json.loads(z.read('detection.json'))
    assert header == 'image_id,label', header
    assert len(rows) == len(ids) and {int(r['image_id']) for r in rows} == ids, 'one row per image, every image'
    assert all(0 <= int(r['label']) <= 8 for r in rows)
    assert isinstance(detections, list)
    for d in detections:
        assert set(d) == {'image_id', 'category_id', 'bbox', 'score'}, d
        assert d['image_id'] in ids and 1 <= d['category_id'] <= 9 and 0 <= d['score'] <= 1, d
        assert len(d['bbox']) == 4 and d['bbox'][2] >= 0 and d['bbox'][3] >= 0, d
    if sample is not None:
        with zipfile.ZipFile(sample) as z:
            assert sorted(z.namelist()) == names
            assert z.read('classification.csv').decode().splitlines()[0] == header
    return {'images': len(rows), 'detections': len(detections),
            'images_with_detections': len({d['image_id'] for d in detections})}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path, help='a run directory (metrics.json, model.pt)')
    p.add_argument('--split', default='val', choices=['val', 'test'])
    p.add_argument('--fusion', choices=FUSIONS, help="detection scoring (default: the run's own; reference runs: none)")
    p.add_argument('--data', type=Path, default=ROOT / 'dataset')
    p.add_argument('--out', type=Path, help='default: <run>/submission_<split>[_<fusion>].zip')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    images_json = args.data / args.split / 'images.json'
    out = args.out or args.run / f'submission_{args.split}{"_" + args.fusion if args.fusion else ""}.zip'

    model, _ = load_run(args.run, args.fusion)
    images = json.loads(images_json.read_text())['images']
    loader = DataLoader(reference_predict.Images(images, images_json.parent), args.batch_size,
                        num_workers=args.workers)
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
    sample = args.data / f'sample_submission_{args.split}.zip'
    summary = check_submission(out, images, sample if sample.exists() else None)
    print(f'{len(images)} images, {1000 * seconds / len(images):.1f} ms per image (data loading included), '
          f'scoring {model.roi_heads.fusion} -> {out}\nformat OK: {summary}')


if __name__ == '__main__':
    main()
