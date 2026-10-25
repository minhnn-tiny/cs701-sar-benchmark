"""Predict the unlabelled val or test images with a trained R50-FPN-MT run and write its leaderboard zip: the boxes and
labels go through make_submission.write_submission, then compact() rounds the detections and compresses the zip, so
300 boxes per image stay under the leaderboard's 25 MB; then workspace/sarmtl/check_submission.py checks it.
Example (from the repo root, through the GPU runner):
  workspace/gpu_run.sh --min-free 8000 -- uv run python workspace/r50fpn/predict.py <run_dir> --split val \
      --decoding reference --out workspace/r50fpn/submissions/<name>.zip"""
import argparse
import importlib.util
import json
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # the repo root, for sarbench/ and make_submission.py
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / 'workspace' / 'sarmtl')]
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from check_submission import check, sample_format  # noqa: E402  workspace/sarmtl's
from make_submission import write_submission  # noqa: E402
from r50fpn_model import DECODINGS, model_from_args, set_decoding  # noqa: E402

# workspace/sarmtl/predict.py under another name (this file is also predict.py), for its Images dataset: the images of
# an images.json, preprocessed as sarbench/data.py evaluates. It puts workspace/sarmtl first on sys.path, which would
# shadow this design's modules, so the path is restored.
_spec = importlib.util.spec_from_file_location('sarmtl_predict', ROOT / 'workspace' / 'sarmtl' / 'predict.py')
_sarmtl, _path = importlib.util.module_from_spec(_spec), list(sys.path)
_spec.loader.exec_module(_sarmtl)
sys.path[:] = _path
Images = _sarmtl.Images


def compact(path, images, digits=2, score_digits=6):
    """Rewrite the zip compressed, with detection.json in compact JSON: box corners rounded to 10^-digits px (right and
    bottom edges clamped to the image, so a box never leaves it), scores to score_digits digits; boxes that round to
    zero width or height are dropped."""
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
    p.add_argument('run', type=Path, help='a run directory (metrics.json, model.pt)')
    p.add_argument('--split', default='val', choices=['val', 'test'])
    p.add_argument('--decoding', required=True, choices=list(DECODINGS), help="evaluate.py's names")
    p.add_argument('--data', type=Path, default=ROOT / 'dataset')
    p.add_argument('--out', type=Path, help='default: <run>/submission_<split>_<decoding>.zip')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    images_json = args.data / args.split / 'images.json'
    out = args.out or args.run / f'submission_{args.split}_{args.decoding}.zip'
    out.parent.mkdir(parents=True, exist_ok=True)

    model = model_from_args(json.loads((args.run / 'metrics.json').read_text())['args'])
    model.load_state_dict(torch.load(args.run / 'model.pt', map_location='cpu'))
    model.cuda().eval()
    set_decoding(model, **DECODINGS[args.decoding])

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
    compact(out, images)
    print(f'{len(images)} images, {1000 * seconds / len(images):.1f} ms per image (batch {args.batch_size}, data '
          f'loading included), decoding {args.decoding} {DECODINGS[args.decoding]} -> {out}')

    sample = args.data / f'sample_submission_{args.split}.zip'
    header, keys = sample_format(sample if sample.exists() else args.data / 'sample_submission_val.zip')
    problems, summary = check(out, images, header, keys)
    for name, value in summary.items():
        print(f'  {name}: {value}')
    if problems:
        print(f'FAILED: {dict(problems)}')
        sys.exit(1)
    print('OK: the zip passes every check of check_submission.py')


if __name__ == '__main__':
    main()
