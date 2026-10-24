"""Predict the unlabelled val or test images with a trained run (train.py's or root train.py's), write its leaderboard
zip with make_submission.write_submission, then check the zip with check_submission.py.
Example (from the repo root, through the GPU runner):
  workspace/gpu_run.sh --min-free 8000 -- uv run python workspace/sarmtl/predict.py <run_dir> --split val --jd on"""
import argparse
import importlib.util
import json
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # the repo root, for sarbench/ and make_submission.py
sys.path[:0] = [str(ROOT), str(HERE)]
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from check_submission import check, sample_format  # noqa: E402
from make_submission import write_submission  # noqa: E402
from sarmtl_model import JD_DECODING, REFERENCE_DECODING, model_from_args, set_decoding  # noqa: E402

# workspace/baseline/predict.py's Images: the images of an images.json, preprocessed as sarbench/data.py evaluates
_spec = importlib.util.spec_from_file_location('baseline_predict', ROOT / 'workspace' / 'baseline' / 'predict.py')
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)
Images = _baseline.Images


def deflate(path):
    """Rewrite the zip with its files compressed. write_submission stores them, and the leaderboard refuses zips over
    25 MB: the stored val zip is 16.4 MB, so a test zip (2,065 images) would come close."""
    with zipfile.ZipFile(path) as z:
        files = [(name, z.read(name)) for name in z.namelist()]
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in files:
            z.writestr(name, data)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path, help='a run directory (metrics.json, model.pt)')
    p.add_argument('--split', default='val', choices=['val', 'test'])
    p.add_argument('--jd', required=True, choices=['on', 'off', 'posterior'],
                   help='on: JD product (DESIGN.md §1); off: the reference decoding; posterior: JD posterior, the '
                        'alternative the 2-fold cross-fit adopts (crossfit.py)')
    p.add_argument('--data', type=Path, default=ROOT / 'dataset')
    p.add_argument('--out', type=Path, help='default: <run>/submission_<split>_jd<on|off>.zip')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    images_json = args.data / args.split / 'images.json'
    out = args.out or args.run / f'submission_{args.split}_jd{args.jd}.zip'

    # The run's model: its architecture from metrics.json, its weights from model.pt
    model = model_from_args(json.loads((args.run / 'metrics.json').read_text())['args'])
    model.load_state_dict(torch.load(args.run / 'model.pt', map_location='cpu'))
    model.cuda().eval()
    set_decoding(model, **{'on': JD_DECODING, 'off': REFERENCE_DECODING,
                           'posterior': dict(JD_DECODING, mode='posterior')}[args.jd])

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
    deflate(out)
    print(f'{len(images)} images, {1000 * seconds / len(images):.1f} ms per image (batch {args.batch_size}, data '
          f'loading included), JD {args.jd} -> {out}')

    sample = args.data / f'sample_submission_{args.split}.zip'
    header, keys = sample_format(sample if sample.exists() else args.data / 'sample_submission_val.zip')
    problems, summary = check(out, images, header, keys)
    for name, value in summary.items():
        print(f'  {name}: {value}')
    if problems:
        print(f'FAILED: {problems}')
        sys.exit(1)
    print('OK: the zip passes every check of check_submission.py')


if __name__ == '__main__':
    main()
