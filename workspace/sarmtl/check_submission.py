"""Check a leaderboard submission zip before it is uploaded: its size (the leaderboard refuses zips over 25 MB), the
files, columns and keys of the sample submission, and the content rules (every image once, labels 0-8, boxes
[x, y, w, h] in original pixels inside the image, scores in [0, 1]). Exit code 0 if the zip passes, 1 otherwise.
Example (from the repo root): uv run python workspace/sarmtl/check_submission.py <zip> --split val"""
import argparse
import csv
import io
import json
import math
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the repo root
FILES = ['classification.csv', 'detection.json']
DETECTION_KEYS = {'image_id', 'category_id', 'bbox', 'score'}  # what make_submission.write_submission writes
EDGE = 1e-3  # px: a box clipped to the 512 input and rescaled can overshoot the image edge by float error
MAX_BYTES = 25_000_000  # the leaderboard answers a larger zip with HTTP 413; 25 MB read as 10^6 bytes, the stricter


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def sample_format(sample_zip):
    """The classification.csv header of the sample submission, and its detection keys (the sample has no
    detections, so they default to DETECTION_KEYS)."""
    with zipfile.ZipFile(sample_zip) as z:
        header = next(csv.reader(io.StringIO(z.read('classification.csv').decode())))
        detections = json.loads(z.read('detection.json'))
    return header, set(detections[0]) if detections else DETECTION_KEYS


def check(zip_path, images, header=('image_id', 'label'), keys=DETECTION_KEYS, max_bytes=MAX_BYTES):
    """Check the zip against the images of images.json. Returns (problems, summary): problems maps a kind of problem to
    [count, first example]; it is empty if the zip is valid."""
    problems, summary = defaultdict(lambda: [0, None]), {}

    def problem(kind, example):
        problems[kind][0] += 1
        problems[kind][1] = problems[kind][1] or example

    size = Path(zip_path).stat().st_size
    if size > max_bytes:
        problem('the zip is larger than the leaderboard accepts', f'{size:,} bytes, limit {max_bytes:,}')
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if sorted(names) != FILES:
            problem('the zip must hold exactly classification.csv and detection.json', names)
        if not set(FILES) <= set(names):
            return dict(problems), summary
        csv_text, det_text = z.read('classification.csv').decode(), z.read('detection.json').decode()
    by_id = {im['id']: im for im in images}

    # classification.csv: header, then one row per image with an integer id of images.json and a label 0..8
    rows = list(csv.reader(io.StringIO(csv_text)))
    if not rows or rows[0] != list(header):
        problem(f'classification.csv header must be {",".join(header)}', rows[0] if rows else 'empty file')
    seen, labels = Counter(), {}
    for line, row in enumerate(rows[1:], 2):
        try:
            image_id, label = (int(v) for v in row)
        except ValueError:
            problem('classification.csv row is not two integers', f'line {line}: {row}')
            continue
        seen[image_id] += 1
        labels[image_id] = label
        if image_id not in by_id:
            problem('classification.csv has an id that is not in images.json', image_id)
        if not 0 <= label <= 8:
            problem('classification.csv label outside 0-8', f'id {image_id}: {label}')
    for image_id in sorted(by_id.keys() - seen.keys()):
        problem('an image of images.json has no classification row', image_id)
    for image_id in sorted(i for i, n in seen.items() if n > 1):
        problem('an image has more than one classification row', image_id)

    # detection.json: a list of {image_id, category_id 1..9, bbox [x, y, w, h] inside the image, score in [0, 1]}
    try:
        detections = json.loads(det_text)
    except json.JSONDecodeError as e:
        problem('detection.json is not valid JSON', str(e))
        detections = []
    if not isinstance(detections, list):
        problem('detection.json must be a list', type(detections).__name__)
        detections = []
    per_image, same_class, scores = Counter(), 0, []
    for k, d in enumerate(detections):
        if not isinstance(d, dict) or set(d) != keys:
            problem(f'a detection must have exactly the keys {sorted(keys)}', f'detection {k}: {d}')
            continue
        image_id, category, bbox, score = d['image_id'], d['category_id'], d['bbox'], d['score']
        if not isinstance(image_id, int) or image_id not in by_id:
            problem('a detection has an image_id that is not in images.json', f'detection {k}: {image_id}')
            continue
        per_image[image_id] += 1
        if not isinstance(category, int) or not 1 <= category <= 9:
            problem('a detection has a category_id outside 1-9', f'detection {k}: {category}')
        elif labels.get(image_id) is not None and category == labels[image_id] + 1:
            same_class += 1
        if not is_number(score) or not 0 <= score <= 1:
            problem('a detection score is not a finite number in [0, 1]', f'detection {k}: {score}')
        else:
            scores.append(score)
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(is_number(v) for v in bbox)):
            problem('a bbox is not 4 finite numbers', f'detection {k}: {bbox}')
            continue
        x, y, w, h = bbox
        im = by_id[image_id]
        if not (w > 0 and h > 0):
            problem('a bbox has w or h <= 0', f'detection {k}: {bbox}')
        if x < -EDGE or y < -EDGE or x + w > im['width'] + EDGE or y + h > im['height'] + EDGE:
            problem('a bbox [x, y, w, h] is not inside its image (original pixels)',
                    f'detection {k}: {bbox} in {im["width"]} x {im["height"]}')

    counts = [per_image[i] for i in by_id]
    summary = {
        'zip size': f'{size / 1e6:.1f} MB (limit {max_bytes / 1e6:.0f} MB)',
        'images': len(by_id),
        'classification rows': len(rows) - 1,
        'labels predicted per class': [sum(1 for v in labels.values() if v == c) for c in range(9)],
        'detections': len(detections),
        'detections per image (min / median / max)': (min(counts), statistics.median(counts), max(counts))
        if counts else None,
        'images without detections': sum(1 for n in counts if n == 0),
        'images with more than 100 detections (COCO scores the top 100)': sum(1 for n in counts if n > 100),
        'detections of the image\'s predicted class': f'{same_class / len(detections):.1%}' if detections else None,
        'score range': (min(scores), max(scores)) if scores else None,
    }
    return dict(problems), summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('zip', type=Path)
    p.add_argument('--split', default='val', choices=['val', 'test'])
    p.add_argument('--data', type=Path, default=ROOT / 'dataset')
    p.add_argument('--images', type=Path, help='images.json to check against (default: <data>/<split>/images.json)')
    p.add_argument('--sample', type=Path, help='sample submission for the columns and keys (default: '
                   '<data>/sample_submission_<split>.zip, or the val one if that does not exist yet)')
    args = p.parse_args()
    images = json.loads((args.images or args.data / args.split / 'images.json').read_text())['images']
    sample = args.sample or args.data / f'sample_submission_{args.split}.zip'
    if not sample.exists():
        sample = args.data / 'sample_submission_val.zip'
    header, keys = sample_format(sample)

    problems, summary = check(args.zip, images, header, keys)
    print(f'{args.zip} ({args.split}; columns and keys from {sample.name})')
    for name, value in summary.items():
        print(f'  {name}: {value}')
    if problems:
        print(f'FAILED: {sum(n for n, _ in problems.values())} problems')
        for kind, (n, example) in problems.items():
            print(f'  {n} x {kind}; first: {example}')
        sys.exit(1)
    print('OK: the zip passes every check')


if __name__ == '__main__':
    main()
