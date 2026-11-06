"""Contact sheets of TRAIN images with their boxes (red), 12 random 512 x 512 images per class (random.Random(1)),
for eyeballing what is and is not annotated. Writes dataset/holdout_sarmtl/analysis/contact_<class>.png.
Usage (repo root): uv run python workspace/sarmtl/data/contact_sheets.py [--classes airport harbor ...]
"""
import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'dataset' / 'train'
OUT = ROOT / 'dataset' / 'holdout_sarmtl' / 'analysis'
NAMES = [c['name'] for c in json.loads((ROOT / 'dataset' / 'classes.json').read_text())]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--classes', nargs='+', default=['airport', 'harbor', 'bridge', 'playground', 'oil_tank', 'aircraft'])
    p.add_argument('--tile', type=int, default=320)
    args = p.parse_args()
    coco = json.loads((DATA / 'instances.json').read_text())
    info = {i['id']: i for i in coco['images']}
    boxes = {}
    for a in coco['annotations']:
        boxes.setdefault(a['image_id'], []).append(a['bbox'])
    with open(DATA / 'labels.csv') as f:
        rows = list(csv.DictReader(f))
    rng, T = random.Random(1), args.tile
    for c in args.classes:
        ids = [int(r['image_id']) for r in rows if NAMES[int(r['label'])] == c and info[int(r['image_id'])]['width'] == 512]
        pick = rng.sample(ids, 12)
        sheet = Image.new('RGB', (4 * T, 3 * T), 'white')
        for n, i in enumerate(pick):
            im = Image.open(DATA / info[i]['file_name']).convert('L')
            s = T / im.size[0]
            im = im.convert('RGB').resize((T, T))
            d = ImageDraw.Draw(im)
            for x, y, w, h in boxes[i]:
                d.rectangle([x * s, y * s, (x + w) * s, (y + h) * s], outline=(255, 0, 0), width=1)
            d.text((3, 3), str(i), fill=(0, 255, 0))
            sheet.paste(im, ((n % 4) * T, (n // 4) * T))
        sheet.save(OUT / f'contact_{c}.png')
        print(c, pick)


if __name__ == '__main__':
    main()
