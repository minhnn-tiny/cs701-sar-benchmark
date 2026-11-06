"""Qualitative figure for the report: one holdout (ND) image per class, drawn at random (seed 0, not hand-picked), shown
for six classes (four that work, the two that fail),
with its ground-truth boxes (green) and the predictions of a holdout-trained model (orange, score >= 0.25) under the
submitted decoding. Run from the repo root through workspace/gpu_run.sh:
  workspace/gpu_run.sh --min-free 8000 -- uv run python reports/scripts/qualitative.py [OUT ROWS WIDTH HEIGHT FONT]
(the talk's version: reports/slides/assets/qualitative.png 2 9 6.4 14)"""
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, 'workspace/mtdet')
from evaluate import load_run  # noqa: E402
from predict import Images  # noqa: E402

RUN = Path('workspace/mtdet/runs/YSP350_s0')  # SAR 0.8, 350 epochs, trained on local train only: ND is unseen
DECODING = 'o2m_nms0.8_a0.5_k100_t0_f1'
ND = Path('dataset/holdout_fitdet/val')
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('reports/report-overleaf/figs/qualitative.pdf')
ROWS, W, H, FONT = (int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])) \
    if len(sys.argv) > 5 else (1, 7.0, 1.3, 5.5)
NAMES = ['aircraft', 'airport', 'bridge', 'car', 'harbor', 'oil tank', 'playground', 'ship', 'wind turbine']
SHOW = 0.25

rows = list(csv.DictReader(open(ND / 'labels.csv')))
coco = json.load(open(ND / 'instances.json'))
sizes = {im['id']: (im['width'], im['height']) for im in coco['images']}
gt = {}
for a in coco['annotations']:
    gt.setdefault(a['image_id'], []).append(a['bbox'])
rng = random.Random(0)
picked = [rng.choice([r for r in rows if int(r['label']) == c]) for c in range(9)]  # same draws as the 9-class version
picked = [picked[c] for c in (0, 3, 5, 8, 2, 6)]  # aircraft, car, oil tank, wind turbine | bridge, playground

model, _ = load_run(RUN)
model.apply_decoding(model.decodings()[DECODING])
images = [{'id': int(r['image_id']), 'file_name': r['file_name']} for r in picked]
batch = torch.stack([Images(images, ND)[i] for i in range(len(images))]).cuda()
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    logits, dets, _ = model(batch)
probs = logits.float().softmax(1).cpu()

fig, axes = plt.subplots(ROWS, 6 // ROWS, figsize=(W, H), dpi=300 if ROWS == 1 else 200)
for ax, r, p, d in zip(axes.flat, picked, probs, dets):
    iid, label = int(r['image_id']), int(r['label'])
    w, h = sizes[iid]
    ax.imshow(Image.open(ND / r['file_name']).convert('L'), cmap='gray', vmin=0, vmax=255)
    for x, y, bw, bh in gt.get(iid, []):
        ax.add_patch(plt.Rectangle((x, y), bw, bh, fill=False, lw=0.6 if ROWS == 1 else 1.4, ec='#2ecc40'))
    sx, sy = w / 512, h / 512  # predictions are XYXY on the 512 input
    for (x1, y1, x2, y2), s in zip(d['boxes'].float().cpu().tolist(), d['scores'].float().cpu().tolist()):
        if s >= SHOW:
            ax.add_patch(plt.Rectangle((x1 * sx, y1 * sy), (x2 - x1) * sx, (y2 - y1) * sy, fill=False,
                                       lw=0.6 if ROWS == 1 else 1.4,
                                       ec='#ff8c1a', ls='--'))
    pred = int(p.argmax())
    ok = '' if pred == label else f' → {NAMES[pred]}'
    ax.set_title(f'{NAMES[label]}{ok} ({p[pred]:.2f})', fontsize=FONT, pad=1.5 if ROWS == 1 else 5,
                 color='black' if pred == label else '#c0392b')
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.3)
plt.subplots_adjust(left=0.003, right=0.997, top=0.88 if ROWS == 1 else 0.94, bottom=0.01, wspace=0.03,
                    hspace=0.12)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print('wrote', OUT, [(NAMES[int(r['label'])], r['file_name'], len(gt.get(int(r['image_id']), []))) for r in picked])
