"""PNG assets for the talk (reports/slides/assets/), on the CPU: the report's two charts at projector size, a SAR
augmentation example (the speckle and gamma of workspace/mtdet/common.SARAugment at fixed draws), and a cover grid of
holdout scenes, one per class (seed-0 draws, as in qualitative.py). The pipeline, the detection examples and the
thumbnails come from report-overleaf/figs and reports/scripts/{qualitative,thumbs}.py.
  uv run python reports/scripts/slide_assets.py"""
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import charts  # noqa: E402  (reports/scripts on sys.path when run as a script)

OUT = Path('reports/slides/assets')
ND = Path('dataset/holdout_fitdet/val')
OUT.mkdir(parents=True, exist_ok=True)

# charts at projector size: ~2x the print fonts on a 6.4 x 3.6 in canvas, 200 dpi
plt.rcParams.update({'font.size': 13, 'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
                     'legend.fontsize': 12, 'savefig.dpi': 200})
charts.per_class(OUT / 'per_class.png', size=(7.2, 4.4))
charts.epochs('fuse_b1_jd0.5', OUT / 'epochs.png', size=(7.2, 4.0))
for f in ('per_class.png', 'epochs.png'):
    im = Image.open(OUT / f)
    print(f, im.size)

# SAR augmentation example: original, + speckle (L = 4), + gamma 1.5, on one holdout image
rows = list(csv.DictReader(open(ND / 'labels.csv')))
rng = random.Random(0)
picked = [rng.choice([r for r in rows if int(r['label']) == c]) for c in range(9)]
x = np.asarray(Image.open(ND / picked[3]['file_name']).convert('L').resize((512, 512)), dtype=np.float32) / 255
g = np.random.default_rng(0).gamma(4.0, 1 / 4.0, x.shape).astype(np.float32)
panels = [x, np.clip(x * np.sqrt(g), 0, 1), np.clip(x, 0, 1) ** 1.5]
Image.fromarray(np.concatenate([np.pad((p * 255).astype(np.uint8), ((0, 0), (0, 12)), constant_values=251)
                                for p in panels], axis=1)[:, :-12]).save(OUT / 'aug_example.png')

# cover: a 3 x 3 grid of the seed-0 holdout scenes, one per class
tiles = [Image.open(ND / r['file_name']).convert('L').resize((300, 300)) for r in picked]
grid = Image.new('L', (3 * 300 + 2 * 10, 3 * 300 + 2 * 10), 20)
for i, t in enumerate(tiles):
    grid.paste(t, ((i % 3) * 310, (i // 3) * 310))
grid.save(OUT / 'cover_grid.png')
print('wrote', sorted(p.name for p in OUT.iterdir()))
