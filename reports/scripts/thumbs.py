"""Two thumbnails for the pipeline figure: a holdout harbor image as input, and the same image with the predictions
of reports/scripts/qualitative.py's model (orange) - drawn on the CPU from cached boxes is not possible, so this reruns
the model; run through workspace/gpu_run.sh like qualitative.py."""
import sys
from pathlib import Path
import torch
from PIL import Image, ImageDraw
sys.path.insert(0, 'workspace/mtdet')
from evaluate import load_run  # noqa: E402
from predict import Images  # noqa: E402

ND, F = Path('dataset/holdout_fitdet/val'), Path('reports/report-overleaf/figs')
name = 'images/01078.jpg'  # the car image of the qualitative figure (seed-0 draw)
img = Image.open(ND / name).convert('L')
img.resize((256, 256)).save(F / 'thumb_input.png')
model, _ = load_run(Path('workspace/mtdet/runs/YSP350_s0'))
model.apply_decoding(model.decodings()['o2m_nms0.8_a0.5_k100_t0_f1'])
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    _, dets, _ = model(Images([{'file_name': name}], ND)[0][None].cuda())
out = img.convert('RGB').resize((512, 512))
draw = ImageDraw.Draw(out)
for (x1, y1, x2, y2), s in zip(dets[0]['boxes'].float().tolist(), dets[0]['scores'].float().tolist()):
    if s >= 0.25:
        draw.rectangle((x1, y1, x2, y2), outline=(255, 140, 26), width=4)
out.resize((256, 256)).save(F / 'thumb_output.png')
print('wrote thumbnails')
