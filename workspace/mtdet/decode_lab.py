"""Decoding variants that need no retraining, scored on a run's holdout (ND) from ONE forward pass of the one-to-many
head (YOLO26 only). Each variant is still one model, one pass per image:
- fuse_b<B>: image class = argmax(log softmax(logits) + B * log(max detector score of the class over the image));
  _jd<A>: box scores also x fused probs ** A (joint decoding); jd<A>: joint decoding on the classifier's probs alone.
31 Oct, 9 holdout runs (YSP350, YSA450, YSP450B96 x 3 seeds), against the submitted decoding: fuse_b2_jd0.5 +0.57 +- 0.29
(9/9 wins), fuse_b2 alone +0.11 (6/9), B 0.5 / 1 / 3 / 5 / 10 +0.08 / +0.15 / +0.13 / +0.05 / -0.16, the detector alone
-0.52; JD A 0.25 = 0.5. Dropped on YSP350_s0: per-class NMS IoU chosen on the run itself +0.10, boxes relabelled to the
predicted image class -0.16 / -0.31.
  decode_lab.py RUN [RUN ...]   -> prints one line per variant and writes <run>/decode_lab.json
Example: workspace/gpu_run.sh --min-free 8000 -- uv run python workspace/mtdet/decode_lab.py workspace/mtdet/runs/YSP350_s0"""
import argparse
import json
from pathlib import Path

import torch

from common import ROOT, delta_m, dense_decode, make_loader, to_unit_rgb
from evaluate import load_run
from sarbench.data import to_original_xywh
from sarbench.metrics import classification_metrics, detection_metrics


@torch.no_grad()
def forward_all(model, loader):
    """Per image: boxes [N, 4] (fp32), per-class scores [N, 9], logits [9]; plus labels and COCO targets."""
    model.apply_decoding(dict(head='o2m'))
    boxes, scores, logits, labels, targets = [], [], [], [], []
    for images, y, t in loader:
        x = to_unit_rgb(images.cuda(non_blocking=True))
        if model.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            raw = model.det(x)[1]  # as UltraMultiTask.forward, keeping the dense scores before any decoding
            lg = model.cls_head(model.features())
        b, s = model._boxes_scores(raw, x.shape[-1])
        boxes.append(b), scores.append(s), logits.append(lg.float())
        labels += y.tolist()
        targets += t
    return torch.cat(boxes), torch.cat(scores), torch.cat(logits), labels, targets


def coco(dets, targets):
    out = []
    for d, t in zip(dets, targets):
        xywh = to_original_xywh(d['boxes'].float().cpu(), t['orig_size']).tolist()
        out += [{'image_id': t['image_id'], 'category_id': int(c), 'bbox': b, 'score': float(s)}
                for b, c, s in zip(xywh, d['labels'].tolist(), d['scores'].tolist())]
    return out


def decode(boxes, scores, probs, jd_alpha=0.0, chunk=64):
    """The submitted decoding (one-to-many head, NMS 0.8, top 100, no floor), JD optional, in chunks of images."""
    return [d for i in range(0, len(boxes), chunk) for d in dense_decode(
        boxes[i:i + chunk], scores[i:i + chunk], probs[i:i + chunk], jd_alpha, 0.8, 100, 0.0)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', type=Path, nargs='+')
    p.add_argument('--betas', type=float, nargs='+', default=[1.0, 2.0])
    p.add_argument('--alphas', type=float, nargs='+', default=[0.5])
    args = p.parse_args()
    for run in args.runs:
        model, run_args = load_run(run)
        data = ROOT / run_args['data']
        gt = data / 'val' / 'instances.json'
        boxes, scores, logits, labels, targets = forward_all(model, make_loader(data, 'val', batch_size=16))
        ids = [t['image_id'] for t in targets]
        probs = logits.softmax(1)
        pred = logits.argmax(1)
        results = {}

        def score(name, cls_pred, dets):
            m = {**classification_metrics(labels, cls_pred.tolist()), **detection_metrics(gt, dets, ids)}
            m['delta_m'] = delta_m(m['macro_f1'], m['mAP'])
            results[name] = {k: v for k, v in m.items() if k != 'confusion_matrix'}
            print(f"{run.name:10s} {name:14s} F1 {100 * m['macro_f1']:.2f} mAP {100 * m['mAP']:.2f} "
                  f"AP50 {100 * m['AP50']:.2f} delta_m {m['delta_m']:+.2f}", flush=True)
            return m

        base = coco(decode(boxes, scores, probs), targets)
        score('base', pred, base)  # the submitted o2m_nms0.8_a0_k100_t0
        peak = scores.amax(1).clamp_min(1e-6).log()  # [images, 9]: the detector's strongest box per class
        for a in args.alphas:
            score(f'jd{a:g}', pred, coco(decode(boxes, scores, probs, a), targets))
        for b in args.betas:
            fused = logits.log_softmax(1) + b * peak
            score(f'fuse_b{b:g}', fused.argmax(1), base)
            for a in args.alphas:
                score(f'fuse_b{b:g}_jd{a:g}', fused.argmax(1), coco(decode(boxes, scores, fused.softmax(1), a), targets))
        out = run / 'decode_lab.json'  # merged into earlier sweeps of the run
        out.write_text(json.dumps({**(json.loads(out.read_text()) if out.exists() else {}), **results}, indent=1))


if __name__ == '__main__':
    main()
