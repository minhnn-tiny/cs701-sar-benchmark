"""D1 protocol and analysis (cand_detect.md section 2): run a checkpoint once on the holdout, caching its raw RoI outputs
and image logits, then re-decode and score any decoding setting on the CPU.

  collect(model, dataset)          -> cache (one forward per image; on the GPU through gpu_run.sh, or on the CPU)
  decode(cache, heads, ...)        -> predicted image classes + COCO detections for one decoding setting
  report(cache, heads, gt_json, ...) prints, all paired on the same cache:
      the decision: reference decoding (alpha 0, thresh 0.05) vs (0, 0.001) vs D1 = (1, 0.001), split into the
        threshold's share and the prior's share of the gain, and the adopt-or-not rule;
      the downside (critic D1-b): per-class AP change; D1 on correctly classified images only vs on all images (the
        gap is the cost of misclassification); mean max-softmax on wrongly and correctly classified images;
      the analysis sweep (alpha, D1b lambda, NMS) and, given holdout halves, the 2-fold cross-fit.
Self-test on the CPU (random weights, a few holdout images, checks the plumbing only):
  HF_HUB_OFFLINE=1 uv run python workspace/sarmtl/scratch/cv-detect/d1_analysis.py --selftest
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_modules import ROOT, DetectModel  # noqa: E402
from sarbench.data import SARMultiTask, collate, to_original_xywh  # noqa: E402
from sarbench.metrics import classification_metrics, detection_metrics  # noqa: E402

INF = float('inf')
def setting(alpha=0.0, lam=0.0, score_thresh=0.001, nms_thresh=0.5, mode='product'):
    return dict(alpha=alpha, lam=lam, score_thresh=score_thresh, nms_thresh=nms_thresh, mode=mode)


REFERENCE = setting(score_thresh=0.05)
THRESH_ONLY = setting()
D1 = setting(alpha=1.0)  # fixed a priori: the product rule, alpha 1, threshold 0.001
SWEEP = [setting(alpha=a) for a in (0.0, 0.5, 1.0, 2.0, INF)] + \
        [setting(alpha=a, mode='posterior') for a in (0.5, 1.0, INF)] + \
        [setting(alpha=1.0, lam=l) for l in (1.0, 2.0, 4.0)] + [setting(alpha=1.0, nms_thresh=0.6)]


@torch.no_grad()
def collect(model, dataset, batch_size=16, device='cpu', workers=0):
    """One eval forward per image (reference decoding); returns per-image dicts with the raw RoI outputs."""
    model.eval().to(device)
    model.prior_alpha = model.evidence_lambda = 0.0
    loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=workers, collate_fn=collate)
    cache = []
    for images, labels, targets in loader:
        model.roi_heads.capture = []
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            logits, _, _ = model(images.to(device))
        raw, model.roi_heads.capture = model.roi_heads.capture, None
        for r, lg, y, t in zip(raw, logits.float().cpu(), labels.tolist(), targets):
            cache.append(dict(r, logits=lg, label=y, image_id=t['image_id'], orig_size=t['orig_size']))
    return cache


def decode(cache, heads, alpha=0.0, lam=0.0, score_thresh=0.05, nms_thresh=0.5, mode='product'):
    """Re-decode the cached RoI outputs with JointDecodeRoIHeads (the same code path as inference)."""
    old = heads.score_thresh, heads.nms_thresh
    heads.score_thresh, heads.nms_thresh = score_thresh, nms_thresh
    logp = torch.stack([c['logits'] for c in cache]).log_softmax(dim=1)
    heads.decode = dict(logp=logp, alpha=alpha, lam=lam, mode=mode) if alpha > 0 or lam > 0 else None
    try:
        boxes, scores, labels = heads.postprocess_detections(
            torch.cat([c['class_logits'] for c in cache]), torch.cat([c['box_regression'] for c in cache]),
            [c['proposals'] for c in cache], [(512, 512)] * len(cache))
        pred = (heads.image_logp if heads.decode is not None else logp).argmax(dim=1)
    finally:
        heads.decode = heads.image_logp = None
        heads.score_thresh, heads.nms_thresh = old
    dets = [[{'image_id': c['image_id'], 'category_id': int(l), 'bbox': b, 'score': float(s)}
             for b, l, s in zip(to_original_xywh(bx, c['orig_size']).tolist(), lb, sc)]
            for c, bx, sc, lb in zip(cache, boxes, scores, labels)]
    return pred.tolist(), dets  # dets: per image


def score(cache, pred, dets, gt_json, keep=None):
    """macro-F1, mAP, AP50 and per-class AP on the images `keep` (indices into cache; default all)."""
    keep = range(len(cache)) if keep is None else keep
    cls = classification_metrics([cache[i]['label'] for i in keep], [pred[i] for i in keep])
    det = detection_metrics(gt_json, [d for i in keep for d in dets[i]], [cache[i]['image_id'] for i in keep])
    return dict(macro_f1=cls['macro_f1'], accuracy=cls['accuracy'], mAP=det['mAP'], AP50=det['AP50'],
                per_class_AP=det['per_class_AP'])


def delta_m(a, b, ref):
    """Paired Δm of a over b (points), with the local reference's F1 / mAP (fractions) as denominators."""
    return 50 * ((a['macro_f1'] - b['macro_f1']) / ref[0] + (a['mAP'] - b['mAP']) / ref[1])


def report(cache, heads, gt_json, ref, halves=None):
    """ref = (macro-F1, mAP) as fractions: in real use, the LOCAL reference's 3-seed holdout means (ViT + LoRA on
    dataset/holdout_sarmtl), not the val reference; the self-test passes the val reference only as a placeholder."""
    run = {name: score(cache, *decode(cache, heads, **s), gt_json) for name, s in
           (('reference', REFERENCE), ('thresh 0.001', THRESH_ONLY), ('D1', D1))}
    for name, r in run.items():
        print(f'  {name:13s} macro-F1 {100 * r["macro_f1"]:6.2f}  mAP {100 * r["mAP"]:6.2f}  AP50 {100 * r["AP50"]:6.2f}')
    thr, prior = delta_m(run['thresh 0.001'], run['reference'], ref), delta_m(run['D1'], run['thresh 0.001'], ref)
    print(f'  paired Δm: threshold share {thr:+.2f}, prior share {prior:+.2f}, total {thr + prior:+.2f}; '
          f'adopt alpha = 1 on this seed: {prior > 0} (the rule needs it on all 3 seeds)')
    print('  per-class AP change, D1 vs thresh 0.001: ' + ', '.join(
        f'{k} {100 * (run["D1"]["per_class_AP"][k] - run["thresh 0.001"]["per_class_AP"][k]):+.1f}'
        for k in run['D1']['per_class_AP']))
    # cost of misclassification: D1 on the correctly classified images only (an oracle split, analysis only)
    pred0, dets0 = decode(cache, heads, **THRESH_ONLY)
    pred1, dets1 = decode(cache, heads, **D1)
    right = [p == c['label'] for p, c in zip(pred0, cache)]
    oracle = score(cache, pred0, [d1 if r else d0 for d0, d1, r in zip(dets0, dets1, right)], gt_json)
    print(f'  D1 on correctly classified images only: prior share {delta_m(oracle, run["thresh 0.001"], ref):+.2f} '
          f'vs on all images {prior:+.2f} (gap = cost of the {right.count(False)} misclassified images)')
    maxp = [c['logits'].softmax(0).max().item() for c in cache]
    for flag, name in ((False, 'wrongly'), (True, 'correctly')):
        v = [m for m, r in zip(maxp, right) if r == flag]
        print(f'  mean max-softmax on {name} classified images: {sum(v) / max(len(v), 1):.3f} ({len(v)} images)')
    # the analysis sweep, paired against the reference decoding
    print('  sweep (analysis only), paired Δm vs reference decoding:')
    sweep = []
    for s in SWEEP:
        r = score(cache, *decode(cache, heads, **s), gt_json)
        sweep.append((s, r))
        print(f'    {s}: Δm {delta_m(r, run["reference"], ref):+.2f} (mAP {100 * r["mAP"]:.2f}, '
              f'F1 {100 * r["macro_f1"]:.2f})')
    if halves:  # 2-fold cross-fit: choose on one half, score on the other
        index = {c['image_id']: i for i, c in enumerate(cache)}
        folds = [[index[i] for i in h if i in index] for h in (halves['A'], halves['B'])]
        for choose, test in ((0, 1), (1, 0)):
            base = score(cache, *decode(cache, heads, **REFERENCE), gt_json, folds[choose])
            best = max(SWEEP, key=lambda s: delta_m(score(cache, *decode(cache, heads, **s), gt_json, folds[choose]),
                                                    base, ref))
            out = delta_m(score(cache, *decode(cache, heads, **best), gt_json, folds[test]),
                          score(cache, *decode(cache, heads, **REFERENCE), gt_json, folds[test]), ref)
            print(f'  cross-fit: chosen on half {"AB"[choose]}: {best}; out-of-fold paired Δm on half '
                  f'{"AB"[test]}: {out:+.2f}')
    return run, sweep


def synthetic_cache(dataset, scenario, seed=0, conf=0.95):
    """A cache built from the ground truth instead of a model, so that the decodings can be compared where the answer
    is known. Every image: one RoI per GT box with zero regression (the decoded box is the GT box) and a right-class
    logit drawn in [0.5, 5] (scores 0.6-0.99), plus 5 background RoIs whose image-class logit is drawn in [-2, 2] (the
    usual false positives). The image classifier gives the right class p = 0.95. Scenarios:
      clean:         nothing else;
      confusion:     each GT RoI also gives a confuser class a logit drawn in [0.5, 5] (the RoI head is unsure);
      unannotated:   3 more RoIs per image on real objects of another class (logit 3-4, image class -1), like the
                     unannotated ships in harbor scenes;
      misclassified: the image classifier is confidently wrong (p = conf on another class) on every third image.
    conf: the classifier's confidence (0.95; 0.999 is close to one-hot, as a model with train CE ~ 0.03)."""
    g = torch.Generator().manual_seed(seed)
    u = lambda n, lo, hi: lo + (hi - lo) * torch.rand(n, generator=g)  # noqa: E731
    cache = []
    for i in range(len(dataset)):
        _, label, t = dataset[i]
        k, n = label + 1, len(t['boxes'])  # k: the detector's class index (1..9)
        boxes, logits = t['boxes'].clone(), torch.full((n, 10), -4.0)
        logits[:, 0], logits[:, k] = 0.0, u(n, 0.5, 5)
        if scenario == 'confusion':
            logits[:, 1 + (label + 1) % 9] = u(n, 0.5, 5)
        xy = torch.rand(5, 2, generator=g) * 480
        bg = torch.full((5, 10), -4.0)
        bg[:, 0], bg[:, k] = 0.0, u(5, -2, 2)
        boxes, logits = torch.cat([boxes, torch.cat([xy, xy + 8 + 24 * torch.rand(5, 2, generator=g)], 1)]), \
            torch.cat([logits, bg])
        if scenario == 'unannotated':
            xy = torch.rand(3, 2, generator=g) * 480
            other = torch.full((3, 10), -4.0)
            other[:, 0], other[:, k], other[:, 1 + (label + 4) % 9] = 0.0, -1.0, u(3, 3, 4)
            boxes = torch.cat([boxes, torch.cat([xy, xy + 8 + 24 * torch.rand(3, 2, generator=g)], 1)])
            logits = torch.cat([logits, other])
        image = torch.full((9,), (1 - conf) / 8).log()
        image[label] = torch.tensor(conf).log()
        if scenario == 'misclassified' and i % 3 == 0:
            image = torch.full((9,), (1 - conf) / 8).log()
            image[(label + 2) % 9] = torch.tensor(conf).log()
        cache.append(dict(proposals=boxes, class_logits=logits, box_regression=torch.zeros(len(boxes), 40),
                          logits=image, label=label, image_id=t['image_id'], orig_size=t['orig_size']))
    return cache


def compare_modes(cache, heads, gt_json):
    """mAP under the reference decoding, the product rule and the posterior rule (alpha 1 and hard)."""
    out = []
    for name, st in (('reference', REFERENCE), ('product a=1', D1), ('product hard', setting(alpha=INF)),
                     ('posterior a=1', setting(alpha=1.0, mode='posterior')),
                     ('posterior hard', setting(alpha=INF, mode='posterior'))):
        out.append(f'{name} {100 * score(cache, *decode(cache, heads, **st), gt_json)["mAP"]:5.1f}')
    return ', '.join(out)


def selftest():
    """CPU checks on 18 LOCAL-TRAIN images (2 per class, so that macro-F1 is defined; no holdout data is touched, so
    the a priori choice of the decoding mode uses none): (a) plumbing with a random-weight ViT + LoRA; (b) synthetic
    caches from the ground truth (see synthetic_cache), decoded with the reference, the product rule and the
    posterior rule."""
    from sarbench.adapters import add_adapters
    from sarbench.backbones import build_backbone
    torch.set_num_threads(16)
    torch.manual_seed(0)
    backbone = build_backbone('vit', pretrained=False)
    add_adapters(backbone, 'lora')
    model = DetectModel(backbone, task_routing=False)
    for p in model.parameters():
        p.data.add_(0.01 * torch.randn_like(p))
    data = ROOT / 'dataset/holdout_sarmtl'
    gt = data / 'train' / 'instances.json'  # the local-train side of the split
    local_train = SARMultiTask(data, 'train')
    first = {}
    for i, row in enumerate(local_train.rows):
        first.setdefault(int(row['label']), []).append(i)
    pick = [i for c in range(9) for i in first[c][:2]]
    subset = torch.utils.data.Subset(local_train, pick)
    holdout_ids = {int(r['image_id']) for r in SARMultiTask(data, 'val').rows}
    picked_ids = [int(local_train.rows[i]['image_id']) for i in pick]
    assert not holdout_ids & set(picked_ids), 'a self-test image is in the holdout'
    print(f'self-test images (local train, none in the {len(holdout_ids)}-image holdout): {picked_ids}')
    print('(a) random-weight model, 18 local-train images (plumbing only; mAP ~ 0 is expected)')
    cache = collect(model, subset, batch_size=6)
    print(f'  cached {len(cache)} images, RoIs per image {sorted({len(c["proposals"]) for c in cache})}')
    halves = {'A': [c['image_id'] for c in cache[0::2]], 'B': [c['image_id'] for c in cache[1::2]]}
    report(cache, model.roi_heads, gt, ref=(0.9331, 0.2613), halves=halves)
    for conf in (0.95, 0.999):
        print(f'(b) synthetic caches from the ground truth (18 local-train images), classifier confidence {conf}, '
              f'mAP by decoding, mean over 5 draws:')
        for scenario in ('clean', 'confusion', 'unannotated', 'misclassified'):
            runs = [compare_modes(synthetic_cache(subset, scenario, seed, conf), model.roi_heads, gt)
                    for seed in range(5)]
            names = [r.rsplit(' ', 1)[0] for r in runs[0].split(', ')]
            means = [sum(float(r.split(', ')[j].rsplit(' ', 1)[1]) for r in runs) / len(runs)
                     for j in range(len(names))]
            print(f'  {scenario:13s}: ' + ', '.join(f'{n} {m:5.1f}' for n, m in zip(names, means)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--selftest', action='store_true')
    args = p.parse_args()
    if args.selftest:
        selftest()
