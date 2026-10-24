"""Candidate XFuse: cross-task fusion at inference, from ONE forward pass of ONE model (0 parameters, no retraining).

Every box of an image has the image's class (CLAUDE.md), so the two heads predict the same thing at two scales:
  cls -> det:  s'(box) = s(box) * p_cls(label(box) | image) ** gamma   then the top `keep` boxes by s' per image
  det -> cls:  label'  = argmax_c [ log p_cls(c | image) + lam * max_{boxes of class c} s(box) ]
gamma = lam = 0 gives exactly the reference outputs when the RoI head keeps >= `keep` boxes per image.
Use: model.roi_heads.detections_per_img = 300 (eval only; the RoI head then returns up to 300 per-class boxes
after its usual 0.05 threshold and per-class NMS), run the model once, then call fuse(...).
Per-class NMS is unaffected by fusion: every box of one class in one image is scaled by the same factor."""
import torch


def fuse(logits, detections, gamma=1.0, lam=0.0, keep=100):
    """logits [B, 9]; detections: per-image dicts with 'boxes' [N, 4], 'labels' [N] in 1..9, 'scores' [N].
    Returns (predicted labels [B] in 0..8, fused per-image detections with at most `keep` boxes each)."""
    logp = logits.float().log_softmax(dim=1)
    p = logp.exp()
    evidence = torch.zeros_like(p)  # max raw box score per class and image (0 if none)
    fused = []
    for i, d in enumerate(detections):
        cls = d['labels'] - 1
        if len(cls):
            evidence[i].scatter_reduce_(0, cls, d['scores'].float(), reduce='amax', include_self=True)
        score = d['scores'].float() * p[i, cls] ** gamma
        top = score.argsort(descending=True)[:keep]
        fused.append({'boxes': d['boxes'][top], 'labels': d['labels'][top], 'scores': score[top]})
    labels = (logp + lam * evidence).argmax(dim=1)
    return labels, fused


def _self_test():
    """CPU check on a random-weight model: shapes, gamma = lam = 0 reproduces the reference outputs."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from wide_lora import ROOT  # noqa: F401  (puts the repo root on sys.path)
    from sarbench.adapters import add_adapters
    from sarbench.backbones import build_backbone
    from sarbench.model import MultiTaskModel
    torch.manual_seed(0)
    bb = build_backbone('vit', pretrained=False)
    add_adapters(bb, 'lora')
    model = MultiTaskModel(bb, task_routing=False).eval()
    x = torch.randn(2, 1, 512, 512)
    with torch.no_grad():
        logits, ref, _ = model(x)  # reference: top 100 per image
        model.roi_heads.detections_per_img = 300
        logits2, wide, _ = model(x)
    labels, fused = fuse(logits2, wide, gamma=0.0, lam=0.0)
    assert torch.equal(labels, logits.argmax(1))
    for r, f in zip(ref, fused):
        assert torch.equal(r['labels'], f['labels']) and torch.allclose(r['scores'], f['scores'])
        assert torch.allclose(r['boxes'], f['boxes'])
    labels, fused = fuse(logits2, wide, gamma=1.0, lam=2.0)
    print('ok: gamma = lam = 0 reproduces the reference outputs;',
          [len(d['scores']) for d in wide], 'boxes before fusion ->', [len(d['scores']) for d in fused], 'after')


if __name__ == '__main__':
    _self_test()
