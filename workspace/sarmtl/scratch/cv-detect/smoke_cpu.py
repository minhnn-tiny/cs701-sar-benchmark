"""CPU checks of the prototype (no GPU, random weights, 2 synthetic 512 px images):
1. DetectModel(D2 off, alpha = lambda = 0) loads a MultiTaskModel state_dict and gives identical outputs;
2. D2 trains: under CPU bf16 autocast, finite losses; at step 1 only the (zero-init) lateral GN gets gradients, after
   one AdamW step gradients reach neck.prior;
3. D1 at inference: with alpha > 0, every image's detections move towards the predicted image class;
4. D1 math on the RoI head alone: posterior mode, a uniform prior changes nothing and a one-hot prior gives the image
   class the RoI's whole objectness; product mode, a one-hot prior keeps only the image class at its raw score and no
   score ever goes up; D1b with a large lambda predicts the class of the image's top box;
5. common random numbers + zero-init: built after the same seed, the D2 model leaves the global RNG where the no-D2
   model leaves it, shares every reference parameter bit for bit, and gives identical losses and detections at init.
Run from the repo root:  HF_HUB_OFFLINE=1 uv run python workspace/sarmtl/scratch/cv-detect/smoke_cpu.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_modules import DetectModel  # noqa: E402
from sarbench.adapters import add_adapters  # noqa: E402
from sarbench.backbones import build_backbone  # noqa: E402
from sarbench.model import MultiTaskModel  # noqa: E402

IMAGES = None
TARGETS = [{'boxes': torch.tensor([[100., 100., 108., 107.], [10., 20., 30., 44.]]), 'labels': torch.tensor([9, 9])},
           {'boxes': torch.tensor([[200., 220., 260., 300.]]), 'labels': torch.tensor([1])}]


def lora_vit():
    b = build_backbone('vit', pretrained=False)
    add_adapters(b, 'lora')
    return b


def train_losses(model, seed=123):
    """One bf16-autocast training forward with the global RNG at `seed` (the RPN / RoI samplers draw from it)."""
    model.train()
    torch.manual_seed(seed)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        logits, _, losses = model(IMAGES, TARGETS)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([8, 0])) + sum(losses.values())
    return loss, losses


def main():
    global IMAGES
    torch.set_num_threads(16)
    torch.manual_seed(0)
    IMAGES = torch.randn(2, 1, 512, 512)

    # 1. equivalence with the reference
    ref = MultiTaskModel(lora_vit(), task_routing=False).eval()
    for p in ref.parameters():  # LoRA up = 0 and random heads: perturb so that the check is not trivial
        p.data.add_(0.01 * torch.randn_like(p))
    same = DetectModel(lora_vit(), task_routing=False).eval()
    same.load_state_dict(ref.state_dict())
    with torch.no_grad():
        a, b = ref(IMAGES), same(IMAGES)
    print('1. reference == DetectModel(D2 off, alpha = lambda = 0):', torch.equal(a[0], b[0]) and all(
        torch.equal(x[k], y[k]) for x, y in zip(a[1], b[1]) for k in x), '| detections per image',
        [len(x['boxes']) for x in a[1]])

    # 2. D2 trains (zero-init lateral GN: the branch gets gradients from the second step on)
    d2 = DetectModel(lora_vit(), task_routing=False, spatial_prior=True, prior_seed=10_000)
    opt = torch.optim.AdamW([p for p in d2.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.05)
    grads = []
    for step in range(2):
        opt.zero_grad()
        loss, losses = train_losses(d2)
        loss.backward()
        grads.append({k: sum(p.grad.abs().sum().item() for n, p in d2.named_parameters()
                             if n.startswith(k) and p.grad is not None) for k in ('neck.prior.', 'neck.lateral.')})
        opt.step()
    print('2. D2 train steps: losses', {k: round(v.item(), 4) for k, v in losses.items()},
          f'| |grad| step 1: prior {grads[0]["neck.prior."]:.3g}, lateral {grads[0]["neck.lateral."]:.3g}'
          f'; step 2: prior {grads[1]["neck.prior."]:.3g}, lateral {grads[1]["neck.lateral."]:.3g}',
          '| fp32 RPN/RoI losses:', all(t.dtype == torch.float32 for t in losses.values()))

    # 3. D1 at inference, on the perturbed reference weights (the a priori setting: product, alpha 1, thresh 0.001)
    for mode, alpha, thresh in (('product', 0.0, 0.05), ('product', 1.0, 0.001), ('product', float('inf'), 0.001),
                                ('posterior', 1.0, 0.001)):
        same.prior_mode, same.prior_alpha, same.roi_heads.score_thresh = mode, alpha, thresh
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            logits, dets, _ = same(IMAGES)
        pred = (logits.argmax(1) + 1).tolist()
        share = [round((d['labels'] == c).float().mean().item(), 2) if len(d['labels']) else None
                 for d, c in zip(dets, pred)]
        print(f'3. D1 {mode} alpha {alpha} thresh {thresh}: predicted image class {pred}, share of detections with '
              f'that class {share}, detections per image {[len(d["labels"]) for d in dets]}')
    same.prior_mode, same.prior_alpha, same.roi_heads.score_thresh = 'product', 0.0, 0.05

    # 4. D1 math on the RoI head alone (random RoIs and logits, 2 images)
    heads = same.roi_heads
    proposals = [torch.tensor([[10., 10., 30., 30.], [100., 100., 140., 120.], [300., 300., 310., 312.]]),
                 torch.tensor([[50., 60., 90., 80.], [200., 20., 260., 70.]])]
    class_logits, box_regression = torch.randn(5, 10), 0.1 * torch.randn(5, 40)
    shapes = [(512, 512)] * 2
    ref_out = heads.postprocess_detections(class_logits, box_regression, proposals, shapes)
    uniform = torch.full((2, 9), -torch.log(torch.tensor(9.)))
    onehot = torch.full((2, 9), -1e4)
    onehot[0, 3], onehot[1, 6] = 0.0, 0.0
    heads.score_thresh = 0.0
    raw = heads.postprocess_detections(class_logits, box_regression, proposals, shapes)
    raw_score = {(i, tuple(b.tolist()), int(l)): float(sc) for i in range(2) for b, sc, l in zip(*[r[i] for r in raw])}
    # posterior mode: a uniform prior is the reference; a one-hot prior gives the image class the RoI's objectness
    heads.score_thresh = 0.05
    heads.decode = dict(logp=uniform, alpha=2.0, lam=0.0, mode='posterior')
    uni = heads.postprocess_detections(class_logits, box_regression, proposals, shapes)
    post_uniform = all(torch.allclose(a, b) for x, y in zip(ref_out, uni) for a, b in zip(x, y))
    heads.decode, heads.score_thresh = dict(logp=onehot, alpha=1.0, lam=0.0, mode='posterior'), 0.0
    boxes, scores, labels = heads.postprocess_detections(class_logits, box_regression, proposals, shapes)
    objectness = (1 - class_logits.softmax(-1)[:, 0]).split([3, 2])
    post_onehot = (labels[0] == 4).all().item() and (labels[1] == 7).all().item() and all(
        torch.allclose(scores[i].sort().values, objectness[i].sort().values, atol=1e-5) for i in range(2))
    # product mode: a one-hot prior keeps only the image class, at its raw score; no score ever goes up
    heads.decode = dict(logp=onehot, alpha=1.0, lam=0.0, mode='product')
    boxes, scores, labels = heads.postprocess_detections(class_logits, box_regression, proposals, shapes)
    prod_onehot = (labels[0] == 4).all().item() and (labels[1] == 7).all().item() and all(
        abs(float(sc) - raw_score[(i, tuple(b.tolist()), int(l))]) < 1e-5
        for i in range(2) for b, sc, l in zip(boxes[i], scores[i], labels[i]))
    heads.decode = dict(logp=torch.randn(2, 9).log_softmax(-1), alpha=1.0, lam=0.0, mode='product')
    boxes, scores, labels = heads.postprocess_detections(class_logits, box_regression, proposals, shapes)
    prod_never_up = all(float(sc) <= raw_score[(i, tuple(b.tolist()), int(l))] + 1e-6
                        for i in range(2) for b, sc, l in zip(boxes[i], scores[i], labels[i]))
    heads.decode = dict(logp=torch.zeros(2, 9).log_softmax(-1), alpha=0.0, lam=50.0)
    heads.postprocess_detections(class_logits, box_regression, proposals, shapes)
    top_class = torch.stack([s.max(0).values for s in class_logits.softmax(-1)[:, 1:].split([3, 2])]).argmax(1)
    ok_lam = torch.equal(heads.image_logp.argmax(1), top_class)
    heads.decode, heads.score_thresh = None, 0.05
    print(f'4. posterior: uniform prior = reference {post_uniform}, one-hot prior -> image class at the RoI objectness '
          f'{post_onehot}; product: one-hot prior -> image class only, at its raw score {prod_onehot}, no score ever '
          f'goes up {prod_never_up}; D1b lambda 50 -> class of the top box {ok_lam}')

    # 5. common random numbers and zero-init: the D2 arm starts as the no-D2 arm of the same seed
    torch.manual_seed(7)
    arm0 = DetectModel(lora_vit(), task_routing=False)
    rng0 = torch.get_rng_state()
    torch.manual_seed(7)
    arm1 = DetectModel(lora_vit(), task_routing=False, spatial_prior=True, prior_seed=7 + 10_000)
    rng1 = torch.get_rng_state()
    p0, p1 = dict(arm0.named_parameters()), dict(arm1.named_parameters())
    shared = all(torch.equal(v, p1[k]) for k, v in p0.items())
    new = sorted({k.split('.')[1] for k in p1 if k not in p0})
    with torch.no_grad():
        l0, l1 = train_losses(arm0)[1], train_losses(arm1)[1]
        arm0.eval(), arm1.eval()
        e0, e1 = arm0(IMAGES), arm1(IMAGES)
    same_losses = all(torch.equal(l0[k], l1[k]) for k in l0)
    same_dets = torch.equal(e0[0], e1[0]) and all(torch.equal(x[k], y[k]) for x, y in zip(e0[1], e1[1]) for k in x)
    print(f'5. same global RNG state after construction: {torch.equal(rng0, rng1)}; every reference parameter '
          f'identical: {shared} (new: neck.{new}); identical training losses: {same_losses}; identical eval '
          f'outputs: {same_dets}')


if __name__ == '__main__':
    main()
