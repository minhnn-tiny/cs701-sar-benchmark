"""CPU checks (~2 min): budgets, joint forward/backward, decoding, data order, the zip, the holdout."""
import json

import pytest
import torch

from common import HOLDOUT, ROOT, SARMultiTask, check_budget, collate, dense_decode, make_loader


@pytest.fixture(scope='module')
def batch():
    ds = SARMultiTask(HOLDOUT, 'train', train=True, limit=4)
    return collate([ds[i] for i in range(2)])


@pytest.mark.parametrize('arch', ['yolo26l', 'yolo26l-p2', 'yolo26x', 'rtdetr-l', 'convnext-b',
                                  'yolo26l --cls-head pyramid'])
def test_budget_forward_backward(arch, batch):
    from train import build_model, parse_args
    model = build_model(parse_args(['--arch', *arch.split(), '--out', '/tmp/unused']))
    total, trainable = check_budget(model)  # asserts <= 130 M / 40 M
    images, labels, targets = batch
    model.train()
    logits, _, losses = model(images, targets)
    loss = torch.nn.functional.cross_entropy(logits, labels) + sum(losses.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad), 'a trainable part gets no gradient'
    model.eval()
    with torch.no_grad():
        logits, dets, _ = model(images)
    assert logits.shape == (2, 9)
    for d in dets:
        assert d['boxes'].dtype == torch.float32 and ((d['boxes'] >= 0) & (d['boxes'] <= 512)).all()
        assert ((d['labels'] >= 1) & (d['labels'] <= 9)).all() and len(d['boxes']) <= 300
    for name, cfg in model.decodings().items():  # every decoding of the sweep runs
        model.apply_decoding(cfg)
        with torch.no_grad():
            model(images)


def test_dense_decode():
    boxes = torch.tensor([[[0., 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]]])
    scores = torch.tensor([[[0.9, 0.1], [0.8, 0.2], [0.5, 0.6]]])
    probs = torch.tensor([[0.1, 0.9]])
    out = dense_decode(boxes, scores, probs, topk=2)[0]  # no NMS: the top 2 (box, class) pairs
    assert out['labels'].tolist() == [1, 1] and torch.allclose(out['scores'], torch.tensor([0.9, 0.8]))
    out = dense_decode(boxes, scores, probs, nms=0.5, topk=10)[0]  # box 1 suppressed by box 0 in class 1
    assert sorted(out['scores'][out['labels'] == 1].tolist()) == pytest.approx([0.5, 0.9])  # boxes 2 and 0
    out = dense_decode(boxes, scores, probs, jd_alpha=1.0, topk=1)[0]  # JD: 0.6 * 0.9 beats 0.9 * 0.1
    assert out['labels'].tolist() == [2] and out['boxes'][0].tolist() == [50, 50, 60, 60]
    out = dense_decode(boxes, scores, probs, nms=0.5, topk=10, merge=True)[0]  # kept boxes absorb, per class, what they suppress
    by_score = {round(s, 4): b for s, b in zip(out['scores'].tolist(), out['boxes'].tolist())}
    assert by_score[0.9] == pytest.approx([0.8 / 1.7, 0.8 / 1.7, 10, 10])  # class 1: box 0 absorbs box 1
    assert by_score[0.2] == pytest.approx([0.2 / 0.3, 0.2 / 0.3, 10, 10])  # class 2: box 1 absorbs box 0
    assert by_score[0.6] == pytest.approx([50, 50, 60, 60]) and len(by_score) == 4  # alone in class 2 (and 0.5 in class 1)


def test_epoch_order_and_flips_are_reproducible():
    """An epoch's images and flips depend only on (seed, epoch): the same from a fresh loader (a resumed run) and from
    persistent workers that already ran other epochs."""
    def epoch(loader, e):
        loader.sampler.epoch = e
        return [(t['image_id'], t['boxes'].tolist()) for _, _, ts in loader for t in ts]
    running = make_loader(HOLDOUT, 'train', train=True, batch_size=4, workers=2, limit=24, seed=3)
    running.pin_memory = False  # the CPU suite must not touch a GPU: pinning opens a CUDA context on GPU 0 (faulty)
    first, second = epoch(running, 1), epoch(running, 2)
    resumed = make_loader(HOLDOUT, 'train', train=True, batch_size=4, workers=2, limit=24, seed=3)
    resumed.pin_memory = False
    assert epoch(resumed, 2) == second and first != second and len(second) == 24


def test_compact_zip_passes_check(tmp_path):
    from check_submission import check
    from make_submission import write_submission
    from predict import compact
    images = json.loads((ROOT / 'dataset' / 'val' / 'images.json').read_text())['images'][:5]
    preds = {im['file_name']: (3, [[10., 10., 20., 520.], [0., 0., 511.999, 30.]], [3, 3], [0.9, 0.123456789])
             for im in images}
    out = tmp_path / 'sub.zip'
    write_submission(images, preds, out=out)
    compact(out, images)
    problems, _ = check(out, images)
    assert not problems, dict(problems)


def test_holdout_disjoint():
    from common import EVAL_SETS, split_images
    train = split_images(HOLDOUT, 'train')
    assert not train & split_images(HOLDOUT, 'val') and not train & split_images(EVAL_SETS['lf'].parent, 'val')
    assert split_images(EVAL_SETS['fit'].parent, 'fit') <= train  # FIT: training images
