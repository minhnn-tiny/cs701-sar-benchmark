"""CPU checks (~2 min): budgets, joint forward/backward, decoding, data order, the zip, the holdout."""
import json

import pytest
import torch

from common import (HOLDOUT, MEAN, ROOT, STD, Mosaic, SARAugment, SARMultiTask, Zoom, check_budget, collate,
                    dense_decode, make_loader)


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



def test_sar_augment_is_photometric_and_seeded():
    """The same seed gives the same pixels; boxes and labels are untouched; intensities stay in [0, 1]."""
    plain = SARMultiTask(HOLDOUT, 'train', limit=4)
    runs = []
    for _ in range(2):
        torch.manual_seed(5)
        runs.append([SARAugment(plain)[i] for i in range(4)])
    for (a, label, target), (b, _, _), (c, label_c, target_c) in zip(*runs, [plain[i] for i in range(4)]):
        assert torch.equal(a, b) and a.shape == c.shape and label == label_c
        assert torch.equal(target['boxes'], target_c['boxes'])
        assert -1e-5 <= float((a * STD + MEAN).min()) and float((a * STD + MEAN).max()) <= 1 + 1e-5
    assert any(not torch.allclose(a, plain[i][0]) for i, (a, _, _) in enumerate(runs[0]))



class Squares(torch.utils.data.Dataset):
    """One bright square per image, with its box, on a dark background (normalised units)."""
    rows = [{'label': str(i % 9)} for i in range(18)]

    def __len__(self):
        return 18

    def __getitem__(self, i):
        x, y = torch.randint(0, 480, (2,), generator=torch.Generator().manual_seed(i)).tolist()
        image = torch.full((1, 512, 512), -1.0)
        image[:, y:y + 8 + i, x:x + 8 + i] = 3.0
        return image, i % 9, {'boxes': torch.tensor([[x, y, x + 8.0 + i, y + 8.0 + i]]), 'labels': torch.tensor([i % 9 + 1])}


@pytest.mark.parametrize('wrap', [lambda d: Zoom(d, 0.5), Mosaic], ids=['zoom', 'mosaic'])
def test_geometric_augment_moves_boxes_with_pixels(wrap):
    """Every box Zoom and Mosaic return covers the square's pixels; labels stay the image's class; seeded."""
    aug, changed = wrap(Squares()), 0
    for i in range(18):
        torch.manual_seed(i)
        image, label, target = aug[i]
        torch.manual_seed(i)
        again = aug[i]
        assert torch.equal(image, again[0]) and torch.equal(target['boxes'], again[2]['boxes'])
        assert image.shape == (1, 512, 512) and len(target['boxes']) >= 1 and (target['labels'] == label + 1).all()
        for x1, y1, x2, y2 in target['boxes'].round().int().tolist():
            assert image[0, y1:y2, x1:x2].mean() > 0.5
        changed += not torch.equal(image, Squares()[i][0])
    assert changed >= 3


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


def test_probe_head_swap():
    """probe.py: a head fit on cached features drops into a model built with that cls_head (strict load)."""
    from probe import fit, with_head
    from train import build_model, parse_args
    p5 = build_model(parse_args(['--arch', 'yolo26l', '--out', '/tmp/unused']), coco=False)
    pyramid = build_model(parse_args(['--arch', 'yolo26l', '--cls-head', 'pyramid', '--out', '/tmp/unused']), coco=False)
    x, y = torch.randn(64, pyramid.cls_head[0].normalized_shape[0]), torch.arange(64) % 9
    head = fit(x, y, epochs=30)
    assert (head(x).argmax(1) == y).float().mean() > 0.9  # random features in 2,560 dimensions are separable
    pyramid.load_state_dict(with_head(p5.state_dict(), head))
