"""What the three architectures share: the data and evaluation sets, the decoding of dense detectors, local Δm and the
parameter budget. Every model here has sarbench's MultiTaskModel interface: model(images [B, 1, 512, 512] normalised
gray, targets) -> (classification logits [B, 9], per-image detections in eval mode, detection losses in training mode),
with detections as {'boxes': XYXY on the 512 input (fp32), 'labels': 1..9, 'scores'}."""
import csv
import math
import sys
from pathlib import Path

import torch
from torchvision.ops import batched_nms, box_iou

ROOT = Path(__file__).resolve().parents[2]
# appended, not prepended: this design's train/evaluate/predict must shadow root train.py and sarmtl's
sys.path += [str(ROOT), str(ROOT / 'workspace'), str(ROOT / 'workspace' / 'sarmtl')]

from sarbench.data import MEAN, STD, SARMultiTask, collate, to_original_xywh  # noqa: E402
from sarbench.metrics import classification_metrics, detection_metrics  # noqa: E402

HOLDOUT = ROOT / 'dataset' / 'holdout_fitdet'  # ND = its val: a random 10 % of holdout_sarmtl/train, drawn like val
EVAL_SETS = {'lf': ROOT / 'dataset/holdout_sarmtl/val',  # leak-free: no near-duplicate in any local train
             'fit': ROOT / 'dataset/fitdet/fit'}  # 373 local-train images: how well a run fits its own training data
REFERENCE = (93.31, 26.13)  # the leaderboard's val reference (ViT + LoRA): macro-F1, mAP in %
MAX_TOTAL, MAX_TRAINABLE = 130_000_000, 40_000_000


def split_images(root, split):
    """The resolved image paths of a split: the derived datasets link images/ to dataset/train/images, so one image has
    one path in all of them."""
    root = Path(root) / split
    with open(root / 'labels.csv') as f:
        return {(root / row['file_name']).resolve() for row in csv.DictReader(f)}


def eval_dirs(data, names):
    """{name: split directory}; 'nd' is the data's own holdout (<data>/val)."""
    return {name: Path(data) / 'val' if name == 'nd' else EVAL_SETS[name] for name in names}


def delta_m(macro_f1, mAP):
    """Δm in % against the leaderboard's val reference, from fractions (sarbench's metrics)."""
    f1_ref, map_ref = REFERENCE
    return 50 * ((100 * macro_f1 - f1_ref) / f1_ref + (100 * mAP - map_ref) / map_ref)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def check_budget(model):
    total, trainable = count_parameters(model)
    assert total <= MAX_TOTAL and trainable <= MAX_TRAINABLE, f'over budget: {total:,} total, {trainable:,} trainable'
    return total, trainable


def to_unit_rgb(images):
    """sarbench's normalised gray -> the [0, 1] 3-channel input that COCO-trained YOLO / RT-DETR weights expect."""
    return (images * STD + MEAN).expand(-1, 3, -1, -1)


def dense_decode(boxes, scores, probs, jd_alpha=0.0, nms=None, topk=300, score_thresh=0.001, pre_nms=3000, merge=False):
    """Detections from a dense detector: boxes [B, N, 4] XYXY on the input (fp32), per-class scores [B, N, 9] in [0, 1],
    the image classifier's softmax probs [B, 9].
    - JD: each (box, class) score is multiplied by probs[class] ** jd_alpha (0: the detector alone), so a box can be
      listed under the classes the image may have, ranked by both heads.
    - nms=None: the top-k (box, class) pairs (NMS-free heads: YOLO26's one-to-one head, RT-DETR's queries);
      else per-class NMS at that IoU over the top pre_nms pairs above score_thresh, then the top-k.
    - merge (with nms): YOLOv5's merge NMS: each kept box becomes the score-weighted mean of the same-class boxes it
      suppresses (itself included)."""
    if jd_alpha:
        scores = scores * probs[:, None, :].pow(jd_alpha)
    out = []
    for b, s in zip(boxes, scores):
        flat = s.flatten()
        k = min(topk if nms is None else pre_nms, flat.numel())
        value, index = flat.topk(k)
        keep = value > score_thresh
        value, index = value[keep], index[keep]
        box, cls = b[index // s.shape[1]], index % s.shape[1]
        if nms is not None:
            kept = batched_nms(box, value, cls, nms)[:topk]
            merged = box[kept]
            if merge:
                w = (box_iou(merged, box) > nms) * (cls[kept, None] == cls) * value
                merged = w @ box / w.sum(1, keepdim=True)
            box, cls, value = merged, cls[kept], value[kept]
        out.append({'boxes': box, 'labels': cls + 1, 'scores': value})
    return out


@torch.no_grad()
def evaluate(model, loader, gt_json):
    """Classification and COCO box metrics of one split (sarbench's), and its detections in COCO result format."""
    model.eval()
    labels, predicted, detections, image_ids = [], [], [], []
    for images, y, targets in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, outputs, _ = model(images.cuda(non_blocking=True))
        labels += y.tolist()
        predicted += logits.argmax(dim=1).tolist()
        for target, out in zip(targets, outputs):
            image_ids.append(target['image_id'])
            xywh = to_original_xywh(out['boxes'].float().cpu(), target['orig_size']).tolist()
            detections += [{'image_id': target['image_id'], 'category_id': int(c), 'bbox': b, 'score': float(s)}
                           for b, c, s in zip(xywh, out['labels'].tolist(), out['scores'].tolist())]
    metrics = {**classification_metrics(labels, predicted), **detection_metrics(gt_json, detections, image_ids)}
    metrics['delta_m'] = delta_m(metrics['macro_f1'], metrics['mAP'])
    return metrics, detections


class EpochSampler(torch.utils.data.Sampler):
    """Training order: a permutation drawn from (seed, epoch) alone; yields (epoch, index) so the dataset can seed the
    augmentation of each item the same way."""

    def __init__(self, n, seed):
        self.n, self.seed, self.epoch = n, seed, 1

    def __iter__(self):
        order = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed * 1000 + self.epoch))
        return iter([(self.epoch, int(i)) for i in order])

    def __len__(self):
        return self.n


class EpochSeeded(torch.utils.data.Dataset):
    """sarbench's dataset with torch's RNG seeded per (seed, epoch, image) before each item, so its random flips do not
    depend on which worker made the item or on a restart."""

    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        epoch, i = key
        torch.manual_seed((self.seed * 1000 + epoch) * 100_000 + i)
        return self.dataset[i]


class SARAugment(torch.utils.data.Dataset):
    """SAR-aware photometric augmentation after sarbench's flips (the TA's suggestion), each with probability p (default
    half the time): extra multiplicative speckle, the amplitude of an L-look Gamma intensity with L in 4-16 (another look
    at the scene, like a revisit), and a gamma of 2/3-3/2 (another source's intensity mapping). Boxes and labels are
    unchanged; draws come from torch's RNG, which EpochSeeded seeds per (seed, epoch, image)."""

    def __init__(self, dataset, p=0.5):
        self.dataset, self.p = dataset, p

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label, target = self.dataset[i]
        x = image * STD + MEAN  # sarbench's normalisation undone: intensities in [0, 1]
        if torch.rand(()) < self.p:
            looks = float(torch.empty(()).uniform_(4, 16))
            x = x * torch.distributions.Gamma(looks, looks).sample(x.shape).sqrt()
        if torch.rand(()) < self.p:
            x = x.clamp(0, 1) ** math.exp(float(torch.empty(()).uniform_(math.log(2 / 3), math.log(3 / 2))))
        return (x.clamp(0, 1) - MEAN) / STD, label, target


def _view(boxes, scale, dx, dy, size=512):
    """XYXY boxes scaled by `scale`, shifted by (dx, dy) and clipped to a size x size view; a box stays if at least a
    quarter of it is in view. Returns the boxes kept and the mask."""
    moved = boxes * scale + boxes.new_tensor([dx, dy, dx, dy])
    clipped = moved.clamp(0, size)
    keep = (clipped[:, 2:] - clipped[:, :2]).prod(1) >= 0.25 * (moved[:, 2:] - moved[:, :2]).prod(1)
    return clipped[keep], keep


class Zoom(torch.utils.data.Dataset):
    """Scale jitter half the time: the image resized by s, uniform in [1 - spread, 1 + spread], then a random 512 crop of
    it (s > 1) or put at random on a 512 canvas of mean grey (s < 1); boxes follow (_view). Redrawn until a box stays in
    view (10 tries, else the image as it is), so the image's class stays visible."""

    def __init__(self, dataset, spread):
        self.dataset, self.spread = dataset, spread

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label, target = self.dataset[i]
        if torch.rand(()) < 0.5:
            for _ in range(10):
                n = round(512 * float(torch.empty(()).uniform_(1 - self.spread, 1 + self.spread)))
                x, y = (int(torch.randint(abs(n - 512) + 1, ())) for _ in range(2))
                boxes, keep = _view(target['boxes'], n / 512, *((-x, -y) if n > 512 else (x, y)))
                if keep.any():
                    resized = torch.nn.functional.interpolate(image[None], size=(n, n), mode='bilinear',
                                                              antialias=n < 512)[0]
                    if n > 512:
                        image = resized[:, y:y + 512, x:x + 512]
                    else:
                        image = resized.new_zeros(1, 512, 512)
                        image[:, y:y + n, x:x + n] = resized
                    return image, label, {**target, 'boxes': boxes, 'labels': target['labels'][keep]}
        return image, label, target


class Mosaic(torch.utils.data.Dataset):
    """Half the time, a 2 x 2 mosaic of the image and three others of its class (so the image's label still holds), cut
    to a random 512 window: parts of up to four scenes at their own scale, with their boxes (_view). Redrawn until a box
    stays in view (10 tries, else the image as it is)."""

    def __init__(self, dataset):
        self.dataset = dataset
        labels = torch.tensor([int(row['label']) for row in dataset.rows])
        self.by_class = [torch.nonzero(labels == c).flatten() for c in range(9)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label, target = self.dataset[i]
        if torch.rand(()) >= 0.5:
            return image, label, target
        same = self.by_class[label]
        tiles = [(image, target)] + [self.dataset[int(same[torch.randint(len(same), ())])][::2] for _ in range(3)]
        canvas = image.new_zeros(1, 1024, 1024)
        for k, (tile, _) in enumerate(tiles):
            canvas[:, 512 * (k // 2):512 * (k // 2) + 512, 512 * (k % 2):512 * (k % 2) + 512] = tile
        for _ in range(10):
            x, y = (int(torch.randint(513, ())) for _ in range(2))
            views = [_view(t['boxes'], 1, 512 * (k % 2) - x, 512 * (k // 2) - y) for k, (_, t) in enumerate(tiles)]
            if any(keep.any() for _, keep in views):
                return canvas[:, y:y + 512, x:x + 512], label, {
                    **target, 'boxes': torch.cat([b for b, _ in views]),
                    'labels': torch.cat([t['labels'][keep] for (_, t), (_, keep) in zip(tiles, views)])}
        return image, label, target


def make_loader(root, split, train=False, flips=True, batch_size=16, workers=8, limit=None, seed=0, sar_aug=False,
                zoom=0.0, mosaic=False, sar_p=0.5):
    """sarbench's data at its 512 px geometry. Training: the order and flips of an epoch depend only on (seed, epoch)
    (set loader.sampler.epoch before each epoch), flips unless flips=False, then Mosaic, Zoom (spread `zoom`) and
    SARAugment if asked, the last partial batch dropped (BatchNorm)."""
    dataset = SARMultiTask(root, split, train=train and flips, limit=limit)
    if train and mosaic:
        dataset = Mosaic(dataset)
    if train and zoom:
        dataset = Zoom(dataset, zoom)
    if train and sar_aug:
        dataset = SARAugment(dataset, sar_p)
    sampler = None
    if train:
        dataset, sampler = EpochSeeded(dataset, seed), EpochSampler(len(dataset), seed)
    return torch.utils.data.DataLoader(dataset, batch_size, sampler=sampler, drop_last=train, num_workers=workers,
                                       collate_fn=collate, pin_memory=True, persistent_workers=train and workers > 0)
