"""What the three architectures share: the data and evaluation sets, the decoding of dense detectors, local Δm and the
parameter budget. Every model here has sarbench's MultiTaskModel interface: model(images [B, 1, 512, 512] normalised
gray, targets) -> (classification logits [B, 9], per-image detections in eval mode, detection losses in training mode),
with detections as {'boxes': XYXY on the 512 input (fp32), 'labels': 1..9, 'scores'}."""
import csv
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


def make_loader(root, split, train=False, flips=True, batch_size=16, workers=8, limit=None, seed=0):
    """sarbench's data at its 512 px geometry. Training: the order and flips of an epoch depend only on (seed, epoch)
    (set loader.sampler.epoch before each epoch), flips unless flips=False, the last partial batch dropped (BatchNorm)."""
    dataset = SARMultiTask(root, split, train=train and flips, limit=limit)
    sampler = None
    if train:
        dataset, sampler = EpochSeeded(dataset, seed), EpochSampler(len(dataset), seed)
    return torch.utils.data.DataLoader(dataset, batch_size, sampler=sampler, drop_last=train, num_workers=workers,
                                       collate_fn=collate, pin_memory=True, persistent_workers=train and workers > 0)
