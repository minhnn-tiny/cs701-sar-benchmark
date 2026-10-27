"""YOLO26 and RT-DETR (Ultralytics 8.4.189, COCO weights from github.com/ultralytics/assets v8.4.0) as multi-task models.
- One shared trunk: Ultralytics' backbone + neck feed the detector and a classification head (global average pool ->
  LayerNorm -> Linear, as the reference head) on the output of layer ARCHS[arch][2]; with cls_head='pyramid', on the
  mean and max of every map the detection head reads. Both train jointly.
- Detection: Ultralytics' own head and loss (YOLO26: one-to-many + one-to-one heads, ProgLoss, STAL; RT-DETR: Hungarian
  matching with denoising), 9 classes. Its loss runs in fp32, as do RT-DETR's decoder and every box decoding: in bf16,
  coordinates near 512 snap to a 2 px grid.
- Input: sarbench's 512 x 512 gray, mapped back to [0, 1] and repeated as R, G, B (what COCO-trained Ultralytics
  weights expect); nothing is resized, so the geometry stays the baseline's. No TTA (augment=False).
- Inference: dense_decode (common.py) under the decoding set with set_decoding."""
import os

os.environ.setdefault('YOLO_OFFLINE', 'True')  # no downloads, update checks or telemetry
os.environ.setdefault('YOLO_AUTOINSTALL', 'False')

from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torchvision.ops import box_convert  # noqa: E402
from ultralytics.models.utils.loss import RTDETRDetectionLoss  # noqa: E402
from ultralytics.nn.modules.head import RTDETRDecoder  # noqa: E402
from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel  # noqa: E402
from ultralytics.utils.tal import make_anchors  # noqa: E402

from common import ROOT, dense_decode, to_unit_rgb  # noqa: E402

WEIGHTS = ROOT / 'workspace' / 'mtdet' / 'weights'
# name: (Ultralytics model yaml, COCO checkpoint, layer feeding the classification head, frozen layers)
ARCHS = {
    'yolo26l': ('yolo26l.yaml', 'yolo26l.pt', 10, ()),  # 10: the backbone's last layer (C2PSA, P5)
    'yolo26l-p2': ('yolo26l-p2.yaml', 'yolo26l.pt', 10, ()),  # + a stride-4 level; its head is new
    'yolo26x': ('yolo26x.yaml', 'yolo26x.pt', 22, (6, 7, 8, 9, 10)),  # backbone P4/P5 frozen for the 40 M budget,
                                                                       # so the classifier reads the neck's P5 (22)
    'rtdetr-l': ('rtdetr-l.yaml', 'rtdetr-l.pt', 9, ()),  # 9: the HGNetv2 backbone's last stage
}
P2_SHIFT = {i: i + 6 for i in range(17, 23)}  # yolo26l layers 17-22 (P3 -> P5 path) are yolo26l-p2's 23-28
LOSS_GAINS = dict(box=7.5, cls=0.5, dfl=1.5)  # Ultralytics' defaults (dfl: the L1 box term of DFL-free YOLO26)


def _keep_output(module, inputs, output):
    module.kept = output  # stored on the module itself, so an EMA deep copy keeps its own


class FP32Decoder(RTDETRDecoder):
    """RT-DETR's decoder in fp32: its boxes are sigmoid outputs in [0, 1], which bf16 would round to 2 px at 512."""

    def forward(self, x, batch=None):
        with torch.autocast('cuda', enabled=False):
            return super().forward([f.float() for f in x], batch)


class FP32Loss(RTDETRDetectionLoss):
    """RT-DETR's matching and losses in fp32 (its inputs, the decoder's outputs, already are)."""

    def forward(self, *args, **kwargs):
        with torch.autocast('cuda', enabled=False):
            return super().forward(*args, **kwargs)


def load_coco(det, path, shift=None):
    """Copy every COCO tensor whose name (after the layer shift) and shape match; returns (copied, total) tensors."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)  # official Ultralytics release asset
    source = (ckpt.get('ema') or ckpt['model']).float().state_dict()
    if shift:
        def rename(key):
            parts = key.split('.')
            parts[1] = str(shift.get(int(parts[1]), parts[1]))
            return '.'.join(parts)
        source = {rename(k): v for k, v in source.items()}
    target = det.state_dict()
    copied = {k: v for k, v in source.items() if k in target and target[k].shape == v.shape}
    det.load_state_dict(copied, strict=False)
    return len(copied), len(target)


def set_decoding(model, head='o2o', jd_alpha=0.0, nms=None, topk=300, merge=False, score_thresh=0.001):
    model.decoding = dict(head=head, jd_alpha=jd_alpha, nms=nms, topk=topk, merge=merge, score_thresh=score_thresh)


def decoding_grid(arch):
    """The decodings evaluate.py sweeps. YOLO26: the one-to-one head (NMS-free, its native inference) or the
    one-to-many head with per-class NMS; RT-DETR: its queries, with or without NMS. Each with JD alpha 0 or 1 and 100 or
    300 boxes per image (the scorer keeps 100 per image and class). Added after Y_s0 (where 300 boxes changed nothing),
    with 100 boxes only: NMS 0.8, and merge NMS ('m' after the IoU) at every NMS IoU. Added 27 Oct ('_t0', M1.6): no
    score floor, so every image keeps its top k boxes; boxes ranked below all others can only add AP."""
    heads = [('q', None), ('q', 0.7)] if arch.startswith('rtdetr') else [('o2o', None), ('o2m', 0.6), ('o2m', 0.7)]
    grid = {f'{h}{"" if nms is None else f"_nms{nms}"}_a{a:g}_k{k}': dict(head=h, jd_alpha=a, nms=nms, topk=k)
            for h, nms in heads for a in (0.0, 1.0) for k in (100, 300)}
    more = [('q', 0.7, True)] if arch.startswith('rtdetr') else \
        [('o2m', 0.8, False)] + [('o2m', nms, True) for nms in (0.6, 0.7, 0.8)]
    grid.update({f'{h}_nms{nms}{"m" if merge else ""}_a{a:g}_k100': dict(head=h, jd_alpha=a, nms=nms, topk=100, merge=merge)
                 for h, nms, merge in more for a in (0.0, 1.0)})
    no_floor = [('q', 0.7, 100)] if arch.startswith('rtdetr') else [('o2m', 0.7, 100), ('o2m', 0.7, 300), ('o2m', 0.8, 100)]
    grid.update({f'{h}_nms{nms}_a0_k{k}_t0': dict(head=h, nms=nms, topk=k, score_thresh=0.0) for h, nms, k in no_floor})
    return grid


class UltraMultiTask(nn.Module):
    def __init__(self, arch='yolo26l', coco=True, epochs=24, num_classes=9, cls_head='p5'):
        super().__init__()
        cfg, ckpt, cls_layer, frozen = ARCHS[arch]
        self.rtdetr = arch.startswith('rtdetr')
        self.det = (RTDETRDetectionModel if self.rtdetr else DetectionModel)(cfg, nc=num_classes, verbose=False)
        self.det.args = SimpleNamespace(**LOSS_GAINS, epochs=epochs)  # read by the loss (and YOLO26's ProgLoss)
        self.det.nc = num_classes  # RT-DETR's loss reads it (Ultralytics' trainer sets it)
        self.coco_tensors = load_coco(self.det, WEIGHTS / ckpt, P2_SHIFT if 'p2' in arch else None) if coco else None
        if self.rtdetr:
            self.det.model[-1].__class__ = FP32Decoder
        # max pooling too in 'pyramid': a 10 px bridge is under 1 % of a mean over a stride-8 map
        self.cls_layers, self.cls_max = ([cls_layer], False) if cls_head == 'p5' else (list(self.det.model[-1].f), True)
        for i in self.cls_layers:
            self.det.model[i].register_forward_hook(_keep_output)
        self.det.eval()
        with torch.no_grad():  # the classification head's width
            self.det(torch.zeros(1, 3, 64, 64))
        width = self.features().shape[1]
        self.det.train()
        self.cls_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, num_classes))
        for i in frozen:
            self.det.model[i].requires_grad_(False)
        set_decoding(self, **({'head': 'q'} if self.rtdetr else {}))
        self.channels_last = False  # set by train.py together with the weights' memory format

    def features(self):
        """The pooled outputs of the classification layers from the last forward pass (and free them)."""
        pooled = []
        for i in self.cls_layers:
            kept = self.det.model[i].kept.float()
            del self.det.model[i].kept
            pooled += [kept.mean(dim=(2, 3)), kept.amax(dim=(2, 3))] if self.cls_max else [kept.mean(dim=(2, 3))]
        return torch.cat(pooled, 1)

    def is_backbone(self, name):
        """The pretrained backbone (YOLO26 layers 0-10, RT-DETR's HGNetv2 0-9), for --backbone-lr."""
        return name.startswith('det.model.') and int(name.split('.')[2]) <= (9 if self.rtdetr else 10)

    def build_loss(self):
        """Ultralytics' loss, on the model's device."""
        self.det.criterion = self.det.init_criterion()
        if self.rtdetr:
            self.det.criterion.__class__ = FP32Loss

    def epoch_done(self):
        """YOLO26's ProgLoss: shift weight from the one-to-many to the one-to-one loss after each epoch."""
        if hasattr(getattr(self.det, 'criterion', None), 'update'):
            self.det.criterion.update()

    def decodings(self):
        return decoding_grid('rtdetr' if self.rtdetr else 'yolo26')

    def apply_decoding(self, cfg):
        set_decoding(self, **cfg)

    def replay_epochs(self, epochs):
        """The loss state after `epochs` epochs (resume)."""
        self.build_loss()
        for _ in range(epochs):
            self.epoch_done()

    def forward(self, images, targets=None):
        x = to_unit_rgb(images)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        if self.training:
            batch = self._batch(targets, x)
            if getattr(self.det, 'criterion', None) is None:
                self.build_loss()
            if self.rtdetr:
                loss, items = self.det.loss(batch)  # the forward pass, with the targets for the denoising queries
                losses = {'det': loss}
            else:
                preds = self.det(x)
                with torch.autocast('cuda', enabled=False):
                    loss, _ = self.det.criterion(self._fp32(preds), batch)  # (box, cls, l1) x batch size
                losses = dict(zip(('box', 'cls_det', 'l1'), loss / len(images)))
            return self.cls_head(self.features()), None, losses
        raw = self.det(x)[1]
        logits = self.cls_head(self.features())
        boxes, scores = self._boxes_scores(raw, images.shape[-1])
        d = self.decoding
        return logits, dense_decode(boxes, scores, logits.float().softmax(1), d['jd_alpha'], d['nms'], d['topk'],
                                    d['score_thresh'], merge=d['merge']), None

    @staticmethod
    def _batch(targets, x):
        """sarbench targets (XYXY on the input, labels 1..9) -> Ultralytics' batch (normalised cx, cy, w, h; 0..8)."""
        size = x.shape[-1]
        boxes = [box_convert(t['boxes'], 'xyxy', 'cxcywh') / size for t in targets]
        return {'img': x,
                'batch_idx': torch.cat([torch.full((len(b),), i) for i, b in enumerate(boxes)]).float().to(x.device),
                'cls': torch.cat([t['labels'] - 1 for t in targets]).float().view(-1, 1).to(x.device),
                'bboxes': torch.cat(boxes).to(x.device)}

    @staticmethod
    def _fp32(preds):
        """The raw head outputs in fp32; feature maps only lend their shapes to the anchors, so empty fp32 stand-ins."""
        def convert(p):
            return {'boxes': p['boxes'].float(), 'scores': p['scores'].float(),
                    'feats': [f.new_empty((0, 0, *f.shape[2:]), dtype=torch.float32) for f in p['feats']]}
        return {k: convert(v) for k, v in preds.items()} if 'one2many' in preds else convert(preds)

    def _boxes_scores(self, raw, size):
        """Boxes [B, N, 4] XYXY on the input and per-class scores [B, N, nc], in fp32."""
        if self.rtdetr:
            dec_bboxes, dec_scores = raw[0][-1], raw[1][-1]  # the last decoder layer: [B, queries, 4 / nc]
            boxes = box_convert(dec_bboxes.float(), 'cxcywh', 'xyxy') * size
            return boxes.clamp(0, size), dec_scores.float().sigmoid()
        branch = raw['one2one' if self.decoding['head'] == 'o2o' else 'one2many']
        feats = [f.new_empty((0, 0, *f.shape[2:]), dtype=torch.float32) for f in branch['feats']]
        anchors, strides = make_anchors(feats, self.det.model[-1].stride, 0.5)
        dist = branch['boxes'].float().permute(0, 2, 1)  # [B, N, 4]: left, top, right, bottom in strides
        boxes = torch.cat([anchors - dist[..., :2], anchors + dist[..., 2:]], dim=-1) * strides
        return boxes.clamp(0, size), branch['scores'].float().sigmoid().permute(0, 2, 1)
