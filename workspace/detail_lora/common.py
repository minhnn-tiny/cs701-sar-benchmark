"""Shared by train.py, evaluate.py and predict.py: the reference code reused unchanged, rebuilding a trained run's
model (a DetailLoRA run or a reference run of the top-level train.py), and inference timing."""
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]  # the repo root
sys.path.insert(0, str(ROOT))
from sarbench.adapters import add_adapters  # noqa: E402
from sarbench.backbones import build_backbone  # noqa: E402
from sarbench.model import MultiTaskModel  # noqa: E402

from detail_model import add_class_guidance, build_model  # noqa: E402

FUSIONS = ('none', 'hard', 'soft')


def load_module(name, path):
    """Import a script by path (the reference train.py and baseline predict.py share names with ours)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The reference training loop, evaluation (accuracy, macro-F1, COCO AP), schedule and logging, unchanged
reference_train = load_module('reference_train', ROOT / 'train.py')
# The reference preprocessing of unlabelled val/test images (images.json)
reference_predict = load_module('reference_predict', ROOT / 'workspace' / 'baseline' / 'predict.py')


def is_detail_run(metrics):
    return metrics.get('design') == 'detail_lora'


def model_from_args(args, pretrained=False):
    """DetailLoRA model as configured by the arguments of workspace/detail_lora/train.py (runs from before
    --backbone/--adapt existed are ViT + dense LoRA)."""
    return build_model(args.get('backbone', 'vit'), pretrained=pretrained, rank=args['rank'],
                       dense=not args['attn_only'], unfreeze=not args['freeze_pretrained'],
                       detail=not args['no_detail'], small_anchors=not args['reference_anchors'],
                       fusion=args['fusion'], adapt=args.get('adapt', 'dense'))


def load_run(run_dir, fusion=None):
    """The trained model of a run directory (metrics.json + model.pt), on the GPU in eval mode, with class-guided
    scoring available. fusion: detection scoring, default the run's own ('none' for reference runs)."""
    metrics = json.loads((Path(run_dir) / 'metrics.json').read_text())
    args = metrics['args']
    if is_detail_run(metrics):
        model = model_from_args(args)
    else:
        backbone = build_backbone(args['backbone'], pretrained=False)
        if args['adapt'] != 'full':
            add_adapters(backbone, args['adapt'])
        model = add_class_guidance(MultiTaskModel(backbone, task_routing=args['adapt'] == 'moelora'), 'none')
    model.load_state_dict(torch.load(Path(run_dir) / 'model.pt', map_location='cpu'))
    if fusion is not None:
        model.roi_heads.fusion = fusion
    return model.cuda().eval(), metrics


def evaluate_fusions(model, loader, gt_json, fusions=FUSIONS):
    """The holdout metrics under every detection scoring (ablation A2); the model keeps its own scoring after."""
    own, results = model.roi_heads.fusion, {}
    for fusion in fusions:
        model.roi_heads.fusion = fusion
        metrics, _ = reference_train.evaluate(model, loader, gt_json)
        results[fusion] = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    model.roi_heads.fusion = own
    return results


@torch.no_grad()
def time_inference(model, loader, batches=20, batch1_images=100):
    """Forward-pass time per image in ms (bf16, data loading excluded): at the loader's batch size over its first
    `batches` batches, and at batch size 1 over the first `batch1_images` images. One warm-up pass is not timed."""
    model.eval()
    images = [batch[0] for batch in itertools.islice(loader, batches)]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        model(images[0].cuda())  # warm-up
        torch.cuda.synchronize()
        start = time.time()
        for batch in images:
            model(batch.cuda())
        torch.cuda.synchronize()
        batched = 1000 * (time.time() - start) / sum(len(b) for b in images)
        singles = torch.cat(images)[:batch1_images]
        start = time.time()
        for image in singles:
            model(image[None].cuda())
        torch.cuda.synchronize()
        single = 1000 * (time.time() - start) / len(singles)
    return {f'batch{loader.batch_size}': batched, 'batch1': single}
