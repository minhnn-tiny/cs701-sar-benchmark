"""Score a trained run (train.py's or root train.py's) on the local holdout under the reference decoding, JD (the
design) and JD-posterior (the cross-fit alternative), on the whole holdout and on each of its two halves
(halves.json), and write <run>/jd_eval.json. compare.py reads it for the JD keep rule (DESIGN.md §4, rule 3).
Example (from the repo root, through the GPU runner):
  workspace/gpu_run.sh --min-free 8000 -- uv run python workspace/sarmtl/evaluate.py <run_dir>"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # the repo root, for sarbench/
sys.path[:0] = [str(ROOT), str(HERE)]
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from sarbench.data import SARMultiTask, collate, to_original_xywh  # noqa: E402
from sarbench.metrics import classification_metrics, detection_metrics  # noqa: E402
from sarmtl_model import JD_DECODING, REFERENCE_DECODING, model_from_args, set_decoding  # noqa: E402

DECODINGS = {'reference': REFERENCE_DECODING, 'jd': JD_DECODING, 'jd_posterior': dict(JD_DECODING, mode='posterior')}


@torch.no_grad()
def predict(model, loader):
    """The loop of root train.py's evaluate, keeping the per-image predictions: (labels, predicted, detections, ids)."""
    labels, predicted, detections, image_ids = [], [], [], []
    for images, y, targets in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, outputs, _ = model(images.cuda())
        labels += y.tolist()
        predicted += logits.argmax(dim=1).tolist()
        for target, out in zip(targets, outputs):
            image_ids.append(target['image_id'])
            boxes = to_original_xywh(out['boxes'], target['orig_size']).tolist()
            detections += [{'image_id': target['image_id'], 'category_id': c, 'bbox': b, 'score': s}
                           for b, c, s in zip(boxes, out['labels'].tolist(), out['scores'].tolist())]
    return labels, predicted, detections, image_ids


def scores(gt_json, labels, predicted, detections, image_ids, subset=None):
    """Classification and COCO metrics, on all images or on the image ids in subset."""
    keep = [i for i, image_id in enumerate(image_ids) if subset is None or image_id in subset]
    ids = [image_ids[i] for i in keep]
    dets = detections if subset is None else [d for d in detections if d['image_id'] in subset]
    return {**classification_metrics([labels[i] for i in keep], [predicted[i] for i in keep]),
            **detection_metrics(gt_json, dets, ids)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path, help='a run directory (metrics.json, model.pt)')
    p.add_argument('--data', type=Path, default=ROOT / 'dataset' / 'holdout_sarmtl')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()

    model = model_from_args(json.loads((args.run / 'metrics.json').read_text())['args'])
    model.load_state_dict(torch.load(args.run / 'model.pt', map_location='cpu'))
    model.cuda().eval()
    loader = DataLoader(SARMultiTask(args.data, 'val'), args.batch_size, num_workers=args.workers, collate_fn=collate)
    gt_json = args.data / 'val' / 'instances.json'
    halves = {k: set(v) for k, v in json.loads((args.data / 'halves.json').read_text()).items()}

    result = {'run': str(args.run), 'data': str(args.data), 'decodings': DECODINGS}
    for name, decoding in DECODINGS.items():
        set_decoding(model, **decoding)
        start = time.time()
        predictions = predict(model, loader)
        result[name] = {'all': scores(gt_json, *predictions),
                        **{f'half_{k}': scores(gt_json, *predictions, subset=v) for k, v in halves.items()},
                        'ms_per_image': 1000 * (time.time() - start) / len(loader.dataset)}
        print(f"{name:13s} " + ' '.join(f"{split} F1 {r['macro_f1']:.4f} mAP {r['mAP']:.4f}"
                                        for split, r in result[name].items() if split != 'ms_per_image'))
    (args.run / 'jd_eval.json').write_text(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
