"""The post-processing sweep: score a trained R50-FPN-MT run on the local holdout under every decoding of DECODINGS
(r50fpn_model.py), on the whole holdout and on each of its two halves (halves.json), and write <run>/decode_eval.json.
Δm is against the local reference: ViT + LoRA on the same holdout (workspace/sarmtl/runs/ref_s*), mean of its seeds.
Example (from the repo root, through the GPU runner):
  workspace/gpu_run.sh --min-free 8000 -- uv run python workspace/r50fpn/evaluate.py <run_dir>"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # the repo root, for sarbench/
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / 'workspace' / 'sarmtl')]
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from r50fpn_model import DECODINGS, model_from_args, set_decoding  # noqa: E402
from sarbench.data import SARMultiTask, collate  # noqa: E402

# workspace/sarmtl/evaluate.py (its evaluation loop and metrics) under another name: this file is also evaluate.py.
# It puts workspace/sarmtl first on sys.path, which would shadow this design's modules, so the path is restored.
_spec = importlib.util.spec_from_file_location('sarmtl_evaluate', ROOT / 'workspace' / 'sarmtl' / 'evaluate.py')
_sarmtl, _path = importlib.util.module_from_spec(_spec), list(sys.path)
_spec.loader.exec_module(_sarmtl)
sys.path[:] = _path
predict, scores = _sarmtl.predict, _sarmtl.scores

REFERENCE_RUNS = ROOT / 'workspace' / 'sarmtl' / 'runs'


def split_images(data, split):
    """The image files of a split's labels.csv, resolved: the derived datasets (holdout_sarmtl, repro) link their
    images/ to dataset/train/images, so the same image has the same path in all of them."""
    root = (ROOT / data) / split  # a run's args store data relative to the repo root (or absolute)
    with open(root / 'labels.csv') as f:
        return {(root / row['file_name']).resolve() for row in csv.DictReader(f)}


def check_disjoint(run_args, data):
    """Refuse a run that trained on any image of the evaluated split (data/val): its scores would not be held out."""
    overlap = split_images(run_args['data'], 'train') & split_images(data, 'val')
    if overlap:
        raise SystemExit(f"refused: the run trained on {run_args['data']}, which shares {len(overlap)} images with "
                         f'{data}/val (e.g. {sorted(overlap)[0].name}); its holdout scores would not be held out')


def local_reference(data, runs=REFERENCE_RUNS):
    """(macro-F1, mAP) of the local Δm reference: the mean of the ViT + LoRA runs scored on the same split, reference
    decoding."""
    ms = [json.loads(p.read_text()) for p in sorted(runs.glob('ref_s*/vit_pretrained_lora/metrics.json'))]
    assert len(ms) == 3, f'expected the 3 reference seeds under {runs}'
    for m in ms:
        assert (ROOT / m['args']['data']).resolve() == (ROOT / data).resolve(), \
            f"a reference run was scored on {m['args']['data']}, not {data}"
    return sum(m['val']['macro_f1'] for m in ms) / 3, sum(m['val']['mAP'] for m in ms) / 3


def delta_m(f1, mAP, ref):
    """Δm in %: the mean relative change of macro-F1 and mAP against ref = (F1, mAP)."""
    return 100 * ((f1 - ref[0]) / ref[0] + (mAP - ref[1]) / ref[1]) / 2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path, help='a run directory (metrics.json, model.pt)')
    p.add_argument('--data', type=Path, default=ROOT / 'dataset' / 'holdout_sarmtl')
    p.add_argument('--decodings', nargs='+', default=list(DECODINGS), choices=list(DECODINGS))
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()

    run_args = json.loads((args.run / 'metrics.json').read_text())['args']
    check_disjoint(run_args, args.data)
    ref = local_reference(args.data)
    model = model_from_args(run_args)
    model.load_state_dict(torch.load(args.run / 'model.pt', map_location='cpu'))
    model.cuda().eval()
    loader = DataLoader(SARMultiTask(args.data, 'val'), args.batch_size, num_workers=args.workers, collate_fn=collate)
    gt_json = args.data / 'val' / 'instances.json'
    halves = {k: set(v) for k, v in json.loads((args.data / 'halves.json').read_text()).items()}

    result = {'run': str(args.run), 'data': str(args.data), 'reference_f1_mAP': ref, 'decodings': {}}
    for name in args.decodings:
        set_decoding(model, **DECODINGS[name])
        predictions = predict(model, loader)
        r = {'all': scores(gt_json, *predictions),
             **{f'half_{k}': scores(gt_json, *predictions, subset=v) for k, v in halves.items()}}
        for split in r.values():
            split['delta_m'] = delta_m(split['macro_f1'], split['mAP'], ref)
        result['decodings'][name] = {'settings': DECODINGS[name], **r}
        print(f'{name:22s} ' + '  '.join(f"{split} Δm {s['delta_m']:+6.2f} F1 {100 * s['macro_f1']:.2f} "
                                         f"mAP {100 * s['mAP']:.2f}" for split, s in r.items()), flush=True)
    (args.run / 'decode_eval.json').write_text(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
