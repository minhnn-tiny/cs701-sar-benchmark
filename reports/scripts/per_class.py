"""Per-class AP of the final recipe on the three holdout views, under the final decoding: ND (near-duplicate mix like
val), LF (leak-free) and FIT (local-train images). Mean over the three holdout seeds of YSP350 (SAR 0.8, 350 epochs).
Writes reports/scripts/per_class.json. Run from the repo root through workspace/gpu_run.sh:
  workspace/gpu_run.sh --min-free 8000 -- uv run python reports/scripts/per_class.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, 'workspace/mtdet')
from common import ROOT, eval_dirs, evaluate, make_loader  # noqa: E402
from evaluate import load_run  # noqa: E402

DECODING = 'o2m_nms0.8_a0.5_k100_t0_f1'
out = {}
for seed in (0, 1, 2):
    model, args = load_run(Path(f'workspace/mtdet/runs/YSP350_s{seed}'))
    model.apply_decoding(model.decodings()[DECODING])
    for name, d in eval_dirs(ROOT / args['data'], ['nd', 'lf', 'fit']).items():
        metrics, _ = evaluate(model, make_loader(d.parent, d.name, batch_size=16), d / 'instances.json')
        out.setdefault(name, []).append({k: metrics[k] for k in ('macro_f1', 'mAP', 'AP50', 'per_class_AP', 'delta_m')})
        print(seed, name, round(100 * metrics['macro_f1'], 2), round(100 * metrics['mAP'], 2), flush=True)
Path('reports/scripts/per_class.json').write_text(json.dumps(out, indent=1))
