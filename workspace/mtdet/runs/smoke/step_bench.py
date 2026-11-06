"""Training-step throughput of YOLO26-L as train.py runs it (channels_last, bf16, its AdamW groups, clip 10, Ultralytics'
ModelEMA, an isfinite check of the loss every step), against variants with the same math: no per-step check (a host
sync), a foreach EMA update, fused AdamW. The same 100 preloaded batches for each; steps 21-100 timed.
Run: workspace/gpu_run.sh -- .venv/bin/python workspace/mtdet/runs/smoke/step_bench.py"""
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, 'workspace/mtdet')
from common import HOLDOUT, make_loader  # noqa: E402
from train import build_model, param_groups, parse_args  # noqa: E402
from ultralytics.utils.torch_utils import ModelEMA  # noqa: E402

torch.backends.cudnn.benchmark = True
args = parse_args(['--arch', 'yolo26l', '--out', '/tmp/unused'])
batches = [(x.cuda(), y.cuda(), [{'boxes': t['boxes'].cuda(), 'labels': t['labels'].cuda()} for t in ts])
           for x, y, ts in make_loader(HOLDOUT, 'train', train=True, batch_size=16, workers=8, limit=1600, seed=0)]


def float_tensors(module):
    return [v for v in module.state_dict().values() if v.dtype.is_floating_point]


def foreach_update(ema, pairs):
    """ModelEMA.update in two foreach calls: ema = d * ema + (1 - d) * model, over the same tensors."""
    ema.updates += 1
    torch._foreach_lerp_(pairs[0], pairs[1], 1 - ema.decay(ema.updates))


def model():
    torch.manual_seed(0)
    m = build_model(args).cuda().to(memory_format=torch.channels_last)
    m.channels_last = True
    return m.train()


def bench(name, sync=True, foreach_ema=False, fused=False):
    m = model()
    opt = torch.optim.AdamW(param_groups(m, args), lr=args.lr, fused=fused or None)
    ema = ModelEMA(m, decay=args.ema)
    pairs = (float_tensors(ema.ema), [v.detach() for v in float_tensors(m)])
    for i, (x, y, t) in enumerate(batches):
        if i == 20:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _, losses = m(x, t)
        loss = F.cross_entropy(logits.float(), y) + sum(losses.values())
        if sync and not torch.isfinite(loss):
            raise FloatingPointError
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), args.clip)
        opt.step()
        foreach_update(ema, pairs) if foreach_ema else ema.update(m)
    torch.cuda.synchronize()
    ms = 1000 * (time.time() - t0) / (len(batches) - 20)
    print(f'{name:38s} {ms:6.1f} ms/step  {16000 / ms:6.1f} images/s  loss {loss.item():.4f}', flush=True)
    return m, ema


# the EMA math: one update of each kind from the same state
m, ema = bench('train.py (sync, ModelEMA, AdamW foreach)')
twin = ModelEMA(m, decay=args.ema)
twin.ema.load_state_dict(ema.ema.state_dict())
twin.updates = ema.updates
before = [v.clone() for v in float_tensors(ema.ema)]
ema.update(m)
foreach_update(twin, (float_tensors(twin.ema), [v.detach() for v in float_tensors(m)]))
diff = max((a - b).abs().max().item() for a, b in zip(float_tensors(ema.ema), float_tensors(twin.ema)))
moved = max((a - b).abs().max().item() for a, b in zip(float_tensors(ema.ema), before))
print(f'EMA update, ModelEMA vs foreach lerp: max |diff| {diff:.2e} (one update moves weights up to {moved:.2e})')
del m, ema, twin
bench('no per-step sync', sync=False)
bench('no sync + foreach EMA', sync=False, foreach_ema=True)
bench('no sync + foreach EMA + fused AdamW', sync=False, foreach_ema=True, fused=True)
bench('train.py again (drift check)')
print(f'peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB')
