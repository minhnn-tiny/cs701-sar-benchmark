"""Training-step time of YOLO26-L, eager vs torch.compile of each trunk layer (the Detect head and loss stay eager),
same batches, channels_last + bf16. Also checks the first losses agree."""
import sys
import time

import torch

sys.path.insert(0, 'workspace/mtdet')
from common import HOLDOUT, make_loader  # noqa: E402
from ultra_model import UltraMultiTask  # noqa: E402

torch.backends.cudnn.benchmark = True
loader = make_loader(HOLDOUT, 'train', train=True, batch_size=16, workers=8, limit=1280, seed=0)


def bench(compiled):
    torch.manual_seed(0)
    m = UltraMultiTask('yolo26l', epochs=10).cuda().to(memory_format=torch.channels_last)
    m.channels_last = True
    if compiled:
        for layer in m.det.model[:-1]:
            layer.compile()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    m.train()
    times, first = [], None
    start = time.time()
    for x, y, t in loader:
        x, y = x.cuda(), y.cuda()
        t = [{'boxes': a['boxes'].cuda(), 'labels': a['labels'].cuda()} for a in t]
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _, losses = m(x, t)
        loss = torch.nn.functional.cross_entropy(logits.float(), y) + sum(losses.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        first = first if first is not None else loss.item()
    steady = times[20:]
    print(f"{'compiled' if compiled else 'eager   '}: {1000 * sum(steady) / len(steady):.0f} ms/step (steps 20-{len(times)}), "
          f"first loss {first:.4f}, total {time.time() - start:.0f} s incl. compile", flush=True)


bench(False)
bench(True)
