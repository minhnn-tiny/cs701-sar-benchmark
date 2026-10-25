"""Optional Weights & Biases tracking for the workspace train scripts (detail_lora, r50fpn, sarmtl).

On for runs started while WANDB_API_KEY is set: in <repo>/.env, next to WANDB_ENTITY and WANDB_PROJECT, or in the
environment. WANDB_MODE=offline logs to <run dir>/wandb/ without a key (upload later with `wandb sync`), and
WANDB_MODE=disabled turns it off for one run. While it is off, every function here returns at once and wandb isn't
imported, so a run is exactly what it was before.

While it is on, it only watches. It mirrors root train.py's log lines (the losses and learning rate every 50
iterations, each validation) and, at the end, the run's metrics.json. It never touches the model, the data or the
random state, and if wandb fails it turns itself off instead of stopping the training.
Check the set-up, from the repo root: `uv run python workspace/wandb_log.py` (see the end of this file).
"""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the repo root, for .env
# root train.py's progress line (train_one_epoch) and validation line (main, which the designs copy)
ITERATION = re.compile(r'epoch (\d+) iter (\d+)/(\d+) loss (\S+) \((.*)\) lr (\S+) (\S+) images/s')
VALIDATION = re.compile(r'epoch (\d+) val: (.*)')
_run, _step = None, 0  # the W&B run, and the global iteration of the last progress line


def _load_env(path=ROOT / '.env'):
    """Read only the WANDB_* lines of .env (it also holds the team token); a value in the environment wins."""
    for line in path.read_text().splitlines() if path.is_file() else []:
        key, _, value = line.strip().removeprefix('export ').partition('=')
        if key.startswith('WANDB_') and not os.environ.get(key):
            os.environ[key] = value.strip().strip('\'"')
    for key in [k for k, v in os.environ.items() if k.startswith('WANDB_') and not v]:
        del os.environ[key]  # an empty placeholder means unset


def init(design, run_dir, config, train_module):
    """If tracking is on, start a W&B run for run_dir (<out>/<config>, grouped by design) and mirror
    train_module.log, root train.py as the train script loaded it, to it. Each call starts a new W&B run, so a resumed
    training continues in a new one."""
    global _run
    try:
        _load_env()
        mode = os.environ.get('WANDB_MODE')
        if mode == 'disabled' or not (os.environ.get('WANDB_API_KEY') or mode in ('online', 'offline')):
            return
        import wandb
        run_dir = Path(run_dir).resolve()
        # no code upload; console capture off (stdout already goes to the run's .out file); and a run's end waits at
        # most 5 min for the upload (then <run dir>/wandb/ keeps the data for `wandb sync`), so a queue never stalls
        _run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'cs701-sar'), group=design,
                          name=f'{run_dir.parent.name}/{run_dir.name}', dir=run_dir, config=_plain(config),
                          save_code=False, settings=wandb.Settings(console='off', finish_timeout=300))
    except Exception as error:  # tracking must never stop a training run
        print(f'W&B: logging off ({error!r})', flush=True)
        return
    log = train_module.log

    def mirrored(message, *args, **kwargs):
        log(message, *args, **kwargs)
        _mirror(message)
    train_module.log = mirrored  # root train.py's functions look log up at each call, so they now call this


def _mirror(message):
    """Log a root train.py progress or validation line to W&B, at the iteration it was printed."""
    global _run, _step
    if _run is None:
        return
    try:
        if parsed := _parse(message):
            step, data = parsed
            if step is not None:
                _step = step
            _run.log(data, step=_step)  # W&B commits a step's row when a later step arrives, or at finish
    except Exception as error:  # e.g. the wandb service died: train on without it
        print(f'W&B: logging off ({error!r})', flush=True)
        _run = None


def _parse(message):
    """root train.py's progress line -> (global iteration, data); its validation line -> (None, data); else None."""
    if m := ITERATION.fullmatch(message):
        epoch, step, steps = int(m[1]), int(m[2]), int(m[3])
        return (epoch - 1) * steps + step, {'epoch': epoch, 'train/loss': float(m[4]), **_pairs('train/', m[5]),
                                            'train/lr': float(m[6]), 'train/images_per_s': float(m[7])}
    if m := VALIDATION.fullmatch(message):
        return None, {'epoch': int(m[1]), **_pairs('val/', m[2])}


def _pairs(prefix, text):
    """'name value name value ...' -> {prefix + name: value}"""
    words = text.split()
    return {prefix + name: float(value) for name, value in zip(words[::2], words[1::2])}


def _plain(value):
    """JSON types only (paths become strings), as in metrics.json."""
    return json.loads(json.dumps(value, default=str))


def finish(summary):
    """Add the run's metrics.json (final scores, parameter counts, costs) to the W&B summary and close the run."""
    global _run
    if _run is None:
        return
    try:
        _run.summary.update(_plain(summary))
        _run.finish()
    except Exception as error:
        print(f'W&B: could not finish the run ({error!r})', flush=True)
    _run = None


if __name__ == '__main__':
    # Self-check with the .env settings: two root train.py lines go to a run named check/wandb_log (group "check"),
    # which you can delete on the site. WANDB_MODE=offline runs it locally, without a key.
    import tempfile
    import types
    lines = ['epoch 2 iter 50/504 loss 3.696 (classification 1.664 loss_objectness 0.688 loss_rpn_box_reg 0.005 '
             'loss_classifier 1.334 loss_box_reg 0.005) lr 1.01e-05 37.9 images/s',
             'epoch 2 val: accuracy 0.7500 macro_f1 0.7602 balanced_accuracy 0.7643 mAP 0.1542 AP50 0.3440']
    assert _parse(lines[0]) == (554, {'epoch': 2, 'train/loss': 3.696, 'train/classification': 1.664,
                                      'train/loss_objectness': 0.688, 'train/loss_rpn_box_reg': 0.005,
                                      'train/loss_classifier': 1.334, 'train/loss_box_reg': 0.005,
                                      'train/lr': 1.01e-05, 'train/images_per_s': 37.9}), _parse(lines[0])
    assert _parse(lines[1]) == (None, {'epoch': 2, 'val/accuracy': 0.75, 'val/macro_f1': 0.7602,
                                       'val/balanced_accuracy': 0.7643, 'val/mAP': 0.1542, 'val/AP50': 0.344})
    assert _parse('budget OK: 1 <= 2 total') is None and _parse('test: accuracy 0.9\ntraining 1 h') is None
    printed = []
    train = types.SimpleNamespace(log=lambda message, file: printed.append(message))
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp, 'check', 'wandb_log')
        run_dir.mkdir(parents=True)
        init('check', run_dir, {'check': True}, train)
        assert _run is not None, 'tracking is off: fill in WANDB_API_KEY in .env (or set WANDB_MODE=offline)'
        url = _run.url
        for line in lines:
            train.log(line, None)
        assert printed == lines, printed  # the training log itself is unchanged
        assert _run is not None and _step == 554, 'logging failed: see the W&B message above'
        finish({'check': 'ok'})
    print(f'W&B check passed: {url or "offline run (deleted with its temporary directory)"}')
