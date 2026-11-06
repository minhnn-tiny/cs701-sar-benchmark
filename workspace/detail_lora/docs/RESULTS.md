# DetailLoRA: holdout results (24 Oct 2026)

**Setup.**
- Local holdout `dataset/holdout/val`: 939 images, near-duplicate-aware ([DATA.md](DATA.md) §5–6). Training on
  the other 8,453 train images.
- 3 seeds (0, 1, 2) per configuration, the reference protocol (AdamW, batch 16, 24 epochs, 512 px), last-epoch model.
- One NVIDIA L40S per run.
- Δm is per seed against the mean of the 3 local ViT + LoRA runs (macro-F1 91.47, mAP 21.98), then mean ± sample std.

Reproduce with `uv run python workspace/detail_lora/compare.py workspace/detail_lora/runs` (the smoke run shows as
its own group).

## Best configuration: TerraMind + MoE-LoRA + detail branch + small anchors (follow-up, 24 Oct evening)

The TerraMind + MoE-LoRA baseline unchanged, plus the detail branch and small anchors (README step 4b). It is a
single-factor comparison: same adapters, augmentation (flips), scoring and protocol.

| config | total (M) | trainable (M) | macro-F1 | mAP | AP50 | Δm | train h | peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ViT + LoRA (Δm reference) | 107.04 | 21.00 | 91.5 ± 0.3 | 22.0 ± 0.7 | 48.2 ± 1.0 | 0.0 ± 1.6 | 1.00 | 7.4 |
| TerraMind + MoE-LoRA | 106.49 | 21.15 | 90.6 ± 0.5 | 24.0 ± 0.3 | 51.8 ± 0.6 | +4.0 ± 0.7 | 1.51 | 14.2 |
| DetailLoRA (ViT) | 111.22 | 26.20 | 94.2 ± 0.5 | 23.1 ± 0.3 | 51.0 ± 0.4 | +4.0 ± 1.0 | 1.30 | 9.9 |
| **TerraMind + MoE-LoRA + detail + anchors** | **106.84** | **21.49** | 90.1 ± 0.7 | **24.9 ± 0.4** | **53.5 ± 0.5** | **+5.8 ± 1.1** | 1.67* | 15.0 |

\* Seeds 1 and 2 (34.6 images/s). Seed 0 took 3.00 h: another user's job joined its GPU about 40 min after it
started (15 images/s). The scores are unaffected.

- **Per seed**: Δm +6.86 / +5.96 / +4.66 (baseline +3.59 / +3.62 / +4.86); mAP 25.30 / 24.76 / 24.51 (baseline
  23.69 / 23.92 / 24.29).
- **The detail branch adds +0.9 mAP and +1.7 AP50** on TerraMind (half its +1.8 on ViT).
  - **Classes**: wind turbine +2.5 AP (14.7 → 17.2), oil tank +2.0, aircraft +1.2, ship +1.1.
  - **Other metrics**: AP75 +0.9. F1 −0.4, within seed noise.
- **Net +1.8 Δm** over the best in-budget baseline, for 343 k more trainable parameters and ~10 % more training
  time. The gain is below ~2, so report it as mean ± std. The mAP gain itself holds on every seed.
- **A2 scoring**: `soft` +5.9 ± 1.2, `none` +5.8 ± 1.1, `hard` +4.8 ± 1.3. The same null result as before.
- **Inference**: 17.8 ms/image at batch 16, 39.2 ms at batch 1 (the backbone runs once per task).

## First comparison (DetailLoRA vs the baselines)

| config | total (M) | trainable (M) | accuracy | macro-F1 | mAP | AP50 | Δm | train h | peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ViT + LoRA (Δm reference) | 107.04 | 21.00 | 90.5 ± 0.2 | 91.5 ± 0.3 | 22.0 ± 0.7 | 48.2 ± 1.0 | 0.0 ± 1.6 | 1.00 | 7.4 |
| TerraMind + MoE-LoRA (best in-budget baseline) | 106.49 | 21.15 | 89.8 ± 0.6 | 90.6 ± 0.5 | **24.0 ± 0.3** | **51.8 ± 0.6** | +4.0 ± 0.7 | 1.51 | 14.2 |
| **DetailLoRA (primary)** | 111.22 | 26.20 | **93.5 ± 0.4** | **94.2 ± 0.5** | 23.1 ± 0.3 | 51.0 ± 0.4 | **+4.0 ± 1.0** | 1.30 | 9.9 |
| A1: DetailLoRA − detail branch − small anchors | 110.87 | 25.86 | 93.3 ± 0.6 | 94.1 ± 0.6 | 21.3 ± 0.0 | 47.5 ± 0.5 | 0.0 ± 0.4 | 1.19 | 9.0 |

Per seed, Δm: DetailLoRA +5.14 / +3.07 / +3.84; TerraMind + MoE-LoRA +3.59 / +3.62 / +4.86; ViT + LoRA +1.86 /
−0.63 / −1.22; A1 +0.36 / −0.38 / −0.05.

**Reading.**
- **DetailLoRA beats the reference clearly** (+4.0 ± 1.0): macro-F1 +3.0 % relative, mAP +5.0 % relative.
- **It ties the best in-budget baseline, TerraMind + MoE-LoRA, on Δm (+4.0 each), with the opposite profile.**
  DetailLoRA is better at classification (F1 +3.6 points), TerraMind + MoE-LoRA at detection (mAP +0.9).
- **It does not beat that baseline on the holdout.**
- **It gets there with less compute.** One backbone pass instead of two, 14 % less training time and 30 % less
  peak memory, but 5.0 M more trainable parameters.
- **The design proposal estimated Δm +6 to +12**; the measured +4.0 is below it, because the detection side
  under-delivered (see A1).

## Ablations (holdout, 3 seeds)

**A1: detail branch + small anchors** (DetailLoRA vs A1, everything else equal):

| | macro-F1 | mAP | AP50 | AP75 |
|---|---:|---:|---:|---:|
| A1 (without) | 94.1 | 21.3 | 47.5 | 15.8 |
| DetailLoRA (with) | 94.2 | 23.1 | 51.0 | 17.5 |
| change | +0.1 | **+1.8** | **+3.5** | +1.7 |

The gain is detection only, and it comes from the small-object classes: wind turbine 8.2 → 13.4 AP, oil tank
21.7 → 24.7, car 48.2 → 51.0, aircraft 25.9 → 28.5, harbor 18.2 → 20.6. It is consistent across seeds (A1's mAP:
21.39 / 21.33 / 21.29; DetailLoRA's: 23.44 / 22.77 / 23.04). This is the design's clearest positive result.

**Dense LoRA + unfrozen patch/position embeddings and norms + SAR augmentation** (A1 vs ViT + LoRA). This is not a
single-factor ablation: A1 differs from the reference by these three changes together.
- **Classification**: macro-F1 +2.6 points (91.5 → 94.1).
- **Detection**: mAP −0.7 (22.0 → 21.3), AP50 −0.7.
- **Net**: these changes help classification only. Which of the three costs detection is not isolated (see Next
  steps).

**A2: detection scoring** (same checkpoints re-scored, Δm against the unchanged reference):

| config | none (torchvision) | hard (image class only) | soft (product of the two class distributions) |
|---|---:|---:|---:|
| ViT + LoRA | 0.0 ± 1.6 | −0.2 ± 1.7 | 0.0 ± 1.7 |
| TerraMind + MoE-LoRA | +4.0 ± 0.7 | +3.0 ± 0.7 | +4.1 ± 0.8 |
| A1 | 0.0 ± 0.4 | −0.1 ± 0.4 | 0.0 ± 0.4 |
| DetailLoRA | +4.0 ± 1.0 | +3.9 ± 1.1 | +4.0 ± 1.0 |

- **Null result.** `soft` equals `none` within 0.1 Δm: the RoI head's box classes already agree with the image
  class, so the classifier adds nothing to the scores.
- **`hard` hurts** where the classifier is weaker (TerraMind + MoE-LoRA, F1 90.6: mAP −0.5): it moves every box of
  a misclassified image to the wrong class.
- **Keep `soft`** (the default) or `none`; they are equivalent. Report the idea as tested and not useful.

## Per-class AP (%, mean over 3 seeds)

| config | aircraft | airport | bridge | car | harbor | oil_tank | playground | ship | wind_turbine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ViT + LoRA | 28.2 | 29.8 | 0.5 | 48.6 | 21.1 | 23.4 | 0.4 | 34.4 | 11.4 |
| TerraMind + MoE-LoRA | 31.2 | 28.6 | 1.0 | 52.7 | 21.0 | 28.3 | 1.7 | 36.6 | 14.7 |
| DetailLoRA | 28.5 | 31.9 | 1.1 | 51.0 | 20.6 | 24.7 | 0.9 | 35.7 | 13.4 |
| A1 | 25.9 | 32.8 | 1.0 | 48.2 | 18.2 | 21.7 | 0.4 | 35.6 | 8.2 |

**Bridge and playground are near 0 AP for every model**, including the reference scored on its own training images
(about 1.5 % and 2 %, `workspace/baseline/runs/repro`). Their boxes mark 8–16 px crossing points and urban
patches that are barely visible at this resolution. They cap the reachable mAP at about 7/9 of the per-class
average for every method.

## Compute

- 15 holdout runs: **21.4 GPU-h** of training (`train_hours`; seed 0 of the TerraMind follow-up is inflated by
  1.3 h of GPU sharing).
- Smoke test: < 0.01 h.
- The ViT + LoRA full-train reproduction (`workspace/baseline/runs/repro`, 3 seeds): a further 3.46 h.
- DetailLoRA inference: 9.2 ms/image at batch 16 and 19.0 ms at batch 1 (forward only, bf16).

## Next steps (proposed, not run)

1. **Detail branch + small anchors on TerraMind + MoE-LoRA.** Done: Δm +5.8 ± 1.1, the best configuration (top of
   this file).
   - **Why**: the detail branch is the one component with a consistent detection gain (+1.8 mAP), and TerraMind +
     MoE-LoRA has the best mAP.
   - **Expected**: if the +1.8 mAP carries over at TerraMind's F1, Δm ≈ +7 to +8. This is an estimate, and A1 shows
     components don't simply add.
   - **Cost**: a task-routed variant of `DetailMultiTaskModel`, then 3 seeds × ~1.6 h (≈ 2.5 h on GPUs 1–2).
2. **Find what costs mAP in A1**: dense LoRA, the unfrozen first layers, or the augmentation. One `--no-augment` run
   (1.2 h) answers the cheapest part.
3. **Phase 1 (closes 1 Nov)**: `predict.py` on `runs/seed0/detaillora --split val` gives a leaderboard reading now.
   The full-train run (`jobs/final.txt`, ~1.45 h) is for the final submission, once the design is fixed.
