# cv-adapt: candidates for backbone adaptation and multi-task learning

Author: `cv-adapt`. Status: **v6 (final for this step)**. `critic` accepted it in round 2, with conditions C1–C4
(applied here). Counts follow `data-eng`'s final leak-free split (DATA.md and `split.json`, final sweep; the holdout is frozen).
Scripts and their outputs: `workspace/sarmtl/scratch/cv-adapt/`.
- Measurement scripts, each with its `.out`: `count_params.py`, `flops.py`, `step_flops.py`, `activation_memory.py`,
  `task_fork.py`.
- Modules: `wide_lora.py` (WideLoRA), and `fusion.py` (D1b reference code with a CPU self-test).
- All ran on CPU or meta tensors.

## Summary

| candidate | what changes | total / trainable (= heads + backbone adaptation) | time per holdout run (reference ≈ 1.24 h) | expected Δm vs local reference |
|---|---|---|---|---|
| **WideLoRA** | LoRA r64, rank-stable scale γ = 1, on every linear layer of all 12 blocks (qkv, proj, fc1, fc2), plus patch/pos embeddings, LayerNorms and biases trainable; still one backbone pass | 115,591,705 / 30,657,049 (= 20,113,177 + 10,543,872) | 1.30–1.43 h (1.05–1.15×), ~16 GB; a GPU smoke test comes first | prior +2 to +8 (central ≈ +5). With 3 seeds, the smallest effect we can detect is ≈ 2.0–2.5 |

- **Decision rules, stated before any result (§0, §1).** Adopt WideLoRA if its 3-seed mean holdout Δm beats the local
  reference's (> 0). The submission backbone is whichever is higher on the 3-seed mean: the design or TerraMind +
  MoE-LoRA. If TerraMind wins, TerraMind + WideLoRA follows. A difference below the minimum detectable effect is
  reported as inconclusive.
  - Each model is scored **as it would be submitted**: with its own pre-registered D1 decoding (product α = 1 if it
    passes D1's per-model adopt rule, else α = 0, score_thresh 0.001).
  - Both models' D1-off numbers are reported alongside, as the same-decoding comparison that attributes the
    adaptation's effect (critic C2).
- **One candidate.** A second slot is not filled for its own sake. The alternatives I costed are listed in §3 as
  ablation options, with their numbers: TaskFork (task-specific top blocks), WideLoRA r32, TerraMind + WideLoRA.
  None is a bet I'd make over WideLoRA.
- **Cross-task decoding** (my draft-v1 XFuse) is now `cv-detect`'s **D1/D1b** in `cand_detect.md`.
  - D1's a-priori rule is XFuse's product s_c·p_c, which never raises a score.
  - The det→cls step is D1b.
  - It adds 0 parameters and stacks on WideLoRA.
- **Backbone:** ViT-B/16 ImageNet-21k (`vit_base_patch16_224.augreg_in21k`); TerraMind is the measured fallback (§0).
- **The 40 M cap is shared.** WideLoRA + `cv-detect` D2 = 31,021,817 trainable / 115,956,473 total, or 31,975,033
  trainable with D2-wide. The rank is the knob: trainable = 21,219,865 + 147,456·r for WideLoRA alone.
- **Loss balancing** (Kendall, GradNorm, PCGrad) is **not proposed** (§2). By epoch 13 of 24, train cross-entropy is
  ~7× smaller than the detection losses: 0.026 vs 0.183 (repro log). Uncertainty weighting would then up-weight the
  task that's already fitted.

## 0. Backbone choice: ViT (ImageNet-21k)

Δm from CLAUDE.md, test / val:

| adaptation | ViT | TerraMind | TerraMind − ViT | where TerraMind's difference comes from (test) |
|---|---:|---:|---:|---|
| full fine-tuning (most capacity) | +13.5 / +11.8 | +11.5 / +10.7 | **−2.0 / −1.1** | F1 −0.2, mAP −1.0 |
| MoE-LoRA | +2.5 / +1.3 | +3.8 / +4.4 | +1.3 / +3.1 | F1 −1.0, mAP +1.0 |
| LoRA (least capacity) | 0.0 / 0.0 | +1.4 / +1.0 | +1.4 / +1.0 | F1 −1.4, mAP +1.1 |
| from scratch (no pretraining) | −22.5 / −20.3 | −19.3 / −17.3 | +3.2 / +3.0 | architecture only |

- **Capacity reverses the lead.** TerraMind leads only in the low-capacity regime, only in mAP, and its F1 is always
  lower. Once the whole backbone adapts, the lead becomes a 1–2 point deficit. WideLoRA moves from the LoRA regime
  toward full fine-tuning: every layer adapts, and the position and patch embeddings train.
- **Part of TerraMind's edge may be architectural.** Untrained, TerraMind's architecture is already +3 Δm better:
  SwiGLU MLP, bias-free layers, and a fixed 2-D sin-cos position embedding (in terratorch, `pos_emb` is a buffer). So
  its lead under LoRA may come from architecture or positional encoding rather than EO pretraining. In the reference
  LoRA, ViT's learned `pos_embed` is bicubic-resampled from 14×14 to 32×32 and frozen. WideLoRA trains it.
- **ViT fits the input better.** Our images are 8-bit display gray. ViT's 1-channel patch embedding is the sum of its
  RGB filters, i.e. luminance. TerraMind's S1GRD embedding expects dB backscatter (VV, VH) and gets the same gray image
  twice, a gap the reference leaves uncorrected (DETAILS.md).
  - The radiometry is also mixed across sources (DATA.md §0). 24.7 % of images are 256² upsampled ×2. Speckle CV is
    0.11–0.20 in the 512² sources and 0.6–1.2 in the 256² ships and oil tanks.
  - So a single Sentinel-1 GRD prior fits poorly. WideLoRA's trainable patch embedding and LayerNorms can adapt to
    those statistics on either backbone.
- **"Capacity reverses the lead" is an extrapolation** between two endpoints: LoRA/MoE-LoRA versus full fine-tuning.
  WideLoRA sits in between, with 12 % of the backbone parameters, though at every depth. So the backbone is decided by
  measurement, not by this argument.
- **Backbone decision rule, stated now** (critic, round 1 addendum).
  - The bar is the best in-budget reference, TerraMind + MoE-LoRA: val +4.4 ± 0.3, test +3.8 ± 0.7. WideLoRA's prior
    central +5 clears it by less than the minimum detectable effect (§1), so the comparison will probably be
    inconclusive by test.
  - The submission backbone and adaptation is whichever has the **higher 3-seed mean holdout Δm**: the design
    (WideLoRA on ViT, with whatever detection modules the design adopts) or TerraMind + MoE-LoRA. Each is scored with
    its own pre-registered D1 decoding, as submitted. Both models' D1-off numbers are reported alongside.
  - If TerraMind + MoE-LoRA wins, the follow-up run is TerraMind + WideLoRA with the same detection modules (e.g. D2;
    counts in §3).

## 1. WideLoRA: LoRA on every linear layer, plus the small pretrained tensors

**The change** (backbone ViT-B/16 in21k; heads and loss unchanged; code in `scratch/cv-adapt/wide_lora.py`):
- Freeze the backbone. Then wrap **qkv, proj, fc1 and fc2 of all 12 blocks** with `sarbench.adapters.LoRA(rank=64,
  alpha=64)`, the unchanged class. That gives scale **γ = α/r = 1**.
  - **Why γ = 1, not the reference's 2** (critic A2). Under Adam each entry of B moves ~lr per step whatever γ is. So
    the weight-space step γ·ΔB·A grows with the rank: ~√r if directions are incoherent, ~r if coherent. At γ = 2, r64
    would take ~2–4× the reference's steps.
  - The rank-stable rule (rsLoRA, Kalajdzievski 2023) is γ = 8/√r. It reproduces the reference at r16 (γ = 2) and gives
    γ = 1 at r64 and √2 at r32. At r64 this matches the reference's step in the incoherent regime, and is at most 2× in
    the coherent one.
  - So the difference from the reference is placement, rank and the unfrozen tensors, not a hidden lr increase.
    `wide_lora.py` applies the rule by default.
  - Initialisation is the reference's: A is nn.Linear's default, B = 0, so training starts exactly at the pretrained
    model.
- Unfreeze the **small pretrained tensors**, 1,106,688 in total:
  - the patch embedding (197,376);
  - `pos_embed` (787,200);
  - `cls_token`;
  - every LayerNorm (norm1, norm2 and the final norm);
  - the biases of the wrapped linear layers.
- AdamW, weight decay 0.05, **3 param groups**:
  - LoRA A/B at 1e-4 (as the reference);
  - the unfrozen pretrained tensors at 2e-5, the TA's full fine-tuning backbone lr (`run_all.sh`), proven stable on this
    data;
  - heads at 1e-4.
- Everything else is the baseline protocol:
  - batch 16, 24 epochs, 512 px;
  - 1 warmup epoch, then cosine, with the same factor for every group;
  - bf16 autocast with fp32 RPN/RoI;
  - last-epoch model.
- Build: `add_wide_lora(backbone, rank=64)` (α = 64 by the rule) → `MultiTaskModel(backbone, task_routing=False)`
  (unchanged) → `AdamW(param_groups(model), weight_decay=0.05)`. For predict, build the same model and load `model.pt`.
- **Pairing (implementation note).** In our train script:
  - re-seed `torch.manual_seed(seed)` *after* the adapters, right before `MultiTaskModel(...)` builds the heads;
  - give the training DataLoader its own `torch.Generator().manual_seed(seed)`.

  Then any two configurations run through that script with the same seed share the head init, the batch order and the
  flips, and only the adaptation differs. If the local reference hasn't started yet, run it through the same script
  (adapt = reference LoRA) rather than the root `train.py`. GPU nondeterminism remains, so the runs are paired, not
  bit-identical.
  - **Conditions on that script** (critic C1). It must *be* the reference protocol.
    - Import `train_one_epoch`, `evaluate` and `lr_factor` from root `train.py`, and mirror `make_loader`. Only the
      model build, the re-seed, the DataLoader generator and the param groups may differ.
    - At adapt = reference LoRA it must log 107,039,257 total / 20,997,913 trainable.
    - With the re-seed and generator switched off, it must reproduce root `train.py`. Check this once in the GPU smoke
      test: the logged loss at iteration 50 on the same seed must match within bf16 noise.
    - Note that root `train_one_epoch` logs `param_groups[1]`'s lr, which here is the 2e-5 group. That is cosmetic.

**Expected effect (a prior, not a measurement):** mAP +1 to +4 (central +2.5), F1 0 to +1 (central +0.5), **Δm +2 to
+8 (central ≈ +5)**. No in-benchmark run sits between LoRA (0.88 M) and full fine-tuning (86 M), so the position of
WideLoRA on that line is a judgement. The decision below doesn't depend on it.
- **The benchmark itself shows the backbone has to change a lot.** Going from LoRA (0.88 M, attention only, r16) to full
  fine-tuning gains +6.4 mAP and +1.8 F1 on test (+5.9 / +1.1 on val): Δm +13.5, 93 % of it from mAP.
- **The reference leaves three things unadapted:**
  - the MLPs, 2/3 of the block parameters;
  - the patch embedding, the only layer that sees speckle and the 8-bit intensity statistics;
  - the resampled position embedding, which matters for localisation.
- **Literature:**
  - LoRA (Hu et al., ICLR 2022): adapting more weight types beats a higher rank on fewer.
  - QLoRA (Dettmers et al., NeurIPS 2023): LoRA on all linear layers is needed to match full fine-tuning.
  - Biderman et al. (TMLR 2024): under large domain shifts, LoRA's gap to full fine-tuning shrinks with rank and
    placement but does not close. Hence a range, not +13.
  - BitFit (ACL 2022) and SSF (NeurIPS 2022): bias, norm and scale tuning is a strong cheap baseline for ViTs.
  - Surgical fine-tuning (Lee et al., ICLR 2023): the layers that matter depend on the shift, so WideLoRA adapts every
    depth instead of guessing.
- **Calibration of the range.** In-benchmark lower anchor: MoE-LoRA's 1.03 M already gains +1.4 mAP (test). The central
  estimate assumes WideLoRA recovers ~40 % of the full fine-tuning gap; the low end assumes ~15 %.
- **The range is calibrated on the TA's val/test tables, and our holdout may show less.**
  - DATA.md §5 (a hypothesis, not checked): if val/test were drawn at random, ≈ 53 % of their images have a copy,
    revisit or overlapping tile in train; for wind turbine, oil tank and aircraft it is close to 100 %. Part of full fine-tuning's +13.5 would then be fitting train.
  - Our group-aware holdout doesn't reward that, so it may show a smaller WideLoRA gain than val/test would. Read a
    holdout result near the low end as consistent with this estimate, not as a refutation.

**Decision rule and detectability** (critic A1; stated before any WideLoRA result):
- **Minimum detectable effect.** Take 3 seeds per arm, per-seed Δm std σ, and a two-sided t-test with df = 4.
  - At p < .05 a difference must reach 2.27σ (50 % power), or 3.03σ for 80 % power.
  - That is ≈ 2.0 / 2.7 Δm at the TA's σ = 0.9, and ≈ 2.5 / 3.3 at σ = 1.1.
  - Expect a larger σ on our 336-unit holdout: airport rests on 7 scenes and wind turbine on 9 (DATA.md §0.9). Recompute σ from the 3 local reference seeds before reading WideLoRA's
    result. Pairing (above) may shrink the variance of the difference, but I don't count on it.
- **Rule.** Adopt WideLoRA as the primary adaptation, and use it for the submission run, if both hold:
  - its 3-seed mean holdout Δm beats the local reference's 3-seed mean (> 0). Each arm is scored with its own
    pre-registered D1 decoding, as it would be submitted. The D1-off numbers of both arms are reported alongside, as the
    same-decoding attribution of the adaptation effect;
  - no other evaluated in-budget configuration, e.g. TerraMind + MoE-LoRA, has a higher mean.

  The threshold is 0 because the two adaptations cost the same within ~15 %, and the prior favours more adaptation
  (full fine-tuning +13.5).
- **Reporting.** Mean ± std per arm, the 3 per-seed values, and the paired difference with its t-based 95 % CI.
  - A difference below the minimum detectable effect is called **inconclusive** in the report. The adoption then rests
    on the point estimate plus the prior, and the report says so.
  - The leaderboard may reward fitting train more than the holdout does (DATA.md §5), so a holdout difference near 0 is
    not evidence against WideLoRA there.

**Parameters** (`count_params.out`; CPU build with random weights, same architecture):

| model | total | trainable | heads | backbone LoRA | backbone pretrained, unfrozen |
|---|---:|---:|---:|---:|---:|
| ViT + LoRA (reference; matches `train.py` log) | 107,039,257 | 20,997,913 | 20,113,177 | 884,736 | 0 |
| **ViT + WideLoRA r64** | **115,591,705** | **30,657,049** | 20,113,177 | 9,437,184 | 1,106,688 |
| ViT + WideLoRA r64 + `cv-detect` D2 (D2: +364,768, in the neck) | 115,956,473 | 31,021,817 | 20,477,945 | 9,437,184 | 1,106,688 |
| ViT + WideLoRA r128 | 125,028,889 | 40,094,233 | | | **over the 40 M cap** |

WideLoRA adds no other new modules ("other" = 0). Every trainable parameter is a head, a LoRA matrix or an unfrozen
pretrained tensor, so `cv-detect`'s modules add on directly.

**Time and memory on the L40S:**
- **Backbone FLOPs.** Training FLOPs per image, forward + backward (`flops.out`, meta device):

  | config | GFLOP | vs reference |
  |---|---:|---:|
  | reference LoRA | 467 | 1.00× |
  | **WideLoRA r64** | 523 | **1.12×** |
  | MoE-LoRA | 934 | 2.00× |
  | full fine-tuning | 639 | 1.37× |

- **Whole training step** (`step_flops.out`; CPU, attention added analytically because CPU SDPA has no FLOP formula):
  - The reference is 695 GFLOP/image, of which the backbone is 467 (**67 %**) and the neck, RPN, RoI and cls head 229.
    That confirms critic A5's ~65 %.
  - WideLoRA is 752 GFLOP/image: **+8.2 % per step**.
- **Time** (critic A5). The +8.2 % FLOP figure is an upper bound from FLOPs. The fp32 RPN/RoI heads and non-FLOP work
  (anchor matching, NMS, sampling, data loading) dilute the backbone's share of wall time.
  - The only *measured* anchor is MoE-LoRA: +100 % backbone FLOPs for ~+28 % time (TA, Blackwell). At that share
    WideLoRA costs +3.4 %.
  - On the L40S the share could move either way. Both its compute and its memory bandwidth (864 GB/s) are below
    Blackwell's, so the memory-bound parts (GroupNorm, RoIAlign, NMS, elementwise ops) may grow as much as the backbone.
    The smoke test settles it (critic C4).
  - I dropped my earlier 17 % reading of "about 50 min" as full fine-tuning: DETAILS.md doesn't say that.
  - Adding the below-peak efficiency of rank-64 GEMMs gives **1.05–1.15× the reference**:
    - per holdout run: **1.30–1.43 h**. The reference takes ≈ 1.24 h on the final 8,057-image local train, scaled
      from the measured 1.3 h on a 90 % split;
    - 3 seeds: 3.9–4.3 GPU-h;
    - all-train submission run: 1.47–1.61 h.
- **Mandatory GPU smoke test before any seed launches.** Through `workspace/gpu_run.sh`, run ~100 iterations of
  WideLoRA, then of the reference, back to back on the same GPU, with the local holdout data. Compare images/s and
  `torch.cuda.max_memory_allocated`. This replaces the estimate above, costs ~5 GPU-min, and also checks that the
  build, the 3 param groups and the bf16 path run.
- **Memory.** Activations saved for backward grow by +1.86 GiB at batch 16 (`activation_memory.out`, bf16 autocast on
  CPU; full fine-tuning +1.48). The reference process uses ~14 GB on our L40S (nvidia-smi on the running repro), so
  WideLoRA needs **~16 GB**.

**Risk, early signals, fallback:**
- *Instability.* Unlikely: B = 0 initialisation, rank-stable γ = 1, embeddings at the TA's 2e-5, and 1 warmup epoch.
  - Concrete check (critic A2): stop the run if any NaN/inf appears in epoch 1, or if any logged total loss in epoch 1
    (every 50 iterations) is > 2× the local reference's at the same iteration.
  - Fallback: unfreeze nothing beyond LoRA (the attribution variant, 29,550,361 trainable) and rerun.
- *Too slow.* The smoke test above decides before any seed. During a run, the signal is images/s below 0.85× the
  smoke-test reference on the same GPU. Fallback: r32 (1.06× backbone FLOPs, §3).
- *No gain: a gross-failure detector only* (critic A3).
  - With `--eval-every 4`, compare holdout mAP at epoch 8 with the local reference's log at epoch 8.
  - Kill WideLoRA only if it is **≥ 2 mAP points behind on both of the first two seeds**. That saves ~0.9 h per
    remaining run.
  - It is not a test. One early checkpoint on a steep curve favours fast learners. Unless the pairing note above is
    applied, "same seed" doesn't mean the same head init or batches, because WideLoRA's 96 LoRA matrices (48 A, 48 B)
    consume RNG first. And the holdout has only 336 independent units, with 7 for airport and 9 for wind turbine (DATA.md §0.9).
  - It stops runs; it doesn't select checkpoints. Fallback: the reference adaptation, or TerraMind + MoE-LoRA if it
    leads.
- *Classification overfitting.*
  - Why it might happen: the reference's train cross-entropy is already ~0.03 by epoch 13, and more capacity gets there
    sooner.
  - Why it probably won't hurt: full fine-tuning, the highest-capacity configuration, also has the best F1 here (95.8
    test).
  - Signal: holdout F1 at epochs 16–24 falling below the reference while mAP rises.
  - Mitigation: D1b, but only if it passes `cand_detect.md`'s 2-fold cross-fit. Label smoothing is a possible ablation,
    not in the design.
  - A sharper classifier also changes D1. D1's a-priori setting is the product rule s_c·p_c (α = 1), which never
    raises a score. With p_cls near one-hot, it acts like a hard *filter* to the predicted class. Other-class boxes,
    including unannotated objects, drop to ≈ 0, which is good. But a misclassified image's true-class boxes drop to the
    bottom of their class's ranking, which is bad.
  - So D1's adopt-or-not rule has to be re-applied on WideLoRA's own 3 holdout seeds, not carried over from the
    reference's (critic A4; `cv-detect` agrees).
- *Detection under-trained.* Not expected. From mid-training CE is ~7× smaller than the detection losses, and its
  gradient (p − y) is near zero on most training images, so detection already drives the shared parameters.
- *Near-duplicates and overfitting* (DATA.md §0.1, §5, final split).
  - 6,859 of the 9,392 train images (73.0 %) share pixels or objects with another train image. They form 964 groups;
    with 2,533 single images that makes only **3,497 independent units** (2.7 images per unit).
  - Images per unit: wind turbine 102 (the whole class is 9 units), aircraft 20.4, oil tank 8.5, down to bridge 1.1.
  - Memorising train is easy, and the overfitting risk of 10.5 M adapted parameters is far higher than 9.4 k images
    suggest.
  - The holdout keeps every group on one side, so it measures generalisation to new scenes and does not reward
    memorisation. If val or test share groups with train, extra capacity helps there *more* than on the holdout. That
    can't be checked without val/test pixels, which we don't use. So a holdout-based choice is conservative for
    WideLoRA.
  - Signal: holdout mAP flat or falling from epoch 16 to 24 while train losses keep falling.
  - Fallback: r32, the first ablation in §3. It also answers the efficiency question.
- *Budget clash with `cv-detect`.* None at D2's +0.36 M. If a later module needs more, r32 frees 4.7 M.

**Rule check:**
- Total parameters 115.6 M ≤ 130 M; trainable 30.7 M ≤ 40 M, heads included. With D2: 116.0 M and 31.0 M.
- Not full fine-tuning: 10.5 M of the 86 M backbone parameters train (12 %). That is LoRA plus partial fine-tuning,
  both explicitly allowed.
- Pretrained model: timm `vit_base_patch16_224.augreg_in21k` (ImageNet-21k), already cached. No SAR pretraining, no
  downloads.
- One model, one backbone pass, 512 px, each image once, no TTA.
- Train only, on `data-eng`'s group-aware split (seed 0, `--data dataset/holdout_sarmtl`).
  - Final leak-free split (DATA.md §6, `split.json`, after the lead's sweep and H3; the holdout is frozen): local
    train 8,057 images / 27,504 boxes; holdout 940 images / 2,715 boxes.
  - A buffer of 395 images is in neither: an 87-image wind-turbine block, 303 from the sweep and 5 from H3. The
    submission run uses all 9,392.
  - Train has 964 groups + 2,533 singles = 3,497 independent units: 3,176 in local train, 336 in the holdout.
  - The per-split counts overlap (3,176 + 336 = 3,512 > 3,497). Per-side units are the connected components within
    each side, and the wind-turbine block and the swept buffer images cut some groups into several components.
  - Holdout units per class: airport 7, wind turbine 9, oil tank 12, car 14, aircraft 16, harbor 19, playground 60,
    bridge 94, ship 105.
  - No design choice here depends on the exact numbers.

## 2. Considered and not proposed

- **Uncertainty weighting** (Kendall et al., CVPR 2018). At its optimum each task's weight is ∝ 1/L.
  - The repro logs (all of train; mean of the 12 logged batches per epoch) show train CE falling 1.11 → 0.27 → 0.12
    → 0.03 at epochs 1, 4, 8 and 13. Over the same epochs the detection losses go 0.33 → 0.21 → 0.20 → 0.18.
  - So by epoch 13 CE would get ~7× the weight of detection (0.183 / 0.026). That pushes the shared backbone toward
    the task that's already fitted, while mAP is worth 3.5× more Δm per point (1.91 vs 0.54 on val).
- **GradNorm** (ICML 2018) and **PCGrad** (NeurIPS 2020).
  - Both need per-task gradients on the shared parameters, i.e. a second backward through the backbone: ~+30 % time,
    like MoE-LoRA's second pass.
  - Two NeurIPS 2022 studies, Xin et al. ("Do current multi-task optimization methods even help?") and Kurin et al.
    ("In defense of unitary scalarization"), find they rarely beat tuned equal weights.
  - Our 3-seed resolution is ~2 Δm, too coarse to see the small gains they might give.
- **A fixed CE weight sweep.** Each value costs 3 runs (≈ 4 GPU-h) to resolve effects that the analysis above says are
  small. The loss stays the plain sum, as in the reference.
- **"With D1, the classifier also feeds mAP" (DATA.md §0.7) doesn't change this.**
  - D1 needs p_cls to be accurate and calibrated *on unseen images*. Up-weighting a train CE that is already ≈ 0.03
    buys neither; on 3,497 independent units it would mostly fit train harder.
  - D1's setting is fixed a priori (product rule, α = 1, score_thresh 0.001; `cand_detect.md` §2). Its adopt-or-not
    rule is re-applied on each model's 3 holdout seeds, WideLoRA's included. Any other α or mode can only be adopted
    through a 2-fold image cross-fit. Calibration is tested, not tuned.
  - The class cues the classifier uses (grey level, 256² upsampling; DATA.md §0.5, §0.8) are learned easily under
    equal weights anyway: train CE collapses by mid-training.
  - **The one loss-side change worth an ablation is label smoothing** (critic): 0 parameters, 0 time. Train CE ≈ 0
    means an overconfident classifier, which turns D1's product rule into a hard filter and buries misclassified images'
    true-class boxes. Smoothing targets exactly that, unlike GradNorm or PCGrad. It is not in the design; it is an
    ablation option if D1 turns out to be hurt by overconfidence on WideLoRA's checkpoints.
- **MoE-LoRA as-is** on ViT. 2.0× backbone FLOPs (measured) and ≈ 1.3× wall time, for +1.4 mAP / −0.6 F1 (test). On
  TerraMind, MoE-LoRA is the lead's planned reference.
- **Partial fine-tuning of the last 2 blocks** (14.2 M). It spends the budget on 2 of 12 depths and leaves the
  input-level SAR shift (patch embedding, early blocks) and the position embedding unadapted.
- **Full-rank attention-only tuning** (Touvron et al., ECCV 2022). 28.3 M, i.e. 48.4 M trainable with the heads: over
  the cap.
- **Conv adapters** (Mona, CVPR 2025; Convpass). Promising for dense prediction, but they overlap with D2's conv branch
  and need token↔grid reshapes in every block. A follow-up if budget remains.
- **DoRA** (ICML 2024). Small reported gains, and it adds per-step weight-norm work on 48 large matrices.

## 3. Ablation options, if the lead wants one from me (all costed, all CPU-checked)

| option | question it answers | total / trainable | backbone FLOPs vs reference | time per holdout run | expected Δm vs WideLoRA |
|---|---|---|---:|---|---|
| **WideLoRA r32** (γ = √2, rank-stable) | efficiency credit: similar Δm with half the LoRA? Also tests whether r64 overfits the 3,497 independent units. | 110,873,113 / 25,938,457 | 1.06× (step +4.0 %) | 1.26–1.34 h (1.02–1.08×) | −1 to 0 |
| **TaskFork** | does task-specific capacity still help once the shared adaptation is wide? | 116,378,137 / 31,443,481 | 1.46× (step +31 %) | 1.43–1.67 h (1.15–1.35×), ~19 GB | −0.1 to +2 (central ≈ +0.7) |
| **TerraMind + WideLoRA** | is EO pretraining better at this capacity? Run only if TerraMind + MoE-LoRA beats WideLoRA on the holdout. | 115,488,025 / 30,553,369 | ≈ 1.13× its own LoRA reference | ≈ 1.05–1.15× TerraMind + LoRA's time (not yet measured on the L40S) | unknown |

**Efficiency credit** (critic A6). WideLoRA spends +9.7 M trainable parameters over the reference: 30.7 M vs 21.0 M.
In the report that is defensible only with the r32 point beside it, which is why r32 ranks first.
- If r32 matches r64 within the minimum detectable effect, the report should prefer r32, at 25.9 M trainable.
- If r64 is clearly better, the extra parameters are justified by measurement.

**TaskFork** (`scratch/cv-adapt/task_fork.py`, `task_fork.out`):
- **Architecture.** Blocks 0–7 are WideLoRA, shared, run once. Blocks 8–11 keep their frozen weights, small tensors and
  final norm shared, but every linear layer gets one LoRA per task: classification r16 (γ = 2), detection r64 (γ = 1),
  rank-stable, B = 0.
  These 4 blocks run once per task, from the same block-7 tokens.
- **Code.** `TaskForkModel` subclasses `MultiTaskModel` only to route the cls-path tokens to the classification head and
  the det-path tokens to the neck.
- **Trainable split:** heads 20,113,177, shared LoRA 6,291,456, cls LoRA 786,432, det LoRA 3,145,728, small tensors
  1,106,688.
- **Activations and checks.** Saved activations: 10.6 GiB at batch 16 (WideLoRA 8.05). At init both paths are
  identical. After one step, every trainable tensor except the LoRA A matrices has a gradient; A has none on step 1
  because B = 0.
- **Why it's not a candidate.**
  - For: MoE-LoRA's task routers gained +1.4 / +1.0 mAP (ViT, test / val), and MTLoRA and MLoRE (CVPR 2024) report
    gains from task-specific low-rank modules.
  - Against: MoE-LoRA's gain is confounded with per-token routing, and it lost 0.6 / 1.1 F1. Train CE is ≈ 0.03 by
    epoch 13, so interference is mostly an early-training effect. And WideLoRA's 10× capacity leaves less to fight
    over.
  - Net: the central +0.7 is far below the minimum detectable effect of 3 seeds, ≈ 2.0–2.5 Δm at p < .05 (§1), and
    more on this holdout (DATA.md §0.9).
- **Cost of the cheaper k = 2 variant:** 31,050,265 trainable, 1.29× backbone FLOPs.

**My ranking if one is wanted:** r32 first (cheapest, and it may earn efficiency credit), then TaskFork for the report's
multi-task ablation. TerraMind + WideLoRA only if the condition above holds.

## 4. Suggested evaluation order

0. GPU smoke test: WideLoRA vs the reference, ~100 iterations each, the same GPU, through `gpu_run.sh` (~5 GPU-min).
   Use the measured images/s and peak memory in the plan.
1. WideLoRA × 3 seeds on the holdout, paired with the local reference if possible (§1, pairing note). Apply the
   gross-failure kill rule, then the decision rule.
2. Decoding: each model gets its own pre-registered D1 decoding by `cv-detect`'s per-model adopt rule, applied to its
   3 holdout seeds (WideLoRA's included). Models are compared as submitted, with their D1-off numbers alongside.
3. One ablation from §3 if GPU time remains, r32 first.
