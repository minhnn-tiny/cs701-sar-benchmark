# REVIEW: the critic's verdicts on the candidates and the holdout

Author: `critic`. Reviewed: `cand_detect.md` (D1, D2; cv-detect), `cand_adapt.md` (WideLoRA; cv-adapt), `DATA.md` and
`dataset/holdout_sarmtl/` (data-eng). Two rounds per cv teammate. Every check below was run inline on the CPU,
read-only, from the repo root. No GPU was used, and no file in the repo was written except this one. Image renders for
visual checks went to the session's scratchpad and were deleted afterwards.

## Verdicts

| item | verdict | what the conditions are about (details in §5–§8) |
|---|---|---|
| Holdout `dataset/holdout_sarmtl/` (data-eng) | **accept** | The final frozen split (940 / 8,057 / buffer 395; md5 `9915a68a`, `0a6eb4d2`) passes every check, and my independent searches find 0 crossing pairs. H1–H3 are resolved. Caveats for the report: no dark ships in the holdout, a smaller local train (wind_turbine 611), few scenes per class (§5) |
| D1 joint decoding (cv-detect) | **accept with conditions** | per-model adopt rule; each model scored with its own pre-registered decoding (lead); toy table rerun on local train |
| D2 spatial-prior branch (cv-detect) | **accept with conditions** | GPU gate; pre-registered paired ablation; its gain is likely below what 3 seeds resolve |
| WideLoRA r64, γ = 1 (cv-adapt) | **accept with conditions** | GPU smoke test; the local reference runs through the unchanged root `train.py`, so comparisons with it are unpaired (lead) |

Nothing is rejected. No candidate breaks a rule or the budget.

## 1. Rules (CLAUDE.md "Competition rules" and the lead's hard constraints)

- **No val/test data.** Every path in `workspace/sarmtl/data/`, `scratch/cv-detect/` and `scratch/cv-adapt/` points at
  `dataset/train`, `dataset/classes.json` or `dataset/holdout_sarmtl` (grep). Nobody used val/test pixels or metadata,
  including for leakage checks.
- **Pretrained models.** Only `timm/vit_base_patch16_224.augreg_in21k` and `ibm-esa-geospatial/TerraMind-1.0-base` are
  cached. No torchvision or other ImageNet weights exist, so D2 trains from scratch, as proposed. Nothing needs a
  download.
- **One model, one pass, ≤ 512 px.**
  - D1 combines the two heads of the same forward pass. It is a decoding rule, not TTA or an ensemble. The report must
    describe it.
  - D2 reads the same normalised 512 × 512 tensor in the same pass.
  - WideLoRA runs the backbone once.
- **"One shared backbone."** D2 is a 0.36 M conv branch inside the detection neck. That is the TA's own "light conv
  branch" suggestion, and new necks are allowed. The report should present it as part of the neck, with its count.
- **Partial fine-tuning.** WideLoRA trains LoRA (9.4 M) plus 1.1 M pretrained tensors: patch embedding, pos_embed,
  cls_token, LayerNorms and biases. That is 10.5 M of 86 M, not full fine-tuning.
- **Training-time augmentation and labels** (DATA.md §7).
  - P1–P4, P6 and P7 change only training inputs, so they are allowed. P1's zoom-in crops are train-time only; the
    inference input is unchanged.
  - P5 edits our own train labels: allowed, but it must be reported. It must never touch holdout labels, and its sign
    is uncertain, because val/test labels likely follow the same annotation convention.
- **Not in the design, but note if used.** MoE-LoRA and TaskFork run all or part of the backbone twice per image. The
  TA's MoE-LoRA is the precedent.

## 2. Budget (recounted independently on the CPU, random weights)

| model | total | trainable | ≤ 130 M | ≤ 40 M |
|---|---:|---:|:-:|:-:|
| ViT + LoRA (reference; matches the repro log) | 107,039,257 | 20,997,913 | ✓ | ✓ |
| ViT + MoE-LoRA | 107,186,713 | 21,145,369 | ✓ | ✓ |
| TerraMind + MoE-LoRA | 106,493,209 | 21,145,369 | ✓ | ✓ |
| ViT + LoRA + D2 | 107,404,025 | 21,362,681 | ✓ | ✓ |
| ViT + WideLoRA r64 (γ = 1) | 115,591,705 | 30,657,049 | ✓ | ✓ |
| **ViT + WideLoRA r64 + D2 (the likely design)** | **115,956,473** | **31,021,817** | ✓ | ✓ |
| ViT + WideLoRA r64 + D2-wide | 116,909,689 | 31,975,033 | ✓ | ✓ |
| TerraMind + MoE-LoRA + D2 | 106,857,977 | 21,510,137 | ✓ | ✓ |
| TerraMind + WideLoRA r64 | 115,488,025 | 30,553,369 | ✓ | ✓ |
| ViT + WideLoRA r128 | 125,028,889 | 40,094,233 | ✓ | **✗** |

- **Heads.** 20,113,177: cls 8,457, neck 5,560,000, RPN 597,790, RoI heads 13,946,930.
- **The counts add exactly.** D2 changes only the neck and WideLoRA only the backbone. No parameter is shared or counted
  twice, and buffers are excluded (TerraMind's 0.17 M sin-cos `pos_emb` is a buffer).
- **WideLoRA's AdamW groups (verified):** 9,437,184 @ 1e-4 (LoRA), 1,106,688 @ 2e-5 (pretrained tensors) and
  20,113,177 @ 1e-4 (heads; D2 joins this group). All 48 LoRA modules are r64 with scale 1.0, and B = 0 at init.
- **Headroom of the design:** 14.0 M total, 9.0 M trainable.

## 3. Schedule (2 healthy L40S, GPUs 1–2; deadline 31 Oct 12:00 SGT)

| run | h per run | runs | GPU-h |
|---|---:|---:|---:|
| GPU smoke tests / gates (cv-adapt, cv-detect) | – | – | ≈ 0.3 |
| local reference ViT + LoRA, holdout | 1.3 (measured) | 3 | 3.9 |
| design WideLoRA + D2, holdout | 1.41–1.61 | 3 | 4.2–4.8 |
| WideLoRA alone, holdout (D2's paired comparator) | 1.37–1.50 | 3 | 4.1–4.5 |
| TerraMind + MoE-LoRA, holdout (1.3–1.6× LoRA, not measured on the L40S) | 1.7–2.1 | 3 | 5.1–6.2 |
| submission run (the chosen design, all of train) | 1.5–1.75 | 1 | 1.5–1.75 |
| D1 caching + predict | – | – | ≈ 0.2 |
| **total** | | **13** | **≈ 19–21** |

Optional: WideLoRA r32 × 3 (+4.0–4.2 GPU-h), TaskFork × 3 (+4.5–5.3).

- **Capacity.** From approval (24 Oct) to 31 Oct 12:00 is ~6.5 days, ≈ 300 GPU-h nominal on 2 GPUs, of which the plan
  needs ≈ 7 %. 13 runs at 2 per wave make 7 waves, ≈ 10–12 h wall-clock if both GPUs are free. It fits even at 25 %
  availability.
- **Risks that matter more than GPU-hours.**
  - **tom** runs 31–35 GB jobs on GPUs 1–2. `gpu_run.sh` needs ≥ 25,000 MiB free and < 20 % utilisation, so a long tom
    job blocks us entirely. Queue early.
  - **hanliu's app** holds **17.6 GB on GPU 1** (seen 23 Oct 23:35; CLAUDE.md says 2–4 GB). That leaves ≈ 27.9 GB, so
    our process must stay below ~27 GB there. The design should need ~16–17 GB; the gates measure it.
  - **GPU 0** is unhealthy, so every 3-seed config takes 2 waves.
  - **The critical path is implementation.** The new train script carries 3 param groups, D2, the pairing (re-seed and
    DataLoader generator) and D1 caching; predict must rebuild WideLoRA + D2.
- **Order** (pairing-aware):
  1. the gates;
  2. the local reference × 3, which also gives σ. It runs through the unchanged root `train.py`, so it can start as
     soon as the holdout is verified and the design approved;
  3. design seed s alongside WideLoRA-alone seed s, for s = 0, 1, 2. The kill rules compare these concurrent arms at
     epoch 8;
  4. TerraMind + MoE-LoRA × 3;
  5. the decisions, then the submission run, predict val, and the zip.

  Start the submission run by 30 Oct.
- **A side note.** `gpu_run.sh --status` is not strictly read-only: it deletes locks whose pid is gone.

## 4. Seed noise: what 3 seeds can resolve

- **The per-seed Δm std is 0.9 for the TA's reference** (val/test, 1,426 / 2,065 images); mAP drives about 90 % of
  it. Our holdout is smaller (940 images, ×1.23 if the noise scales with 1/√n). With all links applied it has only
  **336 independent units**, with 7 for airport and 9 for wind_turbine. So expect **σ ≈ 1.1–1.5 or more**, and
  recompute it from the 3 local reference seeds before reading any result.
- **The minimum detectable difference between two 3-seed means** (two-sided t, df = 4) is 2.27σ at p < .05 (50 %
  power) and 3.03σ at 80 % power:

  | σ | p < .05 | 80 % power |
  |---:|---:|---:|
  | 0.9 | 2.0 | 2.7 |
  | 1.1 | 2.5 | 3.3 |
  | 1.3 | 2.9 | 3.9 |

  At the val reference, 1 mAP ≈ 1.9 Δm, and more if the local reference's mAP is lower. So ≈ 1.1–1.5 mAP is needed.
- **What that means for each claim:**
  - **WideLoRA vs the reference** (prior +2 to +8) is resolvable only if the prior's centre holds.
  - **D2 on WideLoRA** (+0.8 to +4.0 Δm) is mostly below the threshold, so expect "inconclusive". The pre-registered
    point-estimate rule decides.
  - **The design vs TerraMind + MoE-LoRA** (val +4.4) is likely inconclusive; the higher mean decides (pre-stated).
  - **D1** is paired on the same checkpoint, so seed noise cancels and its decision is sensitive.
- **Common random numbers.** Both cv teammates implemented them: same head init, batch order and flips, with D2's
  modules built from their own RNG (CPU-checked, bit-identical at init). They may shrink the noise of differences;
  nobody counts on that.

## 5. Holdout `dataset/holdout_sarmtl/` (data-eng): accept

**Final state:** 24 Oct, ~05:24 SGT.
- md5: val/labels.csv `9915a68a` (frozen, `workspace/sarmtl/data/frozen_holdout.json`), train/labels.csv `0a6eb4d2`,
  halves.json `d1b94180`.
- **940 holdout / 8,057 local train / 395 buffer** images, with 2,715 / 27,504 / 2,314 boxes.
- The buffer is excluded from local experiments only; the submission run trains on all 9,392. It holds:
  - the 87-image wind-turbine block (option A);
  - 303 local-train images swept for overlapping a holdout image;
  - the 5 remaining H3 ids, forced in by the lead.

**Verified independently** (inline, CPU; the searches ran on the 8,062-image version just before the 5 forced ids,
which removing local-train images can only make cleaner):
- **Integrity.** Holdout, local train and buffer are disjoint and cover all 9,392 images.
  - The holdout equals the frozen list.
  - labels.csv rows, COCO image entries, categories and all 32,533 boxes equal the source.
  - The symlinks resolve.
  - All 22 of data-eng's assertions read OK.
- **Links.** None of the 40,787 links (38,841 near-duplicate + 1,946 audit) joins holdout and local train.
- **Strata.** Every class is at 10.0 % of its images (car 9.9 %).
- **Halves** (475 / 465): disjoint, covering the holdout, with no link between them.
- **Regression.** My 8 leaking pairs are all on one side, and all 33 H3 partner ids are in the buffer.
- **Same-site search** (box geometry under 8 transforms + 64-px high-pass NCC ≥ 0.15): **0** crossing pairs.
- **Shifted-tile search.**
  - Method: zero-padded, overlap-normalised FFT correlation of 32-px high-pass thumbnails under 8 transforms, gated at
    0.5 on ≥ 20 % overlap or 0.85 on ≥ 10 %; then refinement at 128 px and confirmation at full resolution
    (high-pass NCC ≥ 0.3 on an overlap ≥ 5 %).
  - Sensitivity: 58 of 60 known within-train shift links are recovered.
  - Result on the 8,062 version: 11 weak wind-turbine pairs (high-pass NCC 0.31–0.35, no identical pixels). All 11
    involve ids 430, 779, 1439, 5055 or 7686, which are now in the buffer.
  - **Result for the final version: 0 crossing pairs.**

**History of the objections (all resolved):**
- **H1, flipped revisits.** The same site in another acquisition, flipped, with boxes matching to 1 px. 9 holdout
  images were affected. Fixed by data-eng's `georevisit` rule (187 links).
- **H2, shifted tiles.** data-eng's shift searches gated on the raw cross-correlation peak, which scales with the
  overlap, so 20–44 % overlaps went undetected. At worst, 370 of 940 holdout images shared pixels with local train
  (e.g. aircraft 4766 ~ 1301: 44 % overlap, 100 % identical pixels).
  - Fixed by data-eng's reimplementation of my normalised search (`workspace/sarmtl/data/nshift.py`) over all pairs,
    then the wind-turbine block buffer.
- **H3, residual 20 % strips.** These sat at both detectors' 20 % gate: 10 harbor and 1 car holdout images, plus 14
  ambiguous wind-turbine images.
  - Fixed by freezing the holdout and a one-pass sweep of every local-train image that any detector finds overlapping
    a holdout image, plus the 33 ids I listed.
  - A sweep can't create new crossings, which is why this pass terminates.

**Recorded detector settings, the lead's decision:**
- data-eng's 64-px pass for small (5–10 %) overlaps is gated at high-pass NCC ≥ 0.5, from its own null calibration.
  On unrelated pairs its best 5–10 % match reaches 0.3 in 10 % of wind-turbine pairs (max 0.487). At 0.3 it would
  sweep 1,290 images, mostly chance matches.
- My 32-px search keeps 0.3; on unrelated pairs it maxes at 0.253.
- So overlaps below 10 % are searched at a 0.5 bar only.

**Lead decision (a), recorded: no dark ships in the holdout.** The near-black 256² ship chips (538 images, 45 % of
ships) chain into one look-alike group, which stays in local train. Holdout ship AP covers the brighter ships only, so
a design difference on dark chips is invisible locally. The report must say this.

**Checked and fine:**
- **Car mosaics.** 88 of the 443 car images are MSTAR-style 4 × 4 mosaics of 128 px chips. No chip is reused (max chip
  HP-NCC 0.58; copies are ≥ 0.9), so they are look-alikes.
- **Dark single-ship chips** across an earlier split: full-res HP-NCC ≤ 0.07, and box sizes differ. Look-alikes.

**Open, for DESIGN.md:**
- **The buffer costs local training data.** Local train has 8,057 images, and wind_turbine loses the most: 611 local
  vs 918 in all, with 215 in the buffer. Local results for wind_turbine are therefore more pessimistic than the
  submission model's, which trains on all 9,392. State the local train size in the report.
- **Absolute numbers.** The group-aware, buffered holdout is harder than val/test may be: if they were drawn at
  random, many of their images could have a twin in train. Holdout numbers will be lower than the leaderboard's, and
  capacity-heavy designs (WideLoRA) are judged conservatively. Use the holdout to rank designs.
- **Few scenes per class.** The holdout has ~336 independent units; airport has 7 and wind_turbine 9. Act on pooled
  metrics only, and expect σ at the top of §4's range or above.
- **The Δm reference.** Local Δm must use the local reference's 3-seed means, not 93.31 / 26.13.
- **Not searched by anyone:** scale-changed crops, and overlaps below 10 % except at the 0.5 bar.
- **For the report:** how train was split, the link rules, the buffer, the frozen holdout, the dark-ship exclusion and
  the local train size.

## 6. D1 joint decoding (cv-detect): accept with conditions

**Resolved in two rounds:**
- **D1-a.** The adopt rule is applied per model: first to the local reference, then again to the design's 3 seeds
  before its submission.
- **D1-b.** The downside is measured by `d1_analysis.py`, a CPU re-decode of one cached pass: paired per-class AP, an
  oracle split (correctly vs all classified images) and the confidence on wrong images.
  - The synthetic study changed the a priori rule to the **product**, s_c·p_c (α = 1). It never raises a score, so it
    can't turn an unannotated other-class object into a confident image-class FP.
  - The posterior is reachable only through the 2-fold cross-fit.
  - I verified the product-mode logits: q_bg = s_bg + Σ s_c(1 − p_c^α), so the scores sum to 1.
- **D1-c.** The protocol wording matches between the two cand docs.

**Decided by the lead:**
- **C1. The toy table moves off the holdout.** It was built from the ground truth of 18 holdout images (no model
  output). cv-detect reruns it on local-train images after the re-draw.
- **C2. Cross-model comparisons.** Each model is scored with its own pre-registered decoding: product α = 1 if it passes
  the per-model rule, else α = 0 at 0.001. The D1-off numbers are reported alongside. This replaces cv-adapt's "the
  same D1 setting".

**Resolved with the final split:**
- **C3.** `halves.json` (md5 `d1b94180`) comes from the final groups: 475 / 465 images, no link between the halves.

**Open, for DESIGN.md:**
- **C4.** The real `report()` call must use the local reference means; the selftest uses the val reference.
- **Submission size.** `score_thresh` 0.001 fills up to 100 detections per image, ≈ 143 k detections for val (≈ 15 MB
  of JSON). Check that the zip uploads and scores, starting with the sample-format test.

## 7. D2 spatial-prior branch (cv-detect): accept with conditions

**Resolved in two rounds:**
- **Power.** The primary endpoint (paired last-epoch holdout mAP) and the secondary (mean AP of the 4 tiny classes) are
  pre-registered. Keep D2 for Phase 2 iff both 3-seed means are > 0.
- **Common random numbers.** The new modules are built inside `fork_rng`, and the shared modules are identical to the
  no-D2 arm's (CPU check 5).
- **Base.** The expected gain on WideLoRA is +0.5 to +2 mAP. Additivity is no longer assumed.
- **Kill rule.** The comparator is the same-seed WideLoRA-alone arm. A run stops only on gross failure: ≥ 2 mAP behind
  on both of the first two seeds at epoch 8, or divergence.
- **Init.** The lateral GN γ is zero-initialised, fixed a priori: D2 starts as the comparator's function.
- **Time and memory.** A GPU gate comes before any seed: ≤ +10 % per iteration and peak < 27 GB. Its pre-registered
  fallback is a patchify stem.

**Open, for DESIGN.md:**
- **D2's case rests on point estimates.** Its expected gain is mostly below the minimum detectable effect (§4).
- **The gate's conditions.** Run it on the GPU the seeds will use, and count the process peak (max_memory_allocated
  + ~2.5 GiB).
- **The motivation is weaker on WideLoRA.** D2's premise (a frozen ViT lacks sub-patch detail) holds less once the
  patch embedding trains, so the paired no-D2 arm is required, not optional.

## 8. WideLoRA (cv-adapt): accept with conditions

**Resolved in two rounds:**
- **A1. The decision rule and the minimum detectable effect are stated before any result.** Adopt WideLoRA if its
  3-seed mean Δm is > 0 vs the reference and no evaluated in-budget configuration is higher. The submission backbone is
  whichever is higher, the design or TerraMind + MoE-LoRA.
- **A2. γ = 1 at r64** (rank-stable α = 64), verified in code. It replaces γ = 2, which would have taken 2–4× the
  reference's weight-space steps.
- **A3. The kill rule is a gross-failure detector only:** ≥ 2 mAP behind on both of the first two seeds.
- **A4.** The D1 wording is fixed, and D1 is re-checked on WideLoRA's checkpoints.
- **A5. Time is 1.05–1.15×.** The backbone is ≈ 67 % of a training step's FLOPs. A mandatory GPU smoke test comes
  first.
- **A6. Efficiency.** The r32 ablation is first in line; if it matches r64, the report prefers r32 at 25.9 M
  trainable.
- **(c) and (d).** The backbone is decided by measurement. Rejecting loss balancing is justified: I reproduced it from
  the repro log, CE 0.030 vs detection 0.185 at epoch 13. Label smoothing is the only loss-side ablation worth having.

**Decided by the lead:**
- **C1. The local reference runs through the unchanged root `train.py`**, so comparisons with it are unpaired.
  Common random numbers apply only between our own script's configurations: the design vs WideLoRA-alone.
  - So WideLoRA-vs-reference and design-vs-reference differences carry the full unpaired noise of §4.
  - The design script still has to reproduce the reference protocol, apart from the deliberate changes: the same loss,
    schedule, evaluation and precision rule.

**Open, for DESIGN.md:**
- **C2.** The smoke-test numbers replace the 1.05–1.15× estimate.
- **The expected Δm range is a prior.** No in-benchmark point lies between LoRA and full fine-tuning. The learning
  rates are fixed a priori, untuned (LoRA 1e-4 at γ = 1, pretrained tensors 2e-5); say so in the report.

## 9. Interactions and risks across the design

- **WideLoRA × D1.** WideLoRA makes the classifier more confident (train CE ≈ 0.03 by epoch 13), so the product rule
  acts nearly hard. The per-model adopt rule handles this. Label smoothing is the ablation if D1 fails on WideLoRA.
- **WideLoRA × D2.** The mechanisms overlap: both give the detector finer detail. The paired no-D2 arm measures D2's
  marginal gain.
- **Precision rule: unchanged.** The RPN and RoI heads run in fp32. D1 runs on fp32 logits inside that block. D2's
  convs run in bf16 and its GN in fp32 under CUDA autocast. The CPU memory figures (CPU autocast) are indicative only.
- **Predict path.** It must rebuild WideLoRA + D2 from the `metrics.json` args before loading `model.pt`; the
  state_dict keys differ from the reference's. Test predict on a gate checkpoint before the submission run.
- **Untested until the gates run:** GPU time, memory, and the convergence of D2 and WideLoRA. The gates catch only
  gross problems.
- **For the report:**
  - the total and trainable counts (§2);
  - the D1 decoding rule;
  - the group-aware holdout: its link rules (copy, revisit, georevisit, overlap, normalised shift, audit), the frozen
    holdout with its 395-image buffer (block + sweep), the dark-ship exclusion and the 8,057-image local train;
  - every pretrained model with its source;
  - per-seed values with mean ± std for every comparison.

## Final audit (Step 5), 24 Oct 2026, ~23:20 SGT

**What I audited:**
- `workspace/sarmtl/RESULTS.md` (the 22:59 version) against its sources;
- the val submission zip against the rules;
- the rerun commands in `workspace/sarmtl/README.md`.

**How:** CPU only. Read-only, except this section. The team token was loaded in-process only to search for it; it was never
printed.

**Not audited (pending from the lead):** A1_EVAL and A1_DECODING (RESULTS §1.2–§1.3), POST_H and TOTAL_H (§5).

**Bottom line:**
- **0 blocking issues.** The val submission stands:
  - it is within the budget and the rules, and trained on train only;
  - it was chosen by the pre-registered C\* and decoding rules;
  - it is a valid zip, and reproducible from the code.
- **Every number I recomputed matches its source** (A).
- **17 non-blocking issues** (B):
  - Fix these before the report: F1 to F3 (false or inaccurate statements) and F4 (counts).
  - The rest are wording, completeness, reproducibility, one Phase 2 size risk and rounding nits.

### A. Verified, no issue

**Numbers.** Each was recomputed from `metrics.json`, `jd_eval.json`, the queue and smoke logs, or the server replies.
- **Local reference:** macro-F1 76.92 and mAP 12.52, the means of `ref_s{0,1,2}` `val`.
- **§1.1:**
  - all 12 per-seed F1 / mAP / Δm values;
  - every mean ± sample std: accuracy, F1, mAP, AP50 and Δm;
  - the parameter counts;
  - GPU h per run and in total: 0.98 / 2.94, 1.48 / 4.43, 1.12 / 3.37, 1.20 / 3.59;
  - the peak GiB;
  - ms/image: 12.7 and 17.0 from `jd_eval.json`, 13.7 and 14.2 from `metrics.json`.
- **§0 t-statistics:**
  - WideLoRA-SP − reference: 7.35 Δm, SE 1.32, t = 5.54;
  - WideLoRA-SP − TerraMind: −2.02 Δm, SE 1.51, t = −1.34;
  - the 2.27σ threshold comes to 2.8–5.2 Δm.
- **§1.2:**
  - The JD keep-rule gains are identical whether taken from the same `evaluate.py` pass or from `metrics.json`. The
    `metrics.json` `val` equals `jd_eval.json`'s reference to the last digit, so evaluation is deterministic.
  - The JD-posterior gains per seed match.
  - The cross-fit choices and out-of-fold gains match: +0.14 / +0.19, +0.17 / +0.07, +0.20 / +0.29.
- **§1.3:**
  - The as-submitted means ± std and Δm match: +0.60, +9.87, +4.30 and +8.33.
  - C\* is TerraMind + MoE-LoRA, by a margin of 1.54.
  - A1's pending decoding can't change C\*: its seed-0 posterior gain is +0.20 mAP, which leaves it near +5.
- **§1.4:** all 36 per-class APs and the tiny-4 means match.
- **§2:**
  - The SP paired table matches: ΔΔm +4.11 / +3.15 / +1.89, mean +3.05 ± 1.11; ΔF1 +1.29 ± 1.81; ΔmAP +0.55 ± 0.35.
  - A1 − reference: F1 +0.06, mAP +1.07.
  - Per-seed accuracy ranges over 73.5–78.7 %.
- **§3:** the TA's tables match CLAUDE.md; the repro fit rows match `tables.md`; repro val is 469 train images (5 %).
- **§4:**
  - The leaderboard row matches `submission_val_tmfull_s0_jdposterior.response.json`: Δm 3.9152, F1 90.917, mAP 28.847,
    accuracy 91.234, AP50 58.78.
  - Its id is `s_VlNljpmqLS`, created at 13:39:42Z (21:39:42 SGT). Quota: 1 of 5 used.
  - `post.json` shows `posted` and the phase-1 entry `s_VlNljpmqLS`.
  - Both zips have md5 `54695fb6…`.
  - Rerunning `check_submission.py` reproduces the quoted output verbatim.
- **§5:**
  - the parameter table;
  - `train_hours` 1.694 (tmfull_s0) and 1.389 (Pfull_s0);
  - peaks 14.17 and 10.40 GiB;
  - the inference ranges.

  GPU-hour rows:

  | row | measured | source |
  |---|---:|---|
  | queue | 17.895 h | 24 jobs, `start`/`end` |
  | smoke tests | 0.019 h | `runs/smoke/*.out` |
  | repro | 3.469 h | its `queue.log` |
  | first repro attempt | 0.824 h | its `queue.log`; ends in a CUDA error, so "crashed" is right |
  | TA smoke runs | ≈ 0.06 h | log times ("≈ 0.1" is generous) |
  | other session | 19.69 h | 16 `metrics.json` |

**Report facts:**
- **`.venv` versions:** torch 2.9.0+cu126, torchvision 0.24.0+cu126, timm 1.0.24, terratorch 1.2.7, pycocotools 2.0.11,
  numpy 2.5.4, pillow 13.0.0, huggingface_hub 1.33.0.
- **Hardware:** NVIDIA driver 550.163.01, 3× L40S 46 GB.
- **Model ids in the code:**
  - timm `vit_base_patch16_224.augreg_in21k`;
  - terratorch `terratorch_terramind_v1_base` (HF `ibm-esa-geospatial/TerraMind-1.0-base`, through its S1GRD input).

  Both are in the HF cache.
- **Preprocessing:** `convert('L')`, `v2.Resize((512, 512))` (bilinear with antialias), scaling to [0, 1], then
  `Normalize(0.2526, 0.2133)`; H/V flips in training only.
- **Holdout:**
  - 940 / 8,057 / 395 matches `split.json` and my final verification (§5);
  - 336 holdout units;
  - 73.0 % of train images are in near-duplicate groups (DATA.md §0).
- **Budget and tests:** `train.py` asserts the budget, and `pytest workspace/sarmtl/tests` gives 36 passed (CPU).

**The zip against the rules:**
- **Budget.** `tmfull_s0` `metrics.json`: 106,493,209 total, 21,145,369 trainable.
- **One model, one pass.**
  - `predict.py` makes one forward call per image, on the 512 × 512 resize (the baseline's evaluation transform), in
    bf16, with no TTA and no ensemble.
  - Inside that call, MoE-LoRA runs the backbone once per task: that is the TA's own reference method.
- **Trained on train only.**
  - `dataset/repro/train` is a symlink to `../train`. Its labels.csv and instances.json are md5-identical to train's:
    9,392 images, 32,533 boxes.
  - repro's val/test is a 469-image sample of train, linked to train/images.
  - `predict.py` is the only code that reads `dataset/val`.
  - C\* and the decoding were chosen on the holdout (train images) and its halves. The val score came after both were
    fixed.
- **No other SAR dataset.** For the pretrained models, see F3.
- **Format and size:**
  - 2 members;
  - the ids equal `val/images.json` (1,426, no duplicates), and the labels are 0–8;
  - 107,060 valid detections: positive size, inside the image, at most 100 per image;
  - 16,386,976 bytes, against a limit of 25,000,000.
- **Reproducible.** I rebuilt the model from `tmfull_s0/…/model.pt` on the CPU (all keys matched) and decoded 6 val
  images.
  - The labels equal the zip's on 6/6.
  - JD-posterior reproduces 95.7 % of the zip's detections and its per-image counts. Reference decoding at threshold
    0.001 reproduces 92.0 %, product 81.3 %, and reference at 0.05 only 28.1 %.
  - The remainder is CPU vs GPU bf16.
- **No token.**
  - `CS701_TOKEN` (42 characters) is only in `.env`, which is gitignored.
  - Neither the full value nor its first or last 12 characters appears in any of 519 files (`workspace/`, `.claude/`,
    the repo-root files) or in the zips' members.
  - No `Authorization` or `Bearer` lines either.

**README commands:**
- **§1** matches DATA.md §6, which records the commands data-eng ran.
- **§2** matches `runs/smoke/*.out`.
- **§3–§4** match `jobs/plan.txt` and the queue's `cmd` files. The README uses `uv run python` where the queue used
  `.venv/bin/python`.
- **§5** matches `SUBMISSIONS.md` and the predict log. Only `--out` differs, and the file was then copied
  byte-identically.

### B. Issues

None is blocking. Each issue gives its location, the problem, the evidence and the fix.

**F1. The two optional val zips are claimed built, but they don't exist.** Non-blocking; fix before RESULTS is final.
- **Where:**
  - RESULTS §0, the status line "The two optional val zips are built and checked, but not uploaded";
  - §0's caveat, "The design's val zip is built but not uploaded";
  - §4's table rows for `alt_design_widelora_sp_val.zip` and `context_vitlora_repro_s0_val.zip`.
- **Evidence:**
  - `submissions/` holds only the two copies of `submission_val*.zip`;
  - `runs/queue/post_runs.log` records "gave up after 10800 s" at 22:56:48;
  - `post_runs2.sh` has been waiting for a GPU since 22:57.
- **Fix:** write "pending (post-queue jobs waiting for a GPU)" until both zips are built and `check_submission.py`
  passes.
- **Also:** `post_runs.sh` and `post_runs2.sh` live in a session scratchpad (`/tmp/…/scratchpad/`), not in the repo.
  Add their job list to `workspace/sarmtl/jobs/`, so the A1 evaluations and the optional zips are reproducible from the
  code.

**F2. The scene count for the tiny classes is wrong.** Non-blocking.
- **Where:** RESULTS §1.4, "Their holdout AP rests on 7–14 independent scenes per class".
- **Evidence:** DATA.md §6 gives holdout units bridge 94, playground 60, wind_turbine 9, oil_tank 12. The 7–14 range is
  airport, wind_turbine, oil_tank and car.
- **Fix:** "Wind turbine (9 units) and oil tank (12) rest on few scenes; bridge (94) and playground (60) have many,
  but are barely detected."

**F3. "No SAR-pretrained model" is inaccurate for TerraMind.** Non-blocking, but the most important wording fix: the TA
inspects rule compliance.
- **Where:** RESULTS §5, "No SAR-pretrained model, and no other SAR dataset."
- **Evidence:**
  - TerraMind-1.0's self-supervised pretraining (TerraMesh) includes Sentinel-1 SAR, and the submission model uses it
    through its S1GRD input (`sarbench/backbones.py`).
  - It is allowed by name: CLAUDE.md "Allowed: generic pretrained models (ImageNet models, CLIP, TerraMind, …)". It is
    also the TA's own baseline. So this is not a rule problem, but the sentence is false as written.
- **Fix:** "TerraMind-1.0-base, a generic Earth-observation foundation model whose pretraining includes Sentinel-1 SAR,
  allowed by name by the rules. No SAR detection model and no other SAR dataset was used."

**F4. The queue's run count is wrong, and one GPU job is missing from the hours.** Non-blocking.
- **Where:** RESULTS §5, the GPU-hours table: "13 training runs + 1 smoke + 9 evaluations".
- **Evidence:**
  - The queue has 24 jobs: **14** training runs (ref ×3, P ×3, Pfull, tm ×3, A1 ×3, tmfull), `smoke_eval` and 9
    evaluations. The 17.89 h figure is right.
  - The submission zip's predict job (`submissions/predict_val_tmfull_s0_jdposterior.log`: GPU 1, 21:38:12–21:39:30,
    0.02 h) is in no row. It isn't in `post_runs.log`.
- **Fix:** write "14 training runs", and add the predict job, or fold it into POST_H and say so.

**F5. "The TA's reference code, unchanged" is not quite true, and SciPy is missing.** Non-blocking.
- **Where:** RESULTS §5, "External libraries and own code".
- **Evidence:** `git diff --stat` shows `sarbench/data.py` changed in 16 lines and `train.py` in 6. That is the
  course-data path adaptation README §2.3 asks for.
- **Also:** SciPy 1.18.1 is used by `workspace/sarmtl/data/` (near-duplicate search) and isn't listed.
- **Fix:** "unchanged apart from the data-layout paths (git diff)", and add SciPy.

**F6. "GPUs 1 and 2 were used" omits GPU 0.** Non-blocking.
- **Where:** RESULTS §5, Compute.
- **Evidence:** the first repro attempt ran on GPU 0 (`repro_attempt1_gpu0_ecc_crash/queue.log`: "seed 0 -> GPU 0"),
  and it is counted in the hours (0.82).
- **Fix:** "after the first repro attempt crashed on GPU 0 (ECC fault), every run used GPUs 1 and 2".

**F7. "≈ 19 GB in nvidia-smi" is untraceable.** Non-blocking.
- **Where:** RESULTS §5, Peak GPU memory.
- **Evidence:** no listed log contains it. `run.log` records only free memory at claim time.
- **Fix:** cite the observation, or state `max_memory_allocated` + ≈ 2.5 GiB (DETAILS.md).

**F8. The ms/image figures are not like for like.** Non-blocking; it matters for the efficiency claims.
- **Where:** RESULTS §1.1's ms/image column, and §5's inference times.
- **Evidence:**
  - `evaluate.py` times prediction plus **3** COCO scorings (all and the two halves; lines 69–73). That is the source
    of 12.7 and 17.0.
  - `train.py` times one scoring pass (`root.evaluate`). That is the source of 13.7 and 14.2.
  - `predict.py` times none.
  - So TerraMind + MoE-LoRA reads 16.7–17.3 ms on the holdout but 12.8 ms in `predict.py` on val.
- **Fix:** label the column "evaluation ms/image incl. COCO scoring (1 or 3 passes)". For the report's "inference time
  per image", use `predict.py`'s 12.8 ms, or time every configuration the same way.

**F9. "Level with the best in-budget reference" overstates the val result.** Non-blocking wording.
- **Where:** RESULTS §0, "On val, the submission is level with the best in-budget reference".
- **Evidence:** 3.92 against the TA's 4.4 ± 0.3: 0.48 below the TA's mean, 1.6 of its seed σ. All of the gap comes from
  macro-F1 (−0.98); mAP and AP50 match.
- **Fix:** "slightly below (−0.5 Δm, all from macro-F1)".

**F10. The posterior claim covers A1 seeds that aren't evaluated yet.** Non-blocking until A1_EVAL lands.
- **Where:** RESULTS §0, "gains +0.1 to +0.3 mAP on every seed of every configuration".
- **Evidence:** A1 has only seed 0 evaluated (`A1_s0/…/jd_eval.json`, +0.20); seeds 1 and 2 are pending.
- **Fix:** "every evaluated seed", until A1_EVAL lands.

**F11. "Never span the split" isn't literally true.** Non-blocking wording.
- **Where:** RESULTS §5, Validation split: "near-duplicate groups … never span the split".
- **Evidence:** the 873-image wind-turbine component has images on both sides, kept apart by the 87-image buffer (§5 of
  this review).
- **Fix:** "no near-duplicate link joins holdout and local train; one giant wind-turbine component is split with an
  87-image buffer".

**F12. `crossfit.py` adopts on a weaker criterion than the pre-registered one.** Non-blocking: no outcome changes.
- **Where:** `crossfit.py` `decide()`, used by `compare.py`.
- **Evidence:**
  - It adopts when the 3-seed mean out-of-fold gain is > 0.
  - The pre-registered criterion (cand_detect.md §2 step 4, which DESIGN.md rule 3 points to) is "> 0 on **all 3
    seeds**".
  - Every per-seed out-of-fold gain is positive in both folds: ref +0.13 to +0.22, TerraMind +0.05 to +0.20, WideLoRA-SP
    +0.15 to +0.31. So JD-posterior is adopted under either reading.
- **Fix:** add the per-seed condition to `decide()`, or state the deviation, before A1's decoding is decided.

**F13. README can't rebuild the submission model from scratch.** Non-blocking; the TA reproduces results from the code.
- **Where:** README §5.
- **Evidence:**
  - No command builds `dataset/repro` (`workspace/baseline/make_repro_data.py`, defaults `--fraction 0.05 --seed 0`).
  - The submission model's training command appears only as the queue job name `tmfull_s0`.
- **Fix:** add `uv run python workspace/baseline/make_repro_data.py`, and the full line `train.py --backbone terramind
  --init pretrained --adapt moelora --data dataset/repro --seed 0 --out workspace/sarmtl/runs/tmfull_s0` (through
  `gpu_run.sh`).

**F14. One holdout input file is undocumented, and `split.json`'s command is stale.** Non-blocking.
- **Where:** README §1 and DATA.md §6.
- **Evidence:**
  - `make_holdout.py` also reads `critic_h3_buffer.json` (line 40: the 33 H3 ids), and `near_dup_groups.json`. The docs
    name only `audit_links.csv`, `frozen_holdout.json` and `sweep_buffer.json` as tracked inputs.
  - `split.json`'s `command` field still records the old 3-step pipeline.
- **Fix:** list `critic_h3_buffer.json` as a tracked input, and update the `command` field.

**F15. Phase 2 zip size risk.** Non-blocking now; act before Phase 2.
- **Evidence:**
  - The zips are stored uncompressed: `make_submission.py` uses ZIP_STORED.
  - Val is 16,386,976 bytes. Scaled to test's 2,065 images, that is ≈ 23.7 MB, 95 % of the 25,000,000-byte limit
    (`check_submission.py`).
  - A test set with more boxes per image would exceed it (HTTP 413).
- **Fix:** write the test zip with ZIP_DEFLATED (`detection.json` deflates from 16.4 to 4.5 MB; any standard zip reader
  opens it), or round the bbox and score floats. `check_submission.py` already guards the limit.

**F16. Repo hygiene before open-sourcing.** Non-blocking.
- **Evidence:**
  - `.gitignore` covers `runs/`, `dataset/` and `.env`, but not `submissions/*.zip` (2 identical 16.4 MB copies).
  - `workspace/` is untracked, so none of the code the TA will inspect is committed yet.
- **Fix:** decide what to commit: the code, `data/` inputs and docs, with or without the zips.

**F17. Rounding and wording nits.** Non-blocking.
- Full-train wall times: 102.8 and 84.7 min, so "103 min" and "85 min", not 102 and 84.
- A1 vs reference time: 1.122 / 0.980 = 1.146, so "1.15×", not 1.14×.
- "The holdout took three re-draws": it took more. Say "several re-draws (three kinds of leak, H1–H3)".
- README tests "~20 s": the suite took 64 s here.

### Re-check, 24 Oct 2026, ~23:45 SGT

Verified again on the CPU, read-only, against the regenerated `tables.md`, the A1 `jd_eval.json` files, `post_runs.log`
and the code.

**Fixed:**
- **New numbers, all reproduced:**
  - A1's cross-fit adopts JD-posterior: A→B +0.15 (+0.18, +0.10, +0.17), B→A +0.22 (+0.24, +0.22, +0.19).
  - A1 as submitted: mAP 13.76 ± 0.23, AP50 32.87 ± 0.87, Δm +4.98 ± 0.61.
  - A1's per-class row and its whole-holdout posterior gains (+0.20, +0.14, +0.17) match.
  - C\* is unchanged: +9.87 vs +8.33 / +4.98 / +0.60.
  - The posterior gains span +0.10 to +0.27 over all 12 seeds.
- **GPU hours:**
  - POST_H = 56 + 78 + 225 s = 0.10 h (from the logs).
  - This pipeline's total is 22.37 h.
  - The other session's 17 `metrics.json` files sum to 21.36 h, so the team total is 43.73 h.
- **Inference:** `predict.py` gives TerraMind 12.8, WideLoRA-SP 10.5 and ViT + LoRA 8.9 ms (both logs, GPU 1).
- **The optional zips** exist (3,468,594 and 1,426,891 bytes), are deflated, and pass `check_submission.py`.
- **The uploaded val zip is unchanged:** md5 `54695fb6…` for both copies.
- **F12:** `decide()` now requires a gain on every seed in both folds. A test covers the case of a positive mean with
  one negative seed.
- **F15:** `predict.py` deflates the zip, with a test.
- **F13:** README §5's two lines match the queue's `cmd`.
- **F14:**
  - `split.json` md5 is `4b91e62a`; the labels and halves md5s are unchanged.
  - Its `command` equals `make_holdout.py`'s f-string at `--fraction 0.1 --seed 0`, so a rebuild reproduces it.
  - The five inputs are documented.
- **F1–F11 and F17:** the wording and numbers in RESULTS are as proposed.
  - OpenCV 5.0.0, SciPy 1.18.1 and pandas 3.1.0 are installed at those versions and used by `data/`.
  - `pytest` gives 37 passed.

**Left open (non-blocking):**
- **R1. A rerun can overwrite the byte-identical copy of the uploaded zip.**
  - `jobs/post_queue.sh` and README §5's rebuild command both write the C\* val zip to
    `submissions/submission_val.zip`, which RESULTS §4 and README §5 call a byte-identical copy of the uploaded zip.
  - A rerun (e.g. the TA's reproduction) would overwrite it with a newly predicted, now deflated zip with different
    bytes.
  - The original, `submission_val_tmfull_s0_jdposterior.zip`, is not touched.
  - Fix: rebuild to another name (e.g. `submission_val_rebuild.zip`), or skip outputs that exist.
- **R2 (nit).** RESULTS §6 says `jd_eval.json` exists "for the root `train.py` runs and WideLoRA-SP". A1 now has all
  three as well, and they feed §1.2–§1.3.
- **F16.** The commit and `.gitignore` decision is left to the user, as you decided.
