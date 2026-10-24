# DESIGN: WideLoRA-SP with joint decoding

Status: **approved at Checkpoint A, 24 Oct 2026**: the design, rules, plan and submission rule as written. JD
confirmed as one model. Holdout choices accepted. The queue started at 10:50. Lead: main session. Inputs:
[DATA.md](DATA.md) (data-eng), [cand_detect.md](cand_detect.md) (cv-detect), [cand_adapt.md](cand_adapt.md)
(cv-adapt) and [REVIEW.md](REVIEW.md) (critic). Expected effects below are **priors**: nothing has been trained yet.

## 1. Primary design (one)

ViT-B/16 (timm `vit_base_patch16_224.augreg_in21k`, ImageNet-21k, the only pretrained model) gets three changes: a
wide LoRA on every linear layer (WideLoRA), a light pixel branch into the finest pyramid levels (SP, for spatial
prior), and joint decoding of the two heads at inference (JD).

| # | component (owner) | the change | why (data fact) | prior effect |
|---|---|---|---|---|
| 1 | **WideLoRA** (cv-adapt) | `sarbench.adapters.LoRA` (unchanged) at rank 64, α = 64 (scale 1, rank-stable) on qkv, proj, fc1 and fc2 of all 12 blocks. Also unfrozen: patch embedding, `pos_embed`, `cls_token`, all LayerNorms, and the biases of the wrapped layers (1.11 M). AdamW groups: LoRA 1e-4, unfrozen pretrained tensors 2e-5 (the TA's full-fine-tuning backbone lr), heads 1e-4 | The reference adapts 1 % of the backbone (attention only) and leaves 19 M of the 40 M trainable budget unused. LoRA → full fine-tuning gains +5.9 mAP and +1.1 F1 on val. The patch embedding never saw SAR | Δm +2 to +8 |
| 2 | **SP**, spatial-prior branch (cv-detect's D2) | 6 bias-free 3 × 3 convs with GN + ReLU on the normalised 512 input, giving stride-4 (64 ch) and stride-8 (128 ch) maps. A 1 × 1 conv + GN lateral adds each map into the ViTDet neck's stride-4 / stride-8 projection before its 3 × 3 conv. The lateral's GN γ starts at 0, so the model starts exactly as WideLoRA alone. Trained from scratch in the heads group (1e-4) | 94.9 % of boxes are pooled from the stride-4 map, which in the reference is only upsampled 16 px tokens. 50.5 % of boxes are < 16 px; 4 classes are ≥ 92 % COCO-small | mAP +0.5 to +2 on top of WideLoRA |
| 3 | **JD**, joint decoding (cv-detect's D1, with cv-adapt's product formula) | Inference only, no parameters: each RoI's class score becomes s_c · p_c, with p the classification softmax of the same forward pass. The score threshold drops from 0.05 to 0.001; torchvision's per-class NMS and top 100 are unchanged | Every box has the image's class, and images hold unannotated objects of other classes (ships in harbor scenes). The classifier is 93–94 % accurate. The product never raises a score | mAP +0.3 to +1.5 |

**Unchanged:** input (every image resized to 512 × 512, `MEAN/STD`, random H/V flips), heads, losses (CE + the four
Faster R-CNN losses), AdamW wd 0.05, batch 16, 24 epochs, one warmup epoch then cosine, bf16 with the RPN/RoI heads in
fp32, and the last-epoch model.

**Deliberate protocol changes:**
- a third optimizer group, for the unfrozen pretrained tensors at 2e-5;
- JD at inference. Validation during training keeps the reference decoding, so learning curves compare across runs;
  the final holdout scores are recorded with JD on and off.

**Considered and left out** (reasons in the cand docs and REVIEW.md):
- loss balancing: train CE is ≈ 7× smaller than the detection losses by epoch 13;
- MoE or task routing (TaskFork): a second backbone pass, for an expected +0.7;
- new anchors: coverage barely changes;
- Cascade R-CNN, or a stride-2 level: budget and compute;
- despeckling: it smears 8 px objects;
- DATA.md's augmentations P1–P7 are untested. P1's zoom-in crops might also read as "upscaling", so I would ask first.

## 2. Parameters (computed on the CPU, random weights, same architecture)

Output of the lead's CPU check, which builds the teammates' prototypes `scratch/cv-adapt/wide_lora.py` and
`scratch/cv-detect/detect_modules.py` together. The same counts appear in cand_adapt.md and cand_detect.md, and the
critic recounted them independently (REVIEW.md §2):

```
reference ViT + LoRA         107039257 total    20997913 trainable    20113177 heads
WideLoRA r64 + D2 (primary)   115956473 total    31021817 trainable    20477945 heads+D2 OK
WideLoRA r64 (A1, no D2)    115591705 total    30657049 trainable    20113177 heads+D2 OK
```

| model | total | trainable | of which: heads | LoRA | unfrozen pretrained | SP branch |
|---|---:|---:|---:|---:|---:|---:|
| reference ViT + LoRA | 107,039,257 | 20,997,913 | 20,113,177 | 884,736 | 0 | 0 |
| **primary WideLoRA-SP** | **115,956,473** | **31,021,817** | 20,113,177 | 9,437,184 | 1,106,688 | 364,768 |
| A1 WideLoRA alone | 115,591,705 | 30,657,049 | 20,113,177 | 9,437,184 | 1,106,688 | 0 |
| TerraMind + MoE-LoRA (baseline) | 106,493,209 | 21,145,369 | 20,113,177 | 1,032,192 | 0 | 0 |
| budget | ≤ 130,000,000 | ≤ 40,000,000 | | | | |

`workspace/sarmtl/train.py` asserts both limits at startup and logs the counts.

## 3. Ablations (2)

- **A1, WideLoRA alone** (the primary without SP), 3 seeds. It isolates SP and is paired with the primary: same
  seeds, same data order and flips, same initial weights, with the SP modules built from their own forked RNG.
- **A2, JD off**: every checkpoint of every configuration is also scored with the reference decoding (product off,
  threshold 0.05). It costs no training, and it isolates JD paired, per model.

That attributes all three components:
- WideLoRA = A1 − the reference (both with JD off);
- SP = the primary − A1, paired;
- JD = on − off, on the same checkpoint.

## 4. Decision rules (fixed before any result)

1. **Local Δm.** Per seed, against the **local** reference's 3-seed means of macro-F1 and mAP on the holdout (not
   93.31 / 26.13), then mean ± sample std. `compare.py` does this.
2. **Detectable difference.** About 2.3 σ between two 3-seed means (p < .05). σ comes from the 3 local reference
   seeds; expect 1.1–1.5, so differences of ≈ 2.5–3.4 Δm. Smaller differences are reported as inconclusive, and the
   rules below decide on the means.
3. **JD keep rule, per configuration.** JD is on for a configuration iff its paired holdout mAP gain (JD on − off,
   same checkpoint) is > 0 on all 3 seeds. A different setting (e.g. α ≠ 1) is allowed only through the 2-fold
   cross-fit on `halves.json`. This is checked first on the reference, then on every configuration it is used for.
4. **Configuration rule: this also decides SP and the backbone.** The submission configuration C* is the one with
   the highest 3-seed mean holdout Δm, among configurations with 3 complete seeds: primary, A1, TerraMind +
   MoE-LoRA, ViT + LoRA. Each is scored as it would be submitted, with its own JD setting; JD-off numbers are reported
   alongside.
   - This replaces cv-detect's two-endpoint SP rule with the single competition metric. The mean AP of the 4 tiny
     classes is reported as secondary evidence.
5. **Kill rule, gross failure only.** A primary or A1 configuration is stopped if:
   - its holdout mAP at epoch 8 is ≥ 2 points below the same-seed reference's epoch-8 mAP, on both seed 0 and seed 1; or
   - a loss goes non-finite.

   Baselines are never stopped.
6. **GPU gate, before seeds 1–2.** Primary seed 0 must reach ≥ 0.8× the reference's images/s over its first 100
   iterations, and its process peak (`max_memory_allocated` + ~2.5 GiB) must stay < 25 GB. GPU 1 has ≈ 27.9 GB free
   with hanliu's 17.6 GB app. If it fails, I stop and report; the fallbacks are LoRA rank 32 or a patchify stem for SP.

## 5. Run plan

**Basis.** Measured ViT + LoRA times on the L40S (repro queue, all 9,392 images): `train_hours` 1.23 and 1.12, peak
7.4 GiB allocated, ≈ 14 GB per process. The other per-run hours are estimates, scaled by the teammates' FLOP and
memory analysis; the smoke tests and the gate replace them.

**Measured on 24 Oct (gate, rule 6: passed).**
- Smoke tests, 64 images and 1 epoch, both on GPU 1:
  - the primary: peak 10.4 GiB allocated, 23.5 images/s (4 cold iterations);
  - the reference: 7.4 GiB, 27.7 images/s.
- First 100 iterations of the real seed-0 runs (8,057 images, 504 iterations per epoch):

  | run | GPU | images/s at iter 50 → 100 | process memory |
  |---|---|---|---:|
  | `ref_s0` | 1 | 50.6 → 55.3 | 9.5 GB |
  | `P_s0` | 2 | 42.5 → 45.4 | 12.6 GB |

  So the primary runs at 0.82–0.84× the reference (gate ≥ 0.8), with a process peak far under 25 GB.
- Per 24-epoch holdout run, with validation every 4 epochs: the reference ≈ 1.05 h and the primary ≈ 1.27 h.
- On all of train: the primary ≈ 1.45 h. A1 should sit between the two; TerraMind + MoE-LoRA is not measured yet
  (≈ 1.5 h).
- Our queue then needs ≈ 17 GPU-h, ≈ 8.5 h on GPUs 1–2 from its 10:50 start, if nobody else takes them.

| prio | run (`run.sh` job) | config / data | seeds | h per run | GPU-h |
|---|---|---|---:|---:|---:|
| – | repro queue (done 01:24; not ours to manage) | ViT + LoRA, all of train (`dataset/repro`) | 3 | 1.23, 1.12, 1.11 (measured) | 3.46 |
| – | smoke tests 1–4 (through `gpu_run.sh --max-wait 10800`) | `--limit 64 --epochs 1` | – | – | ≈ 0.3 |
| 1 | `ref_s{0,1,2}` | ViT + LoRA, root `train.py`, holdout | 3 | ≈ 1.1 | 3.3 |
| 2 | `P_s{0,1,2}` | primary WideLoRA-SP, holdout | 3 | 1.2–1.4 | 3.9 |
| 3 | `Pfull_s0` | primary, all 9,392 images (`dataset/repro`): the submission model if C* = primary | 1 | ≈ 1.45 | 1.45 |
| 4 | `tm_s{0,1,2}` | TerraMind + MoE-LoRA, root `train.py`, holdout | 3 | 1.4–1.8 | ≈ 4.8 |
| 5 | `A1_s{0,1,2}` | A1, WideLoRA alone, holdout | 3 | 1.15–1.3 | ≈ 3.75 |
| – | JD caching, predict + check val (later test) | – | – | – | ≈ 0.2 |
| – | only if C* ≠ primary: C* on all of train | – | 1 | ≈ 1.1–1.6 | ≈ 1.5 |
| | **total** | | | | **≈ 21 (+ 1.5)** |

- **Job order** in `run.sh`: `ref_s0, P_s0, ref_s1, P_s1, ref_s2, P_s2, Pfull_s0, tm_s0, tm_s1, tm_s2, A1_s0, A1_s1,
  A1_s2`.
  - Priorities 1 and 2 are both required, so they are interleaved: each primary seed then runs next to its
    same-seed reference, for the kill rule and the gate.
  - If time runs short, I cut from the bottom (A1, then TerraMind) and tell you.
- **Concurrency.** One run per GPU, on GPUs 1 and 2. GPU 0 is skipped while unhealthy, and `gpu_run.sh` would take
  it back automatically once it is repaired.

**GPU contention found at 01:58 on Sat 24.** Another session is running `workspace/detail_lora/`'s holdout plan
(12 runs on its own `dataset/holdout/`; its queue started at 00:18) on GPUs 1 and 2. It uses `gpu_run.sh` and our
shared locks since 01:42, so the two queues never collide. At 05:33 it was on its 9th of 12 jobs (TerraMind
s0 on GPU 2, its A1 s1 starting on GPU 1), so it keeps both GPUs busy until ≈ 09:30 today, plus a full-train
DetailLoRA run if that session launches it. I didn't start it and won't touch it; **whether our queue simply waits
behind it is your call** (§10).

**Day by day** (SGT; it assumes approval on Sat 24 Oct morning, with the GPUs free once detail_lora's queue ends):

| day | GPUs 1–2 | lead |
|---|---|---|
| Fri 23 (done) | repro s0 (1.23 h) and s1 (1.12 h) | Steps 1–2 |
| Sat 24, to ≈ 10:00 | repro s2, 00:18–01:24 (done, 1.11 h). detail_lora's queue, 00:21–≈ 10:00 | Step 3 after approval: code + tests (≈ 3 h), then smoke tests 1–4 (they wait for a free GPU, ≤ 3 h) |
| Sat 24, ≈ 10:00–20:00 | `run.sh` on 2 GPUs: ref/P s0–s2 (≈ 4 h), Pfull + tm_s0 (≈ 1.6 h), tm_s1 + tm_s2 (≈ 1.6 h), A1_s0 + A1_s1 (≈ 1.3 h), A1_s2 (≈ 1.3 h) | check every ≤ 30 min, report state changes only |
| Sun 25 | (C* full-train run, if needed) | JD checks, decisions, predict val, `check_submission`, RESULTS.md, critic audit, **Checkpoint B** |
| Mon 26 – Fri 30 | buffer | slack for tom's or other sessions' jobs (one blocks a GPU entirely), reruns, the conditional C* run |
| Sat 31, 12:00 | | **deadline for the val zip** |
| 2–8 Nov | test predict | Step 6: test zip, upload advice |

Our plan needs ≈ 7 % of the GPU-hours in the window. The critical path is the implementation, and the risk is
other jobs holding the GPUs.

## 6. Submission rule

- **Model.** C* (§4, rule 4), trained on all 9,392 train images (`dataset/repro`'s train split is all of train):
  seed 0, 24 epochs, last epoch, the same protocol. It is decoded with its own JD setting (rule 3).
  - `Pfull_s0` trains it in advance for the primary, the expected winner. If C* is another configuration, it gets its
    own full-train run.
- **One model for both phases.** The same checkpoint makes the val zip (Phase 1) and the test zip (Phase 2).
  - Phase 2's two uploads are chosen at Step 6, from holdout and Codabench val evidence only.
- **Why:**
  - C* is chosen by the competition's own metric, on held-out scenes.
  - All of train adds 11 % more images, including the holdout's scenes and the buffer. That matters more if val/test
    share scenes with train (DATA.md §5).
  - A fixed seed and the last epoch mean nothing is selected for that model.
- **Fallback.** If the full-train run fails twice, C*'s holdout seed-0 checkpoint.

## 7. Rule check

| rule | how the design meets it |
|---|---|
| ≤ 130 M total, ≤ 40 M trainable | 115.96 M / 31.02 M; asserted at startup |
| No full fine-tuning, no scratch backbone | 10.5 M of the 86 M backbone parameters train (LoRA + partial fine-tuning); the backbone is pretrained |
| One model, no ensemble, no TTA | One forward pass of one jointly trained model. JD combines the two heads of that pass. Nothing is repeated or augmented at inference |
| Input ≤ 512, fed once, baseline geometry | Unchanged resize to 512 × 512. SP reads the same tensor in the same pass |
| Train only on train | All fitting on the local train split (holdout runs) or all of train (the submission). The holdout is part of train. No val/test pixels or metadata were read for any analysis |
| Generic pretrained only, named | timm `vit_base_patch16_224.augreg_in21k` (ImageNet-21k); TerraMind-1.0-base for the baseline; both cached, no downloads |
| Protocol | AdamW, batch 16, 24 epochs, 512 px; the extra lr group and JD are stated in §1 |

## 8. The critic's open objections (REVIEW.md), with their status

| # | objection | status |
|---|---|---|
| H1–H3 | The holdout leaked through flipped revisits (9 images), shifted tiles (up to 370 of 940 images) and 20 % strips at the detectors' gate (11 images) | **Resolved.** Re-drawn and frozen, with a one-pass local-train buffer. The critic **accepts** it: 0 crossing pairs under its independent searches, which recover 58 of 60 known overlaps. See [§9](#9-holdout-re-drawn-leak-free) |
| H-abs | A group-aware holdout is harder than val/test may be, so capacity-heavy designs are judged conservatively | Accepted. The holdout ranks configurations; leaderboard numbers will differ |
| H-few | Per-class holdout AP rests on 7–19 independent scenes for airport (7), wind turbine (9), oil tank (12), car (14), aircraft (16) and harbor (19) | Act on pooled metrics only |
| H-res | Residual leakage not searched: scale-changed crops, rotation + shift, overlaps without boxes | Open; stated in the report |
| D1-C1 | JD's toy study used ground truth of 18 holdout images | **Done.** Rerun on local-train images only, with an assertion. The product rule stays the default; the ordering held on all four split versions. The posterior rule wins when the box head confuses classes (99.6 vs 56.5 mAP on the toy), so it stays reachable through the cross-fit |
| D1-C2 | One decoding rule for cross-model comparisons | **Decided**: each model with its own JD setting, plus the JD-off numbers |
| D1-C4 / size | Local reference means in `report()`; the score threshold of 0.001 makes a ≈ 15 MB `detection.json` (≈ 143 k detections) | `compare.py` uses the local means. You may want to try one upload early |
| D2 | SP's expected gain is mostly below what 3 seeds resolve; its premise is weaker once the patch embedding trains | A1 (paired) is required; the point estimate decides (rule 4) |
| D2-gate | Measure time and memory on the GPU the seeds use; process peak = allocated + ~2.5 GiB | Rule 6 |
| W-C1 | The local reference through the new script for pairing | **Decided**: the reference stays the unchanged root `train.py`, so its comparisons are unpaired. Pairing applies to primary vs A1 only |
| W-prior | The expected Δm range is a prior; the learning rates are fixed a priori, untuned | Stated; the report will say so |
| X-1 | WideLoRA's near-one-hot classifier makes JD act almost hard | The per-model keep rule (rule 3) |
| X-2 | Predict must rebuild WideLoRA + SP from the `metrics.json` args | Covered by smoke test 3 |
| S | tom's jobs; hanliu's 17.6 GB on GPU 1; GPU 0 unhealthy; implementation is the critical path | Queue early; the gate (rule 6) |

## 9. Holdout (re-drawn, leak-free)

`dataset/holdout_sarmtl/` (data-eng; how it was drawn, verbatim assertion output: DATA.md §6).
- **Split:** 10 % of each class, seed 0, stratified by class × image size, deterministic. A rebuild gives
  byte-identical files.
- **Groups stay whole.** A near-duplicate group never spans the split. Its links cover:
  - pixel copies (including flipped and rotated ones);
  - revisits of the same site, including flipped ones;
  - overlapping and shifted tiles, including flipped + shifted ones and objects boxed at different sizes;
  - an all-pairs normalised shift search (`nshift.py`, the critic's method).

  That makes 40,787 links. An assertion checks that none crosses the split.
- **The giant wind-turbine group** (873 of 918 images) is held out as a connected block of tiles (47 images), plus
  45 images from whole groups. Its 87 direct neighbours form a buffer, excluded from local train only.
- **The holdout is frozen** (`workspace/sarmtl/data/frozen_holdout.json`, asserted by `make_holdout.py`).
- **One-pass buffer sweep.** Every detector ran over holdout × local train:
  - near-duplicate and audit links;
  - the raw-peak shift search;
  - the critic's normalised 32 px search;
  - a 64 px pass for 5–10 % overlaps, gated at high-pass NCC 0.5 by a null calibration. At 0.3 it would sweep 1,290
    images, mostly chance matches.

  Every local-train image found overlapping a holdout image moved to the buffer (303). So did the 5 remaining
  partners of the critic's last 33 flagged pairs (H3).
- **Final check:** `--check` finds 0 crossings under every detector. The critic's 8 regression pairs and its 33 H3
  partners are asserted off local train.
- **How it got here:** the critic found the flipped revisits, then the shifted tiles, then pixel-identical overlaps
  across the split, then 20 % strips at the detectors' gate. Each was fixed without loosening a threshold.
- **Read by** the unchanged root `train.py --data dataset/holdout_sarmtl`. `test → val`, so both splits score the
  holdout.
- **md5:** `val/labels.csv` 9915a68a…, `train/labels.csv` 0a6eb4d2…, `halves.json` d1b94180…. The lead re-checked
  the counts, disjointness, the 33 H3 ids and the symlinks.

| | images | boxes | independent units |
|---|---:|---:|---:|
| local train | 8,057 | 27,504 | 3,176 |
| holdout | 940 | 2,715 | 336 |
| buffer (local runs only; the submission run uses all 9,392) | 395 | 2,314 | – |

- **Per class (holdout / local train / buffer):**

  | class | holdout | local train | buffer |
  |---|---:|---:|---:|
  | aircraft | 118 | 1,011 | 53 |
  | airport | 65 | 585 | 0 |
  | bridge | 125 | 1,112 | 12 |
  | car | 44 | 390 | 9 |
  | harbor | 136 | 1,160 | 59 |
  | oil_tank | 123 | 1,101 | 9 |
  | playground | 118 | 1,056 | 1 |
  | ship | 119 | 1,031 | 37 |
  | wind_turbine | 92 | 611 | 215 |

  Holdout shares are 9.9–10.0 % of each class. Wind turbine loses 26 % of its non-holdout images to the buffer; every
  other class loses ≤ 5 %.
- **Few scenes.** Holdout units are airport 7, wind turbine 9, oil tank 12, car 14, aircraft 16 and harbor 19, so
  act on pooled metrics only. Wind-turbine AP is a test on a held-out block of a few large tiled wind farms.
- **Lead's decision: dark ships.** The 538 near-black 256² ship chips (45 % of ships) chain into one look-alike
  group. Only 2 of them are in the holdout, so holdout ship AP covers the brighter ship chips. This is leak-safe, and
  the bias is the same for every configuration.
- **Halves for JD's cross-fit** (`halves.json`): A 475 / B 465 images, with no link between them. The cross-fit
  decides on overall Δm, never per class.
- **Critic's re-verification of this final split: accepted** (REVIEW.md §5).
  - Integrity, all 40,787 links, strata, halves and the regression pairs pass.
  - Its independent same-site search and its normalised shifted-tile search (sensitivity 58 / 60) find **0 crossing
    pairs**.
- **What it implies:**
  - The holdout measures generalisation to new scenes. 73.0 % of train images share pixels or objects with another
    train image (3,497 independent units in 9,392 images). If val/test were drawn at random, ≈ 53 % of their images
    would have such a partner in train (DATA.md §5), and the leaderboard would reward fitting train more than this
    holdout does.
  - Expect lower absolute numbers than the TA's val/test, and a seed σ at the high end (≈ 1.3–1.5 Δm).

## 10. Decisions needed from you

1. **Approve** this design, the ablations, the decision rules (§4), the run plan (§5) and the submission rule (§6).
2. **JD and the rules.** JD uses the classification head's output to rescore the detections of the same forward
   pass. The critic and I read this as one model with no ensemble and no TTA; please confirm, or ask the TA. Without
   it, JD stays off and we lose an expected +0.6 to +2.9 Δm.
3. **GPU contention.** detail_lora's queue (another session) holds GPUs 1–2 until ≈ 10:00 today. Our queue can wait
   behind it (the default: both queues share `gpu_run.sh`'s locks), or you can pause that session.
4. **Holdout choices** (§9): a leak-free group-aware split, and holdout ships without the dark-chip source.
5. **Already built** during Steps 1–2 (design-independent, CPU-tested; no GPU used): `check_submission.py`,
   `compare.py`, `run.sh` and their tests, `21 passed`. I'll build the rest after approval.
