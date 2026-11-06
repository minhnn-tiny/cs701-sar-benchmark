# Detection and small-object candidates (cv-detect)

Two candidates, ordered by expected Δm per GPU hour. Both are built on the unchanged `sarbench/` (subclasses only); the
prototype that defines them, and every number below, comes from the scripts in `workspace/sarmtl/scratch/cv-detect/`
(outputs in its `*_output.txt` files). Points are val-scale % points; Δm uses +1 mAP ≈ +1.91 Δm and +1 macro-F1 ≈
+0.54 Δm (val reference 93.31 / 26.13).

| | change | new trainable params | L40S time, 24-epoch holdout run | expected macro-F1 / mAP | expected Δm | cost of the evidence |
|---|---|---:|---|---|---|---|
| **D1** joint decoding (merged with cv-adapt's XFuse) | inference only: each RoI's class scores are multiplied by the classification head's p(class \| image) (the product rule, α = 1, fixed a priori); score threshold 0.05 → 0.001 (monotone); kept per model only if its paired gain is > 0 on all 3 holdout seeds; any other setting (posterior mode, D1b) only through a 2-fold cross-fit | 0 | no training | ±0 (D1b, if it passes: up to +0.5) / +0.3 to +1.5 | +0.6 to +2.9 (D1b: up to +0.3 more) | ≈ 0.01 GPU-h per checkpoint (one cached pass; every setting is a CPU re-decode), on checkpoints the plan trains anyway |
| **D2** spatial-prior branch | 6-conv branch on the pixels; its stride-4 and stride-8 maps are added as laterals in the ViTDet neck | 364,768 | ≈ +3–8 % over the same model without D2 (on ViT + LoRA ≈ 1.35–1.4 h vs 1.3 h); a GPU gate first | ±0.3 / on WideLoRA +0.5 to +2 (on LoRA +1 to +2.5) | on WideLoRA +0.8 to +4.0 (on LoRA +1.7 to +5.0) | ≈ 4.2 GPU-h for 3 seeds, plus 3 seeds of the no-D2 arm for the paired ablation |

**Pick: both.** D1 is free and works with any design: apply it from the first local reference checkpoint on, and to the
Phase 1 submission. D2 is my training-time candidate for the primary design. It combines with any cv-adapt adaptation
(the parameter counts add) and is decoded with D1.

**Holdout used for every check below:** data-eng's final, leak-free `dataset/holdout_sarmtl/`.
- The holdout is frozen: 940 images and 2,715 boxes.
- Local train has 8,062 images and 27,571 boxes (`train/labels.csv` md5 6cd34538…).
- The final leak sweep put 390 images in a buffer, in neither side: wind turbine 210, harbor 59, aircraft 53, ship 37,
  bridge 12, car 9, oil tank 9, playground 1.
- It is group-aware: data-eng's 965 near-duplicate groups cover 6,853 train images, and each is kept on one side.
- The cross-fit halves come from the same groups (`halves.json` md5 d1b94180…; critic C3).
- Holdout ships include only 2 of the 538 near-black 256² ship chips (lead decision), so holdout ship AP covers the
  brighter chips only.
- If the split changes again, rerun the CPU commands in section 6; nothing here caches it.

## 1. Facts the candidates rest on

Train split, boxes resized to the 512 × 512 input as in `sarbench/data.py`. Sources: `anchors_output.txt`,
`box_facts_output.txt`.

- **Sizes.** √area at 512 px: p10 6.9, p25 9.0, median 15.9, p75 47.0, p90 79.0 px. 18.4 % of boxes are < 8 px,
  50.5 % < 16 px (one ViT patch) and 62.4 % < 32 px. COCO "small" (< 32² original px) is 69.7 %.
- **Four classes are tiny.** Median √area: wind_turbine 8.3 px (94 % < 16 px, 46 % < 8 px), bridge 9.9 (83 % < 16 px),
  oil_tank 11.5 (58 %), playground 14.8 (65 %). The other five are larger: car 41, harbor 43, ship 55, aircraft 76,
  airport 146 px.
- **Localisation tolerance.** A 1 px error in both x and y fails IoU 0.75 on 43.8 % of the boxes (side < 13.5 px). A 2 px
  error fails IoU 0.5 on 35.7 % (side < 10.9 px). mAP averages IoU 0.50–0.95, so much of the metric needs box edges
  accurate to the pixel.
- **The labels themselves are only that accurate** (data-eng, DATA.md §5). In flipped copies, the boxes are offset by
  1 px (512² images) or 2 px (256², which is 4 px at the input) along each flipped axis. The two annotations of one object
  overlap at a median IoU of 0.67–0.80 (512²) and 0.50–0.69 (256²). So for the tiny classes, IoU ≥ 0.8 is largely out of
  reach for any model; their reachable AP lies at IoU 0.5–0.75 and in recall.
- **The tiny targets are faint or dark, not bright** (DATA.md §4). Wind turbines are only 1.69× their surroundings
  (38 % below 1.5×), in flat scenes (grey std 0.048). Airport, playground and bridge boxes are darker than their
  surroundings in 84 / 83 / 68 % of cases. Speckle is low in the 512² sources (CV 0.11–0.20).
- **The RoI head reads the stride-4 map for almost every box.** torchvision's `MultiScaleRoIAlign` sends a box to level
  ⌊4 + log2(√area / 224)⌋, clamped to strides 4–32. Every box under 112 px is therefore pooled from stride 4: 94.9 % of
  the GT boxes (stride 8: 3.9 %, stride 16: 1.2 %).
- **In the reference, the stride-4 map is two 2 × 2 transposed convs of the stride-16 tokens**
  (`SimpleFeaturePyramid.rescale[0]`). No detail finer than what a frozen ImageNet ViT keeps per 16 × 16 patch reaches
  the detector (LoRA adapts only attention).
- **The anchors are not the bottleneck.** With the reference anchors (8–181 px, 3 ratios), every box gets a positive
  through the matcher's low-quality matches. 89.5 % of boxes have an anchor at IoU ≥ 0.5 and 99.4 % at IoU ≥ 0.3.
  - Only the 4–8 px bin is weak (median best IoU 0.54). Adding a 5.7 px scale raises that median only to 0.57; 5
    aspect ratios change the coverage by < 1 pp.
  - Per class (`anchors_perclass_output.txt`), the weak ones are airport (28 % of boxes with best IoU < 0.5) and
    wind_turbine (22 %). 1:4 / 4:1 ratios bring airport to 17 % and leave wind_turbine unchanged. 3 scales per level
    bring them to 16 % and 13 %.
- **One class per image.** Every box has the image's class. Boxes per image: car 9.2 and wind_turbine 9.4 on average (max
  28 and 41); 1.3–4.5 for the other classes. The image classifier is 93–94 % accurate (reference val / test accuracy
  93.6 / 94.2), helped by source cues confounded with class (DATA.md §8: car is one sharp source, ships 97.5 % 256²).
- **The holdout is small in independent scenes:** 336 units in its 940 images, of which airport 7, wind_turbine 9,
  oil_tank 12, car 14, aircraft 16 and harbor 19 (DATA.md §0.9). Per-class holdout AP of these classes rests on a few
  scenes, so every comparison below is paired on the same holdout, and early signals use pooled metrics.
- **The adaptation gap is mostly detection.** From LoRA to full fine-tuning on val, mAP gains +5.9 (26.1 → 32.0) and
  macro-F1 +1.1.

## 2. D1: joint decoding of the two heads (inference only)

D1 merges my image-class prior with cv-adapt's XFuse, agreed with cv-adapt. XFuse is dropped from `cand_adapt.md`; its
reverse step is D1b below, and its product rule is now D1's a priori rule (see "Why the product"). data-eng reached the
same rescoring idea independently (DATA.md §8).

**The change.** Prototype: `detect_modules.py`, `JointDecodeRoIHeads` and
`DetectModel(prior_alpha, prior_mode, evidence_lambda, score_thresh, nms_thresh)`.
- **Where.** `RoIHeads.postprocess_detections`, in eval mode only. `DetectModel` passes the classification logits of the
  same forward pass. Both steps happen before torchvision's score threshold, per-class NMS and top-100 cut, which then
  run unchanged, so the 100 cut never needs lifting.
- **Inputs.** Each RoI has softmax scores s_bg, s_1…s_9; the image has p = softmax(classification logits).
- **Step 1, D1b (λ), the image class from both heads.** This is cv-adapt's reverse step.
  - log p' = log_softmax(log p + λ m), where m_c is the highest raw score of class c over the image's RoIs.
  - arg max p' is the class output.
- **Step 2, the image class as a prior on every RoI.** Two modes:
  - **product** (the a priori rule, cv-adapt's XFuse form): score'_c = s_c · p'_c^α. The score removed goes to the
    background.
  - **posterior** (an alternative): score'_c = (1 − s_bg) · s_c p'_c^α / Σ_k s_k p'_k^α. The RoI's objectness is kept,
    and its class mass moves towards the image class.

  Both are written as new logits for torchvision's postprocess. α = ∞ means p'^α = 1 for the arg-max class and 0 for
  the others. λ = α = 0 reproduces the reference exactly.
- **Why the product, a priori.** Only the image's own class is annotated. Other-class objects are present but unboxed:
  ships in ~8 of 12 sampled harbor scenes, piers next to oil tanks, aircraft tiles inside unboxed airports (DATA.md
  §8). So P(a box is a true positive of class c) ≈ P(it is a class-c object) · P(the image is class c) = s_c p_c.
  - The product never raises a score and never moves mass to another class, so it cannot create a false positive.
  - The posterior assumes every object shares the image's class. With a confident classifier, it relabels a real
    other-class object as an image-class false positive at the object's full objectness.
  - The trade-off, on synthetic RoI outputs built from the ground truth of 18 local-train images, where the right
    answer is known. mAP, mean of 5 draws (`d1_analysis.py --selftest`, output in `d1_analysis_selftest.txt`).
  - **The rule choice used no holdout data** (critic C1). The 18 images are 2 per class from the final local train
    (`train/labels.csv` md5 6cd34538…; ids 1, 2, 3, 4, 5, 7, 9, 11, 17, 19, 24, 27, 29, 35, 37, 40, 41, 44). The
    self-test asserts that none is in the 940-image holdout. No model output was used either.
  - The rerun on each re-draw of the split (00:28, 03:19, 04:39 and the final sweep) gave the same ordering every
    time.

    | scenario | p of the classifier | reference | product α = 1 | product hard | posterior α = 1 | posterior hard |
    |---|---:|---:|---:|---:|---:|---:|
    | clean (only background FPs) | 0.95 / 0.999 | 92.0 | 92.0 | 92.0 | 92.0 | 92.0 |
    | RoI head unsure between the right class and a confuser | 0.95 | 42.5 | 56.5 | 56.5 | 99.3 | 99.6 |
    | | 0.999 | 42.5 | 56.5 | 56.5 | 99.6 | 99.6 |
    | 3 unannotated other-class objects per image | 0.95 | 62.0 | 91.3 | 91.3 | 90.7 | 57.3 |
    | | 0.999 | 62.0 | **91.3** | 91.3 | **65.9** | 57.3 |
    | classifier confidently wrong on 1 image in 3 | 0.95 | 92.0 | 75.1 | 62.7 | 86.4 | 53.2 |
    | | 0.999 | 92.0 | 62.7 | 62.7 | 71.7 | 53.2 |

  - The product gains in both failure modes, at either confidence. The posterior gains more on confusion, but loses
    all of its gain on unannotated objects once p is near one-hot, which is the regime of a model with train CE ≈ 0.03
    (cv-adapt's WideLoRA).
  - Neither protects images whose class is predicted wrong. At the real ~6 % rate (not 33 % as here), that cost is what
    the adopt rule below measures.
  - The magnitudes are toys; the table only fixes the direction of each trade-off.
- **The fixed setting**, chosen before any result: product mode, α = 1, λ = 0, `score_thresh` 0.001, `nms_thresh` 0.5,
  `detections_per_img` 100.
  - `score_thresh` 0.001 instead of 0.05 only appends detections scored below every kept one, at the tail of each
    class's ranking (the top-100 cut is by score), so COCO's interpolated AP cannot go down.
  - It matters for the product rule, which lowers scores: true positives in uncertain images would otherwise fall under
    0.05.
- **Protocol** (the lead's decision; the critic's round 1 added re-applying the rule per model and the analyses).
  1. **No tuning.** Score the fixed setting paired: same checkpoint, same images, decoding on vs off.
     - On is (product, α = 1, 0.001); off is the reference (α = 0, 0.05). Also report (α = 0, 0.001), so that the
       threshold's share and the prior's share of the gain are separate.
     - Seed noise cancels in a paired difference.
  2. **Adopt or not, by a rule stated now, applied per model.** Keep the prior if its own paired gain, (product, α = 1,
     0.001) vs (α = 0, 0.001), is > 0 in Δm on all 3 holdout seeds of that model.
     - Apply it first to the local reference (ViT + LoRA). Apply it again to the design's 3 holdout seeds before the
       design's submission: the design's classifier is near one-hot, which changes what the prior does. Report both.
     - If a model fails the rule, it is decoded with α = 0 and `score_thresh` 0.001. This one binary choice per model
       is the only selection made on the holdout.
  3. **Measure the downside** (analysis, `d1_analysis.py report()`; CPU re-decode of the cached RoI outputs):
     - the paired per-class AP change;
     - the prior applied to the correctly classified images only vs to all images (an oracle split, analysis only). The
       gap is the cost of misclassification;
     - the mean max-softmax of the wrongly and of the correctly classified images.
  4. **Any other setting only through a 2-fold cross-fit.** The other settings are the posterior mode, α ∈ {0.5, 2, ∞},
     λ ∈ {1, 2, 4} and `nms_thresh` 0.6; they also go into the report as sensitivity.
     - The halves are data-eng's `dataset/holdout_sarmtl/halves.json`: A has 475 images and B 465, stratified by class,
       and no near-duplicate unit straddles them or is linked across them (asserted).
     - Some classes rest on very few scenes per half. Units in A / B: airport 1 / 6 (A is one 36-image unit),
       wind_turbine 1 / 8 (A is one 47-image block), oil_tank 5 / 7, aircraft 7 / 9, harbor 9 / 10. So the cross-fit is
       decided on the overall paired Δm, never per class.
     - Choose on A and score on B, then the other way round. Switch only if both folds choose the same setting and its
       out-of-fold paired gain is > 0 on all 3 seeds; the out-of-fold gain is then the reported number.
     - D1b (λ > 0, credited to cv-adapt) can only be adopted this way.
  5. **Freeze before each submission.** Never tune on the leaderboard.
  6. **Comparing models** (critic C2), e.g. the design vs TerraMind + MoE-LoRA. Score each model with its own
     pre-registered decoding, since that is what would be submitted: product α = 1 if the model passes step 2, else
     α = 0 at 0.001. Report the D1-off numbers of both models alongside. This replaces "both with the same D1 setting";
     the lead settles it in REVIEW.md.
  7. **Δm denominators** (critic C4). Every holdout Δm, paired or not, uses the local reference's 3-seed holdout means
     (ViT + LoRA on `dataset/holdout_sarmtl`) as `ref`. The val reference (93.31 / 26.13) is only a self-test
     placeholder.

  During training, validation uses the reference decoding, so the learning curves stay comparable.

**Expected effect.**
- **mAP:** +0.3 to +1.5 (Δm +0.6 to +2.9).
- **macro-F1:** unchanged at λ = 0. D1b could add 0 to +0.5 (Δm 0 to +0.3) if it passes the cross-fit; I don't count on
  it.
- **Mechanism.** In this data, a box whose class differs from the image's class is a false positive by construction.
  Yet torchvision emits a class-specific copy of a proposal for every class scoring > 0.05.
  - The RoI classifier picks among 9 classes from a 7 × 7 RoI on the stride-4 map. Four classes are tiny faint spots or
    dark shapes of similar size (8–15 px median), which such a window can hardly tell apart.
  - The scene-level classifier is 93–94 % right.
  - The product scales every wrong-class copy by p(that class | image), which is small for 93–94 % of images. That
    removes cross-class false positives from every class's ranking.
- **Where.**
  - The four tiny classes (4 of the 9 terms of mAP).
  - Dense images (car and wind_turbine: 9 boxes per image, up to 41), where wrong-class copies also compete for the
    top-100 slots.
  - Ship, harbor and airport, through a second source of cross-class false positives: visible but **unannotated**
    objects of other classes. Ship gains from ships in harbor scenes, harbor from piers next to oil tanks, airport from
    aircraft tiles inside airports.

  On that second source (DATA.md §8; contact sheets `dataset/holdout_sarmtl/analysis/contact_<class>.png`, eyeballed on
  12 images per class, not quantified): the detector learns these objects as background in their own images. Where it
  still fires on one with, say, a ship box, that box is a false positive of the ship class, and the product scales it by
  p(ship | harbor scene).
- **Why not more, or less.** ViT tokens carry global context through attention, so the RoI head may already use some
  scene information; that gives the lower end. The product can't rescue a true object that the RoI head scored mostly
  as another class; the posterior could. If the cross-fit shows that case dominates, the posterior is the route to the
  upper end.
- **Downside.** On the ~6 % of images whose class is predicted wrong, every right-class box is scaled by the small
  p(right class) and drops to the tail of its class's ranking. Protocol step 3 measures this cost, and step 2 decides on
  it.

**Parameters.** The same as the model it decodes. ViT + LoRA: 107,039,257 total; 20,997,913 trainable = heads
20,113,177 + backbone adaptation 884,736 + new 0 (`params_output.txt`).

**Time.**
- **Training:** none.
- **Evaluation:** one holdout pass per checkpoint caches its raw RoI outputs, ≈ 0.5 min on an L40S (`d1_analysis.py
  collect`, 940 images, ≈ 1,000 RoIs × 54 floats per image ≈ 0.2 GB). Every setting, the analyses and the cross-fit are
  then CPU re-decodes of that cache: ≈ 0.01 GPU-h per checkpoint, ≈ 0.05 GPU-h for the 3 reference and 3 design seeds.
- **Inference cost:** about nothing (a softmax, a max and a log over ≤ 1,000 × 9 logits per image).

**Risk.**
- **Images whose class is predicted wrong.** Discussed above. Measured (step 3) and decided on (step 2) per model.
- **Selection bias.** None for the fixed setting: nothing is chosen on the holdout except the stated per-model
  adopt-or-not rule. Any other setting has to pass the 2-fold cross-fit, which scores each choice on images it wasn't
  chosen on.
- **Noisy classes.** The holdout has 2,715 boxes. Airport has 83 boxes in 65 images but only 7 independent scenes, and
  wind_turbine 9. Judge by the paired 3-seed values, not by one class.
- **Transfer to val/test.** p comes from the model itself, and the setting is global and fixed a priori. Nothing about
  val/test is used.
- **Fallback:** α = λ = 0 (the reference), keeping `score_thresh` 0.001.

**Rule check.**
- **One model, no ensemble, no TTA.** One forward pass of one model on one 512 px image. D1 combines the two outputs of
  that pass; nothing is augmented or repeated.
- **Data.** The setting is fixed a priori; the local holdout (part of train) only measures it, and any deviation is
  cross-fitted on holdout halves. Nothing is fitted on val/test pixels, metadata or leaderboard feedback.
- **Parameters.** 0 new; the budget is unchanged.
- **Precision rule.** D1 is computed on fp32 logits inside the fp32 RPN/RoI block.
- **Report.** We state it as the model's decoding rule, from the stated data fact that all boxes share the image's
  class.

## 3. D2: a spatial-prior conv branch into the stride-4 and stride-8 levels

**The change.** Prototype: `detect_modules.py`, `SpatialPrior`, `FusedPyramid` and `DetectModel(spatial_prior=True)`.
- **`SpatialPrior`** runs on the normalised input image [B, 1, 512, 512], the same tensor the ViT gets. Every conv has no
  bias and is followed by GroupNorm (16 groups) and ReLU.
  - `to_s4`: conv3×3 s2 1→32, conv3×3 s2 32→64, conv3×3 64→64, conv3×3 64→64. Output C4 [B, 64, 128, 128]: stride 4,
    receptive field 23 px.
  - `to_s8`: conv3×3 s2 64→128, conv3×3 128→128. Output C8 [B, 128, 64, 64]: stride 8, receptive field 47 px.
- **Fusion.** `FusedPyramid` wraps the reference neck that `MultiTaskModel` built, so it reuses the same modules, the same
  init and the same state_dict keys. At strides 4 and 8 only:

  P_k = GN(Conv3×3( GN(Conv1×1(rescale_k(tokens))) + GN(Conv1×1_k(C_k)) )).

  This is an FPN-style lateral (1×1 conv to 256 channels + GroupNorm(32)), added after the level's 1×1 projection and
  before its 3×3 conv. Unchanged: strides 16, 32 and 64, the RPN, the RoI heads, the anchors and the classification
  head.
- **Forward.** `DetectModel(MultiTaskModel)` overrides forward with the reference forward plus one change: it passes the
  images to the neck. The precision rule stays: the neck runs under bf16 autocast, and the features are cast to fp32
  before the RPN and RoI heads.
- **Initialisation** (fixed a priori, critic D2-d). No pretrained weights.
  - The branch's convs use PyTorch defaults, as for the reference neck.
  - The lateral GN weight starts at 0, and its bias is 0. So at init the lateral adds exactly 0, and D2 starts as the
    reference function. At step 1 only the lateral GN gets gradients; from step 2 on, the branch does
    (`smoke_output.txt`, check 2).
  - D2's new modules are initialised from their own seed (run seed + 10,000) inside `torch.random.fork_rng`. The global
    RNG is therefore left exactly where the run without D2 leaves it: common random numbers. With the same seed, every
    shared module gets the same init, and the training losses and detections at init are bit-identical
    (`smoke_output.txt`, check 5).
- **Training.** The D2 parameters (`neck.prior.*`, `neck.lateral.*`, which don't start with `backbone.`) fall into
  `train.py`'s existing head group. They train like every other head parameter: lr 1e-4, AdamW wd 0.05, one warmup
  epoch then cosine, batch 16, 24 epochs, last-epoch model. The backbone adaptation is whichever adaptation candidate is
  chosen; the counts here use ViT + LoRA r16. Inference is unchanged, apart from D1's decoding.
  - For the paired ablation, both arms (with and without D2) run from the same train script. Its training DataLoader
    gets its own `torch.Generator().manual_seed(seed)`, so the data order and the worker augmentation seeds are the
    same in both arms.
  - What still differs between the two arms after step 1: the RoI sampler's draws (its pool depends on the proposals)
    and GPU non-determinism.

**Expected effect** (critic D2-b: the base matters).
- **On top of ViT + LoRA:** mAP +1 to +2.5 (Δm +1.9 to +4.8 from mAP alone).
- **On top of cv-adapt's WideLoRA,** the design base: mAP +0.5 to +2 (Δm +1.0 to +3.8 from mAP alone).
  - WideLoRA trains the patch embedding, pos_embed and every MLP (LoRA r64).
  - The patch embedding is an injective 256 → 768 linear map, so sub-patch detail does enter the ViT. With trained MLPs,
    more of it can survive to the last block.
  - D2 and WideLoRA therefore partly overlap, and D2's gain on WideLoRA should be smaller.
- **macro-F1:** ±0.3. The classification path is untouched; only the shared backbone gradients shift. With it, Δm on
  WideLoRA is +0.8 to +4.0.
- **Mechanism.**
  - 95 % of RoIs are pooled from the stride-4 map. In the reference, that map only upsamples what the ViT keeps per
    16 px patch. Yet 50 % of the boxes are smaller than one patch, and a 2 px error already fails IoU 0.5 on 36 % of
    them.
  - The conv branch gives the detector a direct, shift-equivariant path from the pixels (overlapping 3 × 3 windows, no
    patch boundaries), whatever the backbone keeps.
  - It can learn local-contrast filters for what the tiny classes look like: faint bright points (wind turbines at
    1.69× their surroundings, oil tanks) and dark structures (playgrounds, bridges, runways, darker than their
    surroundings in 68–84 % of boxes).
  - The GroupNorm after each conv normalises per image, so it acts as a local contrast normalisation on the flat,
    dark wind-turbine scenes.
  - Speckle is low in the 512² sources, so the branch is about contrast and edges, not despeckling.
- **Where.** The gains should show at IoU 0.5–0.75 (AP50, AP75) and in the recall of the four tiny classes. Above
  ≈ 0.8, the 1–2 px offsets between annotations of the same object (section 1) cap any model on those classes, so I
  don't count on AP above 0.8 there.
- **Why this range.**
  - Full fine-tuning gets +5.9 mAP by adapting the whole backbone. D2 addresses only spatial detail, so it should
    recover a fraction of that gap. The fraction is smaller on WideLoRA, which already adapts the patch path.
  - It is the TA's suggested direction: finer maps from a light conv branch, as in ViT-Adapter's spatial prior on a
    ViTDet pyramid.
  - The label-noise cap lowers the top end.
- **Combined with D1.** D1 acts on class assignment, D2 on features (recall and localisation). I expect most of D1's
  gain to persist on a D2 model, but not exactly additively: better tiny-object features may also reduce the RoI head's
  class confusion, which is D1's headroom. D1 is measured paired on every model anyway (section 2), so no additivity is
  assumed in any number.
- **Pre-registered evaluation** (critic D2-a).
  - **Arms:** the adaptation alone vs the adaptation + D2, with the same 3 seeds and common random numbers (above).
  - **Primary endpoint:** holdout mAP at the last epoch. It is reported as the per-seed paired difference (D2 arm minus
    no-D2 arm, same seed) and its 3-seed mean ± std, plus Δm with the local reference's means as denominators.
  - **Secondary endpoint:** the mean AP of the four tiny classes (bridge, oil_tank, playground, wind_turbine), paired the
    same way. AP_small (COCOeval `stats[3]`) and AP50 / AP75 are descriptive only.
  - **Decision for Phase 2 and the report:** keep D2 if the 3-seed mean of the primary paired difference is > 0 and the
    secondary's is > 0. Otherwise the design is the adaptation alone. Both are decided on the holdout before the Phase 2
    submission.
  - **Power, stated honestly.** With 3 seeds and σ ≈ 0.9–1.1 Δm, only differences of ≈ 2–2.5 Δm reach p < 0.05; my low
    end (+0.5 mAP ≈ +1 Δm on WideLoRA) can't be shown significant. A paired t-test (n = 3) is reported, and the
    result is called significant only if p < 0.05. Common random numbers should shrink σ of the difference, and the
    report will state the measured value.

**Parameters** (`params_output.txt`):

| model | total | trainable | heads | backbone adaptation | new (D2) |
|---|---:|---:|---:|---:|---:|
| ViT + LoRA (reference) | 107,039,257 | 20,997,913 | 20,113,177 | 884,736 | 0 |
| ViT + LoRA + D2 | 107,404,025 | 21,362,681 | 20,113,177 | 884,736 | 364,768 |
| ViT + LoRA + D2-wide (C4 128, C8 256; fallback if under capacity) | 108,357,241 | 22,315,897 | 20,113,177 | 884,736 | 1,317,984 |
| TerraMind + MoE-LoRA + D2 | 106,857,977 | 21,510,137 | 20,113,177 | 1,032,192 | 364,768 |
| cv-adapt C1 (wide LoRA r64 + unfrozen norms/biases, cv-adapt's count) + D2 | 115,956,473 | 31,021,817 | 20,113,177 | 10,543,872 | 364,768 |

D2 doesn't touch the backbone, so the counts add: the last row is cv-adapt's C1 (115,591,705 / 30,657,049) plus D2's
364,768.

**Time and memory.**
- **FLOPs per 512 px image.** CPU `FlopCounterMode`, matmul and conv only; attention isn't counted, so the true ratios
  are slightly lower.
  - Eval forward: 266.1 → 271.8 GFLOPs (×1.021). The neck goes from 35.8 to 41.5.
  - Training forward + backward: 579.0 → 596.0 (×1.029).
- **Wall time.** The branch's layers have few channels and are memory-bound (the largest tensor is 16 × 32 × 256 × 256 per
  batch), so wall time grows a little more than FLOPs. Estimate +3–8 % over the same model without D2:
  - on ViT + LoRA, ≈ 1.35–1.4 h per 24-epoch holdout run (reference 1.3 h at 47–51 images/s), ≈ 1.5 h on all of train,
    ≈ 4.2 GPU-h for 3 seeds;
  - on WideLoRA, the same +3–8 % on cv-adapt's estimate for WideLoRA alone.

  This is an estimate from FLOPs plus memory-bound reasoning, not a GPU measurement; the gate under Risk measures it.
- **Memory.** Activations saved for backward at batch 16, measured on the CPU under bf16 autocast (`memory_output.txt`):
  9.26 → 9.81 GiB (+0.55 GiB). This is indicative only: CPU and CUDA autocast cast different ops (on CUDA, GN runs in
  fp32 and conv in bf16). The gate measures the real peak.
- **Inference:** ≈ +2 % per image.

**Risk.**
- **Slower or bigger than estimated** (cuDNN on 1–32-channel convs, GroupNorm at 256²). A gate before any D2 seed
  launches (critic D2-e):
  - a GPU smoke test through `gpu_run.sh`: ~100 training iterations each of WideLoRA + D2 and WideLoRA alone (plus
    ViT + LoRA for the record), on the GPU the seeds will use, logging s/iteration and the process's peak memory;
  - process peak, not only tensors: `torch.cuda.max_memory_allocated()` + ≈ 2.5 GiB of context, or the process's
    nvidia-smi figure (critic C5);
  - pass: ≤ +10 % time per iteration and a process peak below the GPU's free memory. GPU 1 has ≈ 27.9 GB free next to
    hanliu's 17.6 GB app, so the bar there is < 27 GB.
  - On a fail, switch to the pre-registered fallback before any seed starts: a 4 × 4 stride-4 "patchify" first layer
    (1 → 64) in place of the two stride-2 convs, which removes the 256² tensors. Then re-test.
- **No gain** (the branch goes unused, or the RoI head isn't limited by resolution). The kill rule, stated now (critic
  D2-c):
  - **Comparator:** the adaptation-alone arm with the same seed (common random numbers), not the local reference.
    Different architectures with the same seed are otherwise not paired.
  - **Stop a D2 seed only on gross failure:** ≥ 2 mAP behind its comparator at the epoch-8 holdout evaluation on both of
    the first two seeds; or divergence, meaning a non-finite loss, or a total loss above 2× the comparator's for 200
    consecutive iterations.
  - If no comparator arm runs alongside, only divergence stops a run.
  - The zero init makes the two arms identical at step 0 only; after that they drift apart within seed noise. The
    gross-failure threshold (≥ 2 mAP on both seeds), not the init, is what keeps this rule from killing a branch that
    is only warming up.
  - What to look at, without acting on it: the pooled numbers should move first (AP_small and the mean AP of the four
    tiny classes); a single class is too noisy (wind_turbine rests on 9 independent scenes). Also the mean |GN weight|
    of the two laterals per epoch: if it stays ≈ 0, the branch is unused.
- **Instability** (loss spikes from the new high-resolution path). Designed out, with no reactive fallback:
  - the lateral starts at exactly 0 (zero-init GN weight), and GN on both paths keeps their scales matched;
  - the init is fixed now, not switched mid-experiment.
- **Precision rule:** unchanged. The anchors, box coder and RoI heads stay in fp32.
- **Shared budget:** +0.36 M only, so it leaves cv-adapt's room as it is.

**Rule check.**
- **Parameters.** 107.4 M total ≤ 130 M and 21.4 M trainable ≤ 40 M with ViT + LoRA. With another adaptation, add its
  count.
- **One model with one shared backbone.** The ViT stays the only backbone, shared by both heads. D2 is part of the
  detection neck: 6 convs, 0.36 M params (0.3 % of the model), ~2 % of the FLOPs. It is the "light conv branch" the TA
  lists under small objects, and new necks and heads are allowed.
- **Input ≤ 512 px, fed once.** The branch reads the same 512 × 512 tensor in the same forward pass. No tiling, no
  upscaling.
- **Data.** Trained from scratch on the train split, with no pretrained weights and nothing from val or test.

## 4. Order and pick

1. **D1.** Δm +0.6 to +2.9 for ≈ 0.01 GPU-h per checkpoint: one cached holdout pass, after which every setting and
   analysis is a CPU re-decode. Run it on the 3 local reference checkpoints the plan trains first, as soon as they
   exist, then on the design's seeds. It also applies to the Phase 1 submission model.
2. **D2.** Δm +0.8 to +4.0 on WideLoRA (+1.7 to +5.0 on LoRA) for ≈ 4.2 GPU-h over 3 seeds. Its own evidence needs
   the 3 seeds of the no-D2 arm too, paired by common random numbers.

**My pick: both.** D1 changes the decoding and D2 the features, and each is measured on its own: D1 paired on every
checkpoint, D2 against its no-D2 arm.
- **Primary design:** cv-adapt's adaptation pick + D2, decoded with D1.
- **Ablations:**
  - without D2 (the adaptation alone, 3 seeds, same seeds and common random numbers). This is D2's pre-registered
    comparison, and it decides D2 for Phase 2;
  - D1 off: free, since it uses the same checkpoints with the decoding switched off.
- **Before any D2 seed:** the GPU gate (section 3, Risk).

## 5. Considered and not proposed

- **New anchors.** data-eng suggests 1:4 / 4:1 ratios for airports and sizes below 8 px (DATA.md §8). The gains are
  measurable but small (section 1, `anchors_perclass_output.txt`):
  - 1:4 ratios: airport's share of boxes with best IoU < 0.5 goes from 28 % to 17 %;
  - 3 scales per level: airport goes to 16 % and wind_turbine from 22 % to 13 %.

  In this two-stage detector every box already gets a positive anchor, and the RoI head refines the box. So this is a
  training-time change of uncertain value. It would be a cheap later ablation: about 3 k parameters and 1.5× the
  anchors. I don't bundle it with D2, which would confound D2's ablation.
- **NWD or RFLA label assignment for tiny objects.** Their large gains are reported on AI-TOD, with a baseline whose
  smallest anchor is 32 px; ours is already 8 px. The IoU thresholds would also need retuning, a risk with a week left.
- **Cascade R-CNN.** Two more box heads at 13.9 M each would take most of the shared 19 M of headroom, or need slimmer
  heads, and it means new code. Its high-IoU gain is capped by the feature resolution, which D2 addresses first, and, on
  the tiny classes, by the 1–2 px label offsets.
- **A stride-2 level.** Its RPN conv and neck conv would add ≈ 2 × 39 GMACs per image, about +50 % of the model's
  forward.
- **RoI-to-level mapping** (a smaller canonical scale) and **context-enlarged RoIs.** Both are free, but they are
  training-time knobs with an unclear sign: ablation material for later.

## 6. Implementation notes and reproduction

- **The prototype is the spec.** `workspace/sarmtl/scratch/cv-detect/detect_modules.py` passes five CPU checks
  (`smoke_output.txt`):
  1. with D2 off and α = λ = 0, `DetectModel` reproduces `MultiTaskModel`'s outputs bit for bit from the same state_dict,
     so the reference checkpoints load into it for D1;
  2. D2 trains under bf16 autocast with finite losses. At step 1 only the zero-init lateral GN gets gradients; from
     step 2 on, the branch does;
  3. D1 (product, α = 1, 0.001) moves the detections to the predicted image class and keeps their number;
  4. on the RoI head alone:
     - posterior mode: a uniform prior reproduces the reference, and a one-hot prior gives the image class the RoI's
       objectness;
     - product mode: a one-hot prior keeps only the image class, at its raw score, and no score ever goes up;
     - D1b with a large λ predicts the class of the image's top box;
  5. common random numbers: built after the same seed, the D2 model leaves the global RNG where the no-D2 model does,
     shares every reference parameter bit for bit, and gives identical training losses and detections at init.
- **D1 tool:** `d1_analysis.py`.
  - `collect(model, dataset)` makes one holdout pass and caches the raw RoI outputs (on the GPU through `gpu_run.sh`).
  - `report(cache, heads, gt_json, ref, halves)` covers the whole section 2 protocol on the CPU: the paired decision,
    the threshold and prior shares, per-class AP changes, the oracle split, max-softmax, the sweep and the 2-fold
    cross-fit.
  - Its self-test (`d1_analysis_selftest.txt`) also checks the scoring: synthetic perfect detections score mAP 1 under
    every decoding.
- **Train script** (`workspace/sarmtl/`):
  - build `DetectModel(backbone, routing, spatial_prior=True, prior_seed=seed + 10_000)`, and keep the loop of root
    `train.py`;
  - give the training DataLoader `generator=torch.Generator().manual_seed(seed)`, in both arms;
  - store `spatial_prior`, `prior_channels` and `prior_seed` in the `metrics.json` args, so that predict can rebuild
    the model;
  - log the mean |GN weight| of the two laterals per epoch.
- **Predict.** Build `DetectModel(..., prior_alpha=1, prior_mode='product', evidence_lambda=0, score_thresh=0.001)`.
  These are inference flags only. With λ > 0, the returned classification logits are log p', which gives the class
  output.
- **Reproduce** (CPU, from the repo root):

```bash
uv run python workspace/sarmtl/scratch/cv-detect/anchors.py      # anchor / RoI-level coverage (~3 min, 16 threads)
uv run python workspace/sarmtl/scratch/cv-detect/anchors.py --configs ref,ref+s4,ratios1:4   # + per class
uv run python workspace/sarmtl/scratch/cv-detect/box_facts.py    # IoU tolerance, per-class sizes and boxes/image
HF_HUB_OFFLINE=1 uv run python workspace/sarmtl/scratch/cv-detect/params.py --flops   # counts + FLOPs
HF_HUB_OFFLINE=1 uv run python workspace/sarmtl/scratch/cv-detect/memory.py           # saved-activation memory
HF_HUB_OFFLINE=1 uv run python workspace/sarmtl/scratch/cv-detect/smoke_cpu.py        # the five checks
HF_HUB_OFFLINE=1 uv run python workspace/sarmtl/scratch/cv-detect/d1_analysis.py --selftest   # D1 tool + synthetic table (local-train images)
```
