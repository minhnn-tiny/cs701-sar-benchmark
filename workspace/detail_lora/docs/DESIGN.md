# DetailLoRA: design proposal

Status: **approved 23 Oct 2026, implemented** (README.md has the commands). The data facts behind it: [DATA.md](DATA.md).

## Where the points are

Δm averages the relative change of macro-F1 and mAP, so 1 point of mAP (reference 26.1) is worth about 1.9 Δm,
and 1 point of F1 (93.3) about 0.5. ViT + LoRA → full fine-tuning moves F1 +1.8 and **mAP +6.4**. The gap is
detection, and 70 % of boxes are COCO-small. The reference leaves 19 M of the 40 M trainable budget unused
(LoRA uses 0.88 M), and its LoRA touches attention only, with a frozen RGB patch embedding.

## Primary design: ViT-B/16 (ImageNet-21k) + DetailLoRA

| # | change | why it should help | expected effect (estimate, not measured) |
|---|---|---|---|
| 1 | **Dense LoRA**: rank 32 on qkv, proj, fc1, fc2 of all 12 blocks; patch embedding, position embedding, CLS token and all LayerNorms unfrozen at 2e-5 (the TA's learning rate for pretrained weights) | most of the reference's gap to full fine-tuning is backbone capacity. The MLPs hold 2/3 of a block's weights and the reference does not adapt them; the first layer was never trained on SAR statistics or speckle | mAP +2 to +3, F1 +0.5 to +1 |
| 2 | **Detail branch**: 5 conv layers on the pixels (stride 4: 64 ch, stride 8: 128 ch), added to the ViTDet pyramid's two finest levels before their projection; zero-initialised output, so training starts from the reference model | the stride-4/8 levels are upsampled from 16 px tokens, but 18 % of boxes are < 8 px and 4 of the 9 AP classes are mostly < 16 px (ViT-Adapter's spatial prior, made light) | mAP +0.5 to +1.5 (small-object classes) |
| 3 | **Small anchors**: one √2 step below each level (5.7–181 px, 9 per location) | 4–6 px boxes: share with best anchor IoU < 0.5 drops from 92 % to 39 % | small, folded into #2 |
| 4 | **Class-guided detection** (inference only): box class = product of the RoI head's and the classification head's class distributions | every box of an image has the image's class. A small box's own class guess is noisy, and a wrong class is a false positive in one class's AP plus a miss in another's | mAP +0.3 to +1.5; zero parameters, zero training cost |
| 5 | **SAR augmentation** (training only): random transpose (with the existing flips: all 8 right-angle symmetries) and gamma γ ∈ [0.8, 1.25] | orientation-free overhead images; display stretch differs by source (DATA.md §4) | regularisation, ±0.5 Δm |

Unchanged: input geometry (resize to 512 × 512, global MEAN/STD), heads, losses (CE + Faster R-CNN), AdamW,
weight decay 0.05, batch 16, 24 epochs, lr 1e-4, 1 warmup epoch + cosine, bf16, last-epoch model.
Deliberate protocol changes: a third optimizer group (unfrozen pretrained tensors at 2e-5), and the
training-only augmentation of #5.

**Expected total**: Δm ≈ +6 to +12 against the local ViT + LoRA (central guess +8), so above the +3.8 of
TerraMind + MoE-LoRA. These are estimates from the mechanisms and the reference table, not measurements. The
3-seed holdout runs decide.

**Backbone: ViT, not TerraMind.** As adaptation capacity grows toward full fine-tuning, ViT is ahead
(full fine-tuning +13.5 vs +11.5; ViT + LoRA has the better F1). TerraMind's blocks also differ (gated MLP with
three linears, an S1-GRD dB patch embedding), so they would need their own adaptation.

## Parameters (computed by `detail_model.count_parameters`, random init, same architecture)

| model | total | trainable | of which: LoRA | unfrozen pretrained | detail branch | heads |
|---|---:|---:|---:|---:|---:|---:|
| reference ViT + LoRA (`sarbench`) | 107,039,257 | 20,997,913 | 884,736 | 0 | 0 | 20,113,177 |
| **primary DetailLoRA** | **111,216,584** | **26,198,984** | 4,718,592 | 1,023,744 | 339,616 | 20,117,032 |
| ablation A1 (no detail branch, reference anchors) | 110,873,113 | 25,855,513 | 4,718,592 | 1,023,744 | 0 | 20,113,177 |
| budget | ≤ 130,000,000 | ≤ 40,000,000 | | | | |

The trainable backbone is 5.74 M (reference 0.88 M; MoE-LoRA 1.03 M). The heads grow by 3,855 because the RPN
predicts 9 anchors per location instead of 6.

## Ablations (at most 2)

- **A1, training**: primary without the detail branch and small anchors (#2 + #3 off). Isolates the small-object
  changes from dense LoRA.
- **A2, inference only (free)**: detection scoring `none` / `hard` / `soft` on every checkpoint, the local
  baselines included. Isolates #4 at no GPU cost.

Not isolated (budget): dense LoRA vs augmentation. They are reported together as "primary − A1 − fusion".

## Time (L40S)

Measured: ViT + LoRA trains at 57 images/s on an L40S (the GPU 0 run of 23 Oct), so 1.15 h on all of train.
Estimates per run on the holdout (8,453 training images; 939 holdout images scored every 4 epochs):

| run | per run | runs | GPU-h |
|---|---:|---:|---:|
| ViT + LoRA (local Δm reference) | ~1.05 h | 3 | 3.2 |
| TerraMind + MoE-LoRA (best in-budget baseline) | ~1.5 h | 3 | 4.5 |
| primary DetailLoRA (~10–15 % slower than LoRA: +6 % FLOPs for dense LoRA, ~2 % for the branch) | ~1.2 h | 3 | 3.6 |
| A1 | ~1.15 h | 3 (1 if short of time) | 3.5 |
| primary on all of train, for submission | ~1.35 h | 1 | 1.35 |
| **total** | | 13 | **~16 GPU-h ≈ 8 h on GPUs 1 + 2** |

That fits before 1 Nov **only if GPUs 1 and 2 free up**. On 23 Oct 22:25 both ran `tom`'s jobs (31 and 35 GB,
~100 % utilisation, running 2 h), and `workspace/baseline/run_repro.sh` was already queued for them.

## Rule check

- Parameters within budget (asserted at startup). One model, one forward pass per image, input 512 × 512 as the
  reference, no TTA, no ensemble. Train split only; ImageNet-21k ViT-B/16 (timm `vit_base_patch16_224.augreg_in21k`)
  is the only pretrained model.
- #4 uses the classification head's output inside the detection post-processing of the same forward pass: one
  model, not an ensemble or TTA. Worth one line in the report.
- Inherited, unchanged: the reference resizes 256² images up to 512². "No upscaling" is read as "no input above
  512 px / no zoom". This is the reference's own geometry.
