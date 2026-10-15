# Reliability iteration (Kaggle v3)

**RETIRED: the user clarified that labels are for final evaluation only after
v3 was submitted. The supervised training and train-label-fitted calibration below
do not satisfy that requirement. Do not use v3 to support the intended hypothesis.
The current default configuration blocks these training entry points before reading
labels. A replacement training protocol must resolve whether segmentation supervision
is allowed; no further cloud experiment is launched pending that clarification.**

The first 300-image pilot did not establish that InternVL improves segmentation:
overall mIoU/F1 fell while hard-positive recall increased. Its conclusion in
`EXPERIMENT_RESULTS.md` must be read as exploratory; no capacity control was run.

This iteration keeps the historical split: 210 train, 45 validation, 45 test,
with 70/15/15 per category. Seeds 42/43/44 vary initialization and batch order,
not the selected images. Eight epochs replace the initial three-epoch smoke test.
Checkpoint selection uses validation mIoU at threshold 0.5 throughout.
The historical test has already been inspected: these are exploratory ablations,
not a fresh independent confirmation or a population-level significance claim.

## Questions and controls

| Variant | Adapter | Attention | Question |
|---|---|---|---|
| baseline | none | ignored | decoder-only reference |
| legacy_raw | original | original teacher maps | reproduce unsafe fusion under paired training |
| capacity_only | bounded, post-normalization gate | all ones | gain from extra trainable capacity |
| gated_raw | same bounded adapter | original teacher maps | effect of fixing fusion |
| gated_filtered | same bounded adapter | calibrated teacher + baseline agreement | effect of reliability filtering |
| gated_shuffled | same bounded adapter | raw maps from other images | does teacher location matter? |
| filtered_shuffled | same bounded adapter | filtered maps from other images | control for sparse/small residual updates |

Every seed shares decoder initialization, optimizer settings, epoch-specific
training permutations and checkpoint criterion across variants. Shuffles are
derangements within split, independent of labels/category. Adapter variants share
initialization when architectures match. DINOv3 and InternVL stay frozen.

## Auditing truth and reliability

RGB/Mask are the available reference, subject to annotation quality. Audit detection
recall, negative false alarms, union-of-box precision, foreground coverage, IoU,
confidence calibration, foreground-area distribution and exact cross-split image
duplicates. Coarse box precision is not expected to equal mask precision; it measures
how much background a multiplicative prior introduces. Inspect best/worst train/val
examples visually. Neither plausible reasons nor valid JSON establishes correctness.

Train-only reliability targets require box foreground precision >= 0.15 and GT
coverage >= 0.10 (predeclared diagnostic definition, not semantic ground truth).
Regularized logistic calibration uses confidence, box area, log area and box count.
Inference inputs exclude labels, image IDs, category, and split. The same learned
model applies to train/val/test. This small calibrator needs external validation.

At inference reject parse failures, absent PV, missing boxes, confidence < 0.5,
or calibrated localization reliability < 0.5. Remaining attention is multiplied
by calibrated reliability and `clip((frozen_baseline_probability - 0.2)/0.6, 0, 1)`.
The baseline is fitted only on training images with validation checkpoint selection.
It supplies probabilities for every split; test labels never construct attention.
This conservative agreement filter can discard true small PV missed by baseline.

Use teacher information only as a bounded residual feature hint, never as a hard
pseudo-mask or a rule that teacher absence means background. Zero maps bypass
the adapter exactly, including after its weights/biases learn. Gating is applied
after normalization; a low confidence cannot be normalized back to full strength.
The final projection starts at zero; residual scale is bounded by 0.2.
The decoder is still trained, so a zero map is not a promise to reproduce the
separately trained baseline checkpoint's final prediction.

## Reporting

Report per-seed and mean/sample-std pixel mIoU, PV IoU, F1, precision, recall,
hard-positive false-positive pixels and overlap miss rate (GT coverage < 10%).
Image miss/false alarm denominators are positive/negative images respectively.
Eight training epochs, final validation and test per-image confusion counts,
teacher audits, rejection IDs, shuffle assignments, checkpoints and visualizations
are saved to Kaggle outputs. Model weights and checkpoints stay in Kaggle.
Do not select a method on test results or claim success when controls explain it.
