# Pilot Experiment Results

Current final-evaluation-only Kaggle v4 results are reported separately in
[LABEL_FREE_RESULTS.md](LABEL_FREE_RESULTS.md). This file preserves historical
supervised results only.

**Historical supervised pilot: NOT compliant with the user's final-evaluation-only
Mask constraint. No result here validates the current label-free hypothesis.
The former causal conclusion has been withdrawn; see ITERATION_PROTOCOL.md.**

Kaggle Notebook: `zh0416/notebook468d03cea3`, version 2  
Git commit used by the run: `6fdcb60`  
Status: `COMPLETE`

## Setup

- Dataset audit: 2,327 RGB/mask pairs, all 512 x 512.
- Pilot subset: 300 images, balanced across positive, hard-positive, and negative categories.
- Split: 210 train / 45 validation / 45 test, seed 42.
- Frozen backbone: DINOv3 ViT-L/16 SAT.
- Cached patch feature shape: 1024 x 64 x 64 for every sample.
- Teacher: InternVL3.5-4B, 300 successful records; 210 records contained bounding boxes.
- Trainable components: decoder for experiment A; adapter and decoder for experiment B.
- Training: 3 epochs, Dice plus cross-entropy loss.

## Test Results

| Metric | A: DINOv3 baseline | B: InternVL adapter | B - A |
|---|---:|---:|---:|
| mIoU | 0.7682 | 0.7559 | -0.0124 |
| PV IoU | 0.5549 | 0.5333 | -0.0216 |
| F1 | 0.7138 | 0.6957 | -0.0181 |
| Precision | 0.7332 | 0.6527 | -0.0805 |
| Recall | 0.6953 | 0.7446 | +0.0493 |

## Hard-Positive Analysis

The test split contained 15 hard-positive images.

| Metric | Baseline | InternVL adapter | Change |
|---|---:|---:|---:|
| PV IoU | 0.2706 | 0.2902 | +0.0196 |
| F1 | 0.4260 | 0.4498 | +0.0239 |
| Precision | 0.8304 | 0.4212 | -0.4091 |
| Recall | 0.2865 | 0.4826 | +0.1961 |
| Image miss rate | 46.67% | 13.33% | -33.33 pp |
| Pixel false-negative rate | 71.35% | 51.74% | -19.61 pp |

The adapter reduced hard-positive misses substantially, but it also over-segmented:
overall precision dropped by 8.05 percentage points, and the negative-image false-alarm
rate increased from 26.67% to 33.33%.

## Conclusion

This pilot does not establish support for the intended hypothesis. Recall rose
while precision, overall mIoU and F1 fell. No capacity or shuffled-location control
was run, so gains cannot be attributed specifically to teacher semantics. Both
models used manual Mask training; their checkpoints and label-fitted calibrators
must not be reused by the current protocol.

Cloud outputs include both best checkpoints, 16 prediction visualizations, the data
inspection visualization, full-resolution and 64 x 64 attention maps, stage logs,
and JSON/CSV reports. Base-model weights remained in Kaggle cloud storage.
