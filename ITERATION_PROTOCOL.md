# Final-evaluation-only pilot: fresh120_pseudo_v1

Manual masks cannot enter feature extraction, training, teacher filtering, loss,
threshold tuning, checkpoint selection or pre-final visualization. Category
directory names are not training labels. Dice/CE below use **machine-generated
pseudo-labels only**; unknown patches have target 255 and zero loss gradient.

## Historical evidence and retirement

Kaggle v2 used 300 images (210 train / 45 val / 45 test), with supervised Dice/CE.
mIoU fell .7682 -> .7559 and precision .7332 -> .6527; hard-positive FP pixels rose
5,692 -> 64,490 while recall improved. This does not establish improvement under
the intended protocol. V3 was submitted before the constraint was clarified and
is retired too. Never reuse their models, probabilities, mask-fitted calibrators
or validation-selected checkpoints in this pilot.

Confirmed defects: legacy adapter biases/normalization permit nonzero updates at
zero attention; normalization can erase confidence scaling. The replacement has
a zero-initialized final projection and applies attention after normalization,
with a bounded scalar residual factor. The update vector norm is not bounded.

Teacher confidence concentrated at .95; the old prompt explicitly gave .95 as an
example. This is not a calibrated probability. Valid JSON or plausible reasons do
not establish localization correctness. Coarse boxes include background. Tiny-PV
information loss from 512 -> 448 and inadequate training remain hypotheses, not
demonstrated causes. Model capacity was uncontrolled in v2.

## Image isolation

Select 120 unique RGBs by seeded content-hash ranking, excluding the historical
300 and their exact RGB duplicates. No category stratification: 84 train / 36 final
test, no annotation-based validation split. Scene/geographic grouping is unknown;
content deduplication does not establish spatial independence.

Preparation enumerates only images directories and stages RGBs in flat,
content-addressed paths. The manifest contains only ID, RGB hash/path and split.
It does not check annotation existence. Pre-evaluation processes install an
accidental-read guard on labels/masks/mask directories. This is an auditable code
guard, not an OS security sandbox.

## Fixed weak-supervision recipe

1. Extract frozen DINOv3 ViT-L/16 SAT features, shape 1024x64x64, from RGB only.
2. Run frozen InternVL3.5-4B on original and horizontally flipped RGB. Restore
   flipped boxes to original coordinates. Remove the confidence example. Retain
   raw responses, errors, patch counts and both coordinate systems.
3. Reject invalid/absent views, box-union IoU below .5, union area outside
   [.0005,.6], or insufficient RGB/DINO support. Absence or disagreement means
   unknown, never an instruction to mark an entire image background.
   Self-confidence is logged but not used as a calibrated acceptance score.
4. Fixed RGB prior at 64x64: foreground luminance [.05,.5], B >= .9 R and
   G <= 1.15 B; background luminance >= .65 or G > 1.25 max(R,B). This can confuse
   water/shadows and miss atypical panels.
5. Fit two normalized DINO prototypes from RGB seeds on training images only,
   equally weighting usable images. Accept seeds only if cosine similarity margin
   supports their class by .04. These are appearance proxies, not probabilities.
6. Filtered foreground intersects RGB/DINO foreground seeds with the two-view
   box intersection; require at least four patches per image. Background requires
   RGB/DINO agreement outside a two-patch dilation of either view's boxes.
   All other patches are ignored. Gaussian sigma=2 patches diffuses accepted
   seeds into attention, not new positive labels. Keep raw-box Gaussian control.

Agreement can preserve the same hallucination in both views. RGB/DINO can share
teacher biases. Filtering trades coverage for reliability. No automatic proxy
guarantees that a proposal is PV. Only the final annotation audit estimates actual
precision, coverage, false acceptance and false rejection.

## Paired experiments

| Variant | Pseudo supervision | Adapter attention | Purpose |
|---|---|---|---|
| heuristic_decoder (A) | RGB/DINO seeds | no adapter | explicit no-InternVL weak prior |
| heuristic_capacity | same heuristic seeds | ones | capacity without teacher |
| raw_adapter | original coarse boxes + conservative BG | raw Gaussian | noisy teacher control |
| filtered_decoder | filtered seeds | no adapter | teacher pseudo supervision alone |
| filtered_capacity | same filtered seeds | ones | capacity with same supervision |
| filtered_adapter (B) | same filtered seeds | filtered Gaussian | proposed model |
| filtered_shuffled | same filtered seeds | other image's filtered attention | spatial guidance control |

Seven variants x seeds 42/43/44, eight fixed epochs. Share decoder initialization
and per-epoch batch order. Matching adapters share initialization. Shuffles are
derangements within split. B vs A changes supervision and architecture; B vs the
same-pseudo decoder/capacity/shuffle controls is needed to assess spatial guidance.

Train adapter/decoder using class-balanced partial pseudo CE + partial Dice and
.1 head flip consistency. Consistency on flipped cached features does not mean
re-extracting an augmented DINO backbone. Train/infer the decoder at 256x256 then
resize logits to 1024x1024. AdamW lr=.0003, weight decay=.0001, batch=2. Last epoch,
threshold=.5, no early stopping or supervised warm start. Stop explicitly if any
pseudo method has no positive training signal; never loosen thresholds using GT.

## Final evaluation

Save every last checkpoint and all test masks. Hash config, RGB manifest, teacher
JSON, pseudo cache, prototypes, checkpoints and predictions in prediction_seal.json.
A separate process verifies every hash before opening annotations, using the
frozen config. No label-based model selection or retrospective tuning.

Report mIoU, PV IoU, F1, precision, recall, FP pixels, hard-positive overlap misses
(GT coverage <10%) and negative false alarms, per-seed and mean/sample std. Paired
image bootstrap contrasts acknowledge spatial dependence and small subgroups.
Evaluate at native GT resolution; never resize GT into training pseudo targets.
Audit pseudo precision/coverage, background contamination and teacher false
acceptance/rejection after sealing. Save fixed-order GT/prediction visualizations.
Weights/features/checkpoints stay in Kaggle. Do not retune and call the same
holdout untouched.

## Soft-semantic v2 protocol

The active redesign is `fresh120_soft_semantic_v2` in
`configs/kaggle_soft_semantic.yaml`. It explicitly excludes the 120 RGB hashes
from `fresh120_pseudo_v1` using metadata only, so the new holdout is not the prior
36-image test. InternVL is queried on the full image, a horizontal flip, and four
overlapping local crops. Each view emits a 4x4 grid probability/status record;
coordinates are fused into a nonzero-base soft map rather than a hard bounding box.

Machine targets are three-way: stable positive core=1, stable machine background=0,
and uncertain/transition/conflicting pixels=255. The retention circuit measures only
machine pseudo-candidate retention before final annotation access. If it falls below
50%, the pipeline retains a wider candidate with soft weights and does not invent
background labels. Human masks remain unavailable until the prediction seal.

Experiment A is a Decoder-only baseline and B is the zero-initialized FiLM Adapter
with the soft map. They share exactly the same machine targets, optimizer, decoder
capacity, seeds, fixed epoch count and final checkpoint rule. `semantic_no_adapter`
and `shuffled_semantic` are preregistered controls. Loss is ignore-aware Dice plus
Focal; soft weights affect valid machine pixels only. The Adapter never multiplies
DINO features by attention and starts as an exact identity.
