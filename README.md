# Multimodal Teacher Guided DINOv3 Adapter for PV Segmentation

Test whether frozen InternVL guidance improves frozen DINOv3 SAT features for
remote-sensing photovoltaic segmentation. **Manual masks are for final evaluation
only**, including no Mask-based Dice/CE, teacher calibration, threshold tuning or
checkpoint selection. Training uses machine-generated partial pseudo-labels.
Directory categories are not supervision.

Active protocol: [ITERATION_PROTOCOL.md](ITERATION_PROTOCOL.md).
Previous config: [configs/kaggle_label_free.yaml](configs/kaggle_label_free.yaml).
Active soft-semantic v4: [configs/modelscope_soft_semantic_v4.yaml](configs/modelscope_soft_semantic_v4.yaml)
for ModelScope, [configs/kaggle_soft_semantic.yaml](configs/kaggle_soft_semantic.yaml) for Kaggle.
ModelScope setup and commands: [MODELSCOPE_RUN.md](MODELSCOPE_RUN.md).
Completed Kaggle v4 results: [LABEL_FREE_RESULTS.md](LABEL_FREE_RESULTS.md).
Overall mIoU improved from .5061 to .7333, but B missed all PV pixels in the
three hard-positive test images and falsely alarmed on both negative images.
This pilot does not establish reliable hard-scene segmentation.
Historical supervised v2/v3 pilots do not satisfy this constraint. Legacy
supervised entry points are blocked by default.

## Cloud workflow

Run from the repository root on a CUDA server. On ModelScope, configure the data
and historical RGB exclusion manifest paths, and set `DINOV3_MODEL_PATH` and
`INTERNVL_MODEL_PATH` to local weights as described in [MODELSCOPE_RUN.md](MODELSCOPE_RUN.md).
The soft-semantic protocol
selects 120 RGBs while explicitly excluding the previous 120 RGB hashes, runs global
and overlapping local InternVL grid views, keeps 0/1/255 machine targets, and trains
paired A/B controls (plus constant-map capacity and shuffled-semantic controls) x three seeds.

```bash
pip install -r requirements.txt
pip install -e . --no-deps --no-build-isolation
python scripts/08_label_free_pipeline.py --config configs/modelscope_soft_semantic_v4.yaml --stage prepare
python scripts/08_label_free_pipeline.py --config configs/modelscope_soft_semantic_v4.yaml --stage teacher
python scripts/08_label_free_pipeline.py --config configs/modelscope_soft_semantic_v4.yaml --stage features
python scripts/08_label_free_pipeline.py --config configs/modelscope_soft_semantic_v4.yaml --stage pseudo
python scripts/08_label_free_pipeline.py --config configs/modelscope_soft_semantic_v4.yaml --stage train
python scripts/08_label_free_pipeline.py --config configs/modelscope_soft_semantic_v4.yaml --stage seal
python scripts/09_final_evaluation.py --output-root ./runs/fresh120_soft_semantic_v4
```

The image-only manifest has no annotation path or category. Pre-evaluation
processes block accidental reads from annotation directories. Prototypes use only
training RGBs. Teacher failure/disagreement means abstention, not background.
Epochs/threshold are fixed before annotation access. Final evaluation verifies
SHA256 hashes of predictions and decision artifacts before reading GT.

## Reliability and controls

Each view emits a compact 4x4 probability/status grid, using `[probability,"p"|"n"|"u"]`
per cell. New runs require all 16 locations before proceeding. Explicit positive
and negative conflicts at the same location become ignored pixels; unknowns and
areas outside crops abstain. `local_consistency` measures positive votes among
observed cells. Each class needs at least two supporting views. Non-conflicting
observations are averaged and smoothed into a map with base 0.2. A coarse positive
grid marks a candidate region, never all its pixels as PV. Positive seeds require
both RGB support and a DINO cosine margin; background needs explicit repeated
negative evidence plus background appearance. Prototypes use only training RGB
seeds inside the corresponding multi-view teacher regions. Missing either class
or indistinguishable prototypes stops training, including direct training calls.
Self-reported confidence is logged, not treated
as correctness probability. Consistent hallucinations remain possible. Final
audits measure pseudo-label precision/coverage and false acceptance, without
feeding back into thresholds or weights.

A (`baseline_decoder`) and B (`semantic_adapter`) share the same soft machine
targets, including teacher supervision. B adds an Adapter conditioned on its own
semantic map; new-image inference for B also requires a teacher map.
`semantic_capacity` uses the same Adapter with a constant 0.2 map, and
`shuffled_semantic` uses another image's map within the same split. These controls
help assess added parameters and spatial guidance. Improvement
is not assumed. Small/spatially correlated holdouts limit conclusions.

## Outputs and weights

Each stage logs to stdout and `<runtime.output_dir>/logs`. Reports include sample
counts, teacher responses, rejection reasons, pseudo support, losses, final
per-image/per-category metrics and paired bootstrap CIs. Cloud outputs retain last
checkpoints, full prediction masks, a hash seal and fixed-order visualizations.

Base weights/cache use `runtime.model_cache_dir`; checkpoints/features use
`runtime.output_dir`. Keep them on the cloud server, outside Git. Use local weights
via `DINOV3_MODEL_PATH` / `INTERNVL_MODEL_PATH`, or the official gated checkpoint with `HF_TOKEN`.
Teacher: OpenGVLab/InternVL3_5-4B.

## Small smoke tests and cache recovery

Run a four-image cloud smoke before the full protocol. It keeps six teacher
views per image, requires complete grids, trains A/B for two epochs (at least two
successful optimizer steps each) and verifies the prediction seal,
without opening manual masks or reporting accuracy. Use a new output directory:

```bash
python scripts/10_smoke_soft_semantic.py --config configs/modelscope_soft_semantic_v4.yaml --output-root ./runs/soft_smoke_v4 --samples 4
```

For a CPU-only replay of four saved v6 images, add
`--replay-cache /path/to/semantic_prompt.json`. This tests cache migration and
attention fusion without loading models or opening images. Results are written
to `smoke_report.json`; this mode does not test feature extraction or training.

Resuming the grid teacher reparses cached raw responses and preserves their
generation prompt and view geometry. Normal resume requires identical prompt,
generation parameters and preprocessing version in every view. Fully cached runs do not load InternVL or
require CUDA; only missing/failed views trigger inference. Incomplete cached grids
stop a normal run with the saved raw response; use a new output for regeneration.
Recovery-only replay explicitly allows partial historical grids. Unknown grid cells
abstain regardless of their numeric probability. When changing the frozen
protocol, prepare a new output directory; do not edit an old frozen config or seal.

Global original/hflip decisions are checked before local inference. More than
4/16 explicitly conflicting cells stops the run and saves `teacher_pair_checks.json`;
this is a fixed engineering gate, not calibrated accuracy. A cheaper two-training-image
check is available in `scripts/11_teacher_preflight.py` (see ModelScope instructions).
V4 supplies a global thumbnail alongside dynamic image tiles and removes the
positive-only prompt example. These changes require GPU validation; they do not
establish that InternVL recognizes PV correctly. See [method review](METHOD_REVIEW_20261017.md).

## Synthetic verification

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Tests cover annotation isolation, ignored-region gradients, flip coordinates,
train-only prototypes, empty-gate identity, artifact tamper detection and a
synthetic train/seal/final-evaluation smoke test. They download no model weights.
