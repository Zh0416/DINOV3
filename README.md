# Multimodal Teacher Guided DINOv3 Adapter for PV Segmentation

Test whether frozen InternVL guidance improves frozen DINOv3 SAT features for
remote-sensing photovoltaic segmentation. **Manual masks are for final evaluation
only**, including no Mask-based Dice/CE, teacher calibration, threshold tuning or
checkpoint selection. Training uses machine-generated partial pseudo-labels.
Directory categories are not supervision.

Active protocol: [ITERATION_PROTOCOL.md](ITERATION_PROTOCOL.md).
Previous config: [configs/kaggle_label_free.yaml](configs/kaggle_label_free.yaml).
New soft-semantic config: [configs/kaggle_soft_semantic.yaml](configs/kaggle_soft_semantic.yaml).
Completed Kaggle v4 results: [LABEL_FREE_RESULTS.md](LABEL_FREE_RESULTS.md).
Overall mIoU improved from .5061 to .7333, but B missed all PV pixels in the
three hard-positive test images and falsely alarmed on both negative images.
This pilot does not establish reliable hard-scene segmentation.
Historical supervised v2/v3 pilots do not satisfy this constraint. Legacy
supervised entry points are blocked by default.

## Cloud workflow

Use [notebooks/dinov3_pv_kaggle.ipynb](notebooks/dinov3_pv_kaggle.ipynb) in Kaggle
zh0416/notebook468d03cea3 with pv-gf2, the historical teacher cache (only to exclude
old RGBs), and the mounted DINOv3 SAT checkpoint. The new soft-semantic protocol
selects 120 RGBs while explicitly excluding the previous 120 RGB hashes, runs global
and overlapping local InternVL grid views, keeps 0/1/255 machine targets, and trains
paired A/B controls (plus no-adapter and shuffled-semantic controls) x three seeds.

```bash
pip install -r requirements.txt
pip install -e . --no-deps --no-build-isolation
python scripts/08_label_free_pipeline.py --config configs/kaggle_soft_semantic.yaml --stage prepare
python scripts/08_label_free_pipeline.py --config configs/kaggle_soft_semantic.yaml --stage features
python scripts/08_label_free_pipeline.py --config configs/kaggle_soft_semantic.yaml --stage teacher
python scripts/08_label_free_pipeline.py --config configs/kaggle_soft_semantic.yaml --stage pseudo
python scripts/08_label_free_pipeline.py --config configs/kaggle_soft_semantic.yaml --stage train
python scripts/08_label_free_pipeline.py --config configs/kaggle_soft_semantic.yaml --stage seal
python scripts/09_final_evaluation.py --output-root /kaggle/working/outputs_soft_v2
```

The image-only manifest has no annotation path or category. Pre-evaluation
processes block accidental reads from annotation directories. Prototypes use only
training RGBs. Teacher failure/disagreement means abstention, not background.
Epochs/threshold are fixed before annotation access. Final evaluation verifies
SHA256 hashes of predictions and decision artifacts before reading GT.

## Reliability and controls

Cross-view box agreement, fixed RGB appearance seeds and train-only DINO cosine
margin filter teacher proposals. Self-reported confidence is logged, not treated
as correctness probability. Consistent hallucinations remain possible. Final
audits measure pseudo-label precision/coverage and false acceptance, without
feeding back into thresholds or weights.

A is explicitly an RGB/DINO heuristic decoder baseline: DINO alone has no PV
class identity without supervision. B adds filtered teacher pseudo-supervision
and an attention adapter. Same-pseudo decoder, capacity and shuffled-attention
controls separate supervision, parameter count and spatial guidance. Improvement
is not assumed. Small/spatially correlated holdouts limit conclusions.

## Outputs and weights

Each stage logs to stdout and /kaggle/working/outputs/logs. Reports include sample
counts, teacher responses, rejection reasons, pseudo support, losses, final
per-image/per-category metrics and paired bootstrap CIs. Cloud outputs retain last
checkpoints, full prediction masks, a hash seal and fixed-order visualizations.

Base weights/cache stay in /kaggle/temp/model_cache; checkpoints/features stay in
Kaggle. Neither is downloaded locally or committed. Use the mounted timm mirror
via DINOV3_MODEL_PATH or the official gated checkpoint with Kaggle HF_TOKEN.
Teacher: OpenGVLab/InternVL3_5-4B.

## Small smoke tests and cache recovery

Run a four-image cloud smoke before the full protocol. It keeps six teacher
views per image, trains A/B for one epoch and verifies the prediction seal,
without opening manual masks or reporting accuracy. Use a new output directory:

```bash
python scripts/10_smoke_soft_semantic.py --config configs/kaggle_soft_semantic.yaml --output-root /kaggle/working/soft_smoke --samples 4
```

For a CPU-only replay of four saved v6 images, add
`--replay-cache /path/to/semantic_prompt.json`. This tests cache migration and
attention fusion without loading models or opening images. Results are written
to `smoke_report.json`; this mode does not test feature extraction or training.

Resuming the grid teacher reparses cached raw responses and preserves their
generation prompt and view geometry. Fully cached runs do not load InternVL or
require CUDA; only missing/failed views trigger inference. Unknown grid cells
abstain regardless of their numeric probability. When changing the frozen
protocol, prepare a new output directory; do not edit an old frozen config or seal.

## Synthetic verification

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Tests cover annotation isolation, ignored-region gradients, flip coordinates,
train-only prototypes, empty-gate identity, artifact tamper detection and a
synthetic train/seal/final-evaluation smoke test. They download no model weights.
