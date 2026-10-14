# Multimodal Teacher Guided DINOv3 Adapter for Photovoltaic Segmentation

This repository tests whether semantic guidance from InternVL improves frozen
DINOv3 features for photovoltaic (PV) segmentation in high-resolution remote
sensing imagery.

The project compares two controlled experiments on the same stratified split:

- **Experiment A:** frozen DINOv3 ViT-L/16 SAT features + convolutional decoder.
- **Experiment B:** frozen DINOv3 features + InternVL Gaussian attention +
  PV-aware adapter + the same decoder architecture.

The default Kaggle run uses at most 300 images (100 per category when
available), split 70/15/15 with seed 42. The dataset is expected to contain
`正样本`, `困难正样本`, and `负样本` directories with paired `images` and
`labels` subdirectories.

## Cloud-only model weights

Model weights are never committed to this repository and are not downloaded by
the local test suite. On Kaggle, Hugging Face caches are directed to the
ephemeral `/kaggle/temp/model_cache` directory. Set the Kaggle secret `HF_TOKEN` if the official
DINOv3 checkpoint requires gated access. A pre-mounted Kaggle Model can instead
be supplied through `DINOV3_MODEL_PATH` or `INTERNVL_MODEL_PATH`.

Default checkpoints:

- `facebook/dinov3-vitl16-pretrain-sat493m`
- `OpenGVLab/InternVL3_5-4B`

## Pipeline

```bash
pip install -r requirements.txt
pip install -e . --no-deps --no-build-isolation

python scripts/01_inspect_data.py --config configs/kaggle_small.yaml
python scripts/02_generate_teacher.py --config configs/kaggle_small.yaml
python scripts/03_cache_dino_features.py --config configs/kaggle_small.yaml
python scripts/04_train_experiments.py --config configs/kaggle_small.yaml
python scripts/05_compare_experiments.py --config configs/kaggle_small.yaml
```

Every stage logs to both stdout and `/kaggle/working/outputs/logs`. Outputs
include the split manifest, data inspection figures, `semantic_prompt.json`,
Gaussian attention maps, DINOv3 feature caches, best checkpoints, prediction
visualizations, per-category metrics, and the final A/B comparison report.

The Kaggle entry point is [notebooks/dinov3_pv_kaggle.ipynb](notebooks/dinov3_pv_kaggle.ipynb).

## Metrics

The evaluation reports PV IoU, two-class mIoU, F1/Dice, precision, recall,
pixel false-negative rate, hard-positive image miss rate, and negative image
false-alarm rate. The comparison script does not assume that the adapter wins;
it reports the measured deltas.

## Tests

The local test suite uses synthetic tensors and temporary images only:

```bash
python -m pytest -q
```
