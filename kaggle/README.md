# Kaggle deployment directory

Before pushing a version, copy `notebooks/dinov3_pv_kaggle.ipynb` to
`kaggle/notebook468d03cea3.ipynb`. The metadata enables a GPU, Internet access,
and the `zh0416/pv-gf2` dataset, a historical teacher-cache manifest used only for
RGB exclusion, and a mounted DINOv3 SAT checkpoint. The active config is
`configs/kaggle_label_free.yaml`. No manual Mask is opened until every run's
predictions and checkpoints have been hashed and sealed. Model weights remain
in cloud storage; download only reports, logs, and selected visualization files.

