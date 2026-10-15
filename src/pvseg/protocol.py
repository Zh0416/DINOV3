"""Prevent evaluation-only labels from entering training or teacher selection."""


def require_supervised_labels(config):
    raise RuntimeError(
        "Labels are reserved for final evaluation. Historical supervised Dice/CE "
        "training and label-fitted teacher reliability entry points are retired. "
        "Use scripts/08_label_free_pipeline.py; no config flag re-enables Mask training."
    )
