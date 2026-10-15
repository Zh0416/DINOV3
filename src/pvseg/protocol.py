"""Prevent evaluation-only labels from entering training or teacher selection."""


def require_supervised_labels(config):
    if config.get("protocol", {}).get("label_usage", "final_evaluation_only") != "supervised_training":
        raise RuntimeError(
            "Labels are reserved for final evaluation. Supervised Dice/CE training and "
            "label-fitted teacher reliability are disabled. Define a label-free training "
            "protocol before running a new experiment."
        )
