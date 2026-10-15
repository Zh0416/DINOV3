"""Train-only reliability calibration with label-free inference features."""
from __future__ import annotations

import numpy as np

from pvseg.attention import sanitize_boxes


def reliability_features(record):
    boxes = sanitize_boxes(record.get("bbox", []), 1024)
    area = min(sum((b[2]-b[0])*(b[3]-b[1]) for b in boxes) / 1024**2, 1.)
    return [float(record.get("confidence", 0)), area, np.log(max(area, 1e-5)), len(boxes)]


class TeacherReliability:
    """Regularized logistic model; no category, id, label, or split in inference inputs."""

    def fit(self, records, audit_rows):
        rows = {r["sample_id"]: r for r in audit_rows}
        train = [r for r in records if rows[r["sample_id"]]["split"] == "train" and r.get("bbox")]
        if not train:
            raise ValueError("No training teacher boxes to calibrate")
        x = np.asarray([reliability_features(r) for r in train], dtype=np.float64)
        y = np.asarray([rows[r["sample_id"]]["useful_localization"] for r in train], dtype=float)
        self.mean, self.std = x.mean(0), np.maximum(x.std(0), 1e-4)
        x = np.column_stack([np.ones(len(x)), (x-self.mean)/self.std])
        self.weights = np.zeros(x.shape[1])
        penalty = np.eye(x.shape[1]) * 5.
        penalty[0, 0] = .01
        for _ in range(30):
            p = 1 / (1 + np.exp(-np.clip(x @ self.weights, -20, 20)))
            hessian = x.T @ (x * (p*(1-p))[:, None]) + penalty
            step = np.linalg.solve(hessian, x.T @ (p-y) + penalty @ self.weights)
            self.weights -= step
            if np.linalg.norm(step) < 1e-7:
                break
        self.training_ids = [r["sample_id"] for r in train]
        return self

    def predict(self, record):
        if (record.get("status") != "ok" or record.get("parse_warning") or
                not record.get("pv_exist") or not sanitize_boxes(record.get("bbox"), 1024)):
            return 0.
        x = np.r_[1., (np.asarray(reliability_features(record))-self.mean)/self.std]
        return float(1 / (1 + np.exp(-np.clip(x @ self.weights, -20, 20))))

    def to_dict(self):
        return {"features": ["confidence", "box_area", "log_box_area", "box_count"],
                "mean": self.mean.tolist(), "std": self.std.tolist(), "weights": self.weights.tolist(),
                "fit_split": "train", "training_ids": self.training_ids}


def filter_attention(attention, baseline_probability, reliability, confidence, threshold=.5):
    """Teacher is a soft residual hint. Absence never forces a background pseudo-label."""
    if confidence < threshold or reliability < .5:
        return np.zeros_like(attention)
    support = np.clip((baseline_probability - .2) / .6, 0, 1)
    return attention * support * reliability
