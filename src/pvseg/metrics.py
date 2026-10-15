from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Confusion:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0
    images: int = 0
    image_misses: int = 0
    image_false_alarms: int = 0
    positive_images: int = 0
    negative_images: int = 0
    overlap_misses: int = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.bool()
        target = target.bool()
        self.true_positive += int((prediction & target).sum())
        self.false_positive += int((prediction & ~target).sum())
        self.false_negative += int((~prediction & target).sum())
        self.true_negative += int((~prediction & ~target).sum())
        self.images += 1
        has_target = bool(target.any())
        has_prediction = bool(prediction.any())
        self.image_misses += int(has_target and not has_prediction)
        self.image_false_alarms += int(not has_target and has_prediction)
        self.positive_images += int(has_target)
        self.negative_images += int(not has_target)
        # A prediction elsewhere in the scene must not count as detecting PV.
        coverage = int((prediction & target).sum()) / max(int(target.sum()), 1)
        self.overlap_misses += int(has_target and coverage < .1)

    def compute(self) -> dict[str, object]:
        tp, fp, fn, tn = self.true_positive, self.false_positive, self.false_negative, self.true_negative
        pv_iou = tp / max(tp + fp + fn, 1)
        background_iou = tn / max(tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        return {
            "pv_iou": pv_iou,
            "background_iou": background_iou,
            "miou": (pv_iou + background_iou) / 2.0,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "pixel_false_negative_rate": fn / max(tp + fn, 1),
            "image_miss_rate": self.image_misses / max(self.positive_images, 1),
            "overlap_miss_rate": self.overlap_misses / max(self.positive_images, 1),
            "image_false_alarm_rate": self.image_false_alarms / max(self.negative_images, 1),
            "positive_images": self.positive_images,
            "negative_images": self.negative_images,
            "images": self.images,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }


class SegmentationMetrics:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.overall = Confusion()
        self.by_category: dict[str, Confusion] = {}

    def update(self, logits: torch.Tensor, target: torch.Tensor, categories: list[str]) -> None:
        probability = logits.softmax(dim=1)[:, 1]
        predictions = probability >= self.threshold
        for index, category in enumerate(categories):
            self.overall.update(predictions[index].cpu(), target[index].cpu())
            self.by_category.setdefault(category, Confusion()).update(
                predictions[index].cpu(), target[index].cpu()
            )

    def compute(self) -> dict[str, object]:
        return {
            "overall": self.overall.compute(),
            "by_category": {name: meter.compute() for name, meter in sorted(self.by_category.items())},
        }

