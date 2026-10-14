from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, cross_entropy_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.cross_entropy_weight = cross_entropy_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        cross_entropy = F.cross_entropy(logits, target.long())
        probability = logits.softmax(dim=1)[:, 1]
        target_float = target.float()
        intersection = (probability * target_float).sum(dim=(1, 2))
        denominator = probability.sum(dim=(1, 2)) + target_float.sum(dim=(1, 2))
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        total = self.dice_weight * dice_loss + self.cross_entropy_weight * cross_entropy
        return total, {"dice_loss": float(dice_loss.detach()), "cross_entropy": float(cross_entropy.detach())}

