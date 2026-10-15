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


class PartialPseudoLoss(nn.Module):
    """Dice/CE only on trusted machine pseudo-labels; 255 has zero gradient.

    Dice is computed on images with positive pseudo-labels. Background-only
    images contribute CE without inventing a foreground Dice target.
    """

    def __init__(self, dice_weight=1., cross_entropy_weight=1., ignore_index=255):
        super().__init__()
        self.dice_weight, self.ce_weight, self.ignore_index = dice_weight, cross_entropy_weight, ignore_index

    def forward(self, logits, pseudo_target):
        valid = pseudo_target != self.ignore_index
        if not valid.any():
            return logits.sum()*0, {"valid_pixels": 0, "dice_loss": 0., "cross_entropy": 0.}
        target = pseudo_target.clamp(0, 1).long()
        # Per-image class balancing stops a few large seed maps dominating.
        foreground, background = valid & (target == 1), valid & (target == 0)
        n_fg = foreground.sum((1, 2)).clamp_min(1)
        n_bg = background.sum((1, 2)).clamp_min(1)
        counts = torch.stack((n_bg, n_fg), 1).float()
        weights = (1/counts).gather(1, target.flatten(1)).reshape_as(target) * valid
        per_pixel = F.cross_entropy(logits.float(), target, reduction="none")
        ce = ((per_pixel*weights).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1e-8))
        ce = ce[valid.any((1, 2))].mean()
        probability = logits.float().softmax(1)[:, 1] * valid
        positive = foreground.float()
        dice = 1-(2*(probability*positive).sum((1, 2))+1)/(probability.sum((1, 2))+positive.sum((1, 2))+1)
        dice = dice[foreground.any((1, 2))].mean() if foreground.any() else logits.sum()*0
        loss = self.dice_weight*dice + self.ce_weight*ce
        return loss, {"valid_pixels": int(valid.sum()), "dice_loss": float(dice.detach()),
                      "cross_entropy": float(ce.detach())}

