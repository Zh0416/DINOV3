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


class IgnoreAwareDiceFocalLoss(nn.Module):
    """Dice plus focal loss for weighted three-way machine targets."""

    def __init__(self, dice_weight=1.0, focal_weight=1.0, gamma=2.0, alpha=0.25, ignore_index=255):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.ignore_index = int(ignore_index)

    def forward(self, logits, target, pixel_weight=None):
        valid = target != self.ignore_index
        if not valid.any():
            zero = logits.sum() * 0.0
            return zero, {"valid_pixels": 0, "dice_loss": 0.0, "focal_loss": 0.0}
        safe_target = target.clamp(0, 1).long()
        weights = torch.ones_like(logits[:, 0]) if pixel_weight is None else pixel_weight.to(logits.device).float()
        weights = weights * valid
        per_pixel = F.cross_entropy(logits.float(), safe_target, reduction="none")
        probabilities = logits.float().softmax(dim=1)
        pt = probabilities.gather(1, safe_target.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0 - 1e-6)
        alpha = torch.where(safe_target == 1, self.alpha, 1.0 - self.alpha)
        focal = (alpha * (1.0 - pt).pow(self.gamma) * per_pixel * weights).sum() / weights.sum().clamp_min(1.0)
        foreground = valid & (safe_target == 1)
        probability = probabilities[:, 1]
        intersection = (probability * foreground.float() * weights).sum((1, 2))
        denominator = (probability * valid.float() * weights).sum((1, 2)) + (foreground.float() * weights).sum((1, 2))
        per_image = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        has_fg = foreground.any((1, 2))
        dice = per_image[has_fg].mean() if has_fg.any() else logits.sum() * 0.0
        total = self.dice_weight * dice + self.focal_weight * focal
        return total, {"valid_pixels": int(valid.sum()), "dice_loss": float(dice.detach()),
                       "focal_loss": float(focal.detach())}

