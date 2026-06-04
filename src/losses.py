"""
losses.py — Segmentasyon Kayıp Fonksiyonları

Bu dosya eğitimde kullanılacak farklı loss kombinasyonlarını içerir.
Amacımız küçük hasar bölgelerinde IoU kalitesini artırmak.
"""

import torch
import torch.nn as nn

from src.config import Config


class DiceLoss(nn.Module):
    """Binary segmentasyon için Dice loss."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Binary segmentasyon için logits tabanlı focal loss."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - pt).pow(self.gamma)

        alpha_weight = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()


class CombinedSegmentationLoss(nn.Module):
    """İki farklı kaybı ağırlıklı toplar."""

    def __init__(self, loss_a, loss_b, weight_a=0.5, weight_b=0.5):
        super().__init__()
        self.loss_a = loss_a
        self.loss_b = loss_b
        self.weight_a = weight_a
        self.weight_b = weight_b

    def forward(self, logits, targets):
        return self.weight_a * self.loss_a(logits, targets) + self.weight_b * self.loss_b(logits, targets)


def build_loss_fn():
    """Config'e göre eğitimde kullanılacak loss fonksiyonunu üretir."""

    loss_type = Config.LOSS_TYPE.lower()

    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()

    if loss_type == "dice":
        return DiceLoss()

    if loss_type == "bce_dice":
        return CombinedSegmentationLoss(
            nn.BCEWithLogitsLoss(),
            DiceLoss(),
            weight_a=Config.BCE_WEIGHT,
            weight_b=Config.DICE_WEIGHT,
        )

    if loss_type == "focal":
        return FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    if loss_type == "focal_dice":
        return CombinedSegmentationLoss(
            FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA),
            DiceLoss(),
            weight_a=Config.BCE_WEIGHT,
            weight_b=Config.DICE_WEIGHT,
        )

    raise ValueError(
        "Config.LOSS_TYPE gecersiz. Beklenen: bce, dice, bce_dice, focal, focal_dice"
    )
