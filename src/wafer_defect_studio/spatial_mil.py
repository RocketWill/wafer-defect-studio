"""Losses for weakly supervised spatial logits."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .detection_windows import Rect
from .training_input_bundle import TrainingInputBundle


def derive_train_positive_class_weights(bundle: TrainingInputBundle) -> Tensor:
    """Return capped inverse-positive-frequency weights from Training bags."""

    train_ids = {
        source.image_asset_id for source in bundle.sources if source.split == "train"
    }
    train_bags = tuple(
        bag for bag in bundle.patch_bags if bag.image_asset_id in train_ids
    )
    total = len(train_bags)
    return torch.tensor(
        [
            1.0
            if (positive := sum(code in bag.class_codes for bag in train_bags)) == 0
            else min(10.0, max(1.0, (total - positive) / positive))
            for code in bundle.class_codes
        ],
        dtype=torch.float32,
    )


def positive_spatial_mil_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Reward one strong spatial response for each present class."""

    positive_logits = torch.amax(logits, dim=(-2, -1))[targets == 1]
    if positive_logits.numel() == 0:
        return logits.sum() * 0.0
    return F.softplus(-positive_logits).mean()


def absent_class_hard_negative_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Penalize the strongest spatial response for each absent class."""

    absent_logits = torch.amax(logits, dim=(-2, -1))[targets == 0]
    if absent_logits.numel() == 0:
        return logits.sum() * 0.0
    return F.softplus(absent_logits).mean()


def overlap_consistency_loss(
    logits: Tensor,
    patch_rects: Sequence[Rect],
    feature_stride: int = 4,
) -> Tensor:
    """Penalize probability disagreement where source-coordinate patches overlap."""

    patch_count, _, feature_height, feature_width = logits.shape
    if len(patch_rects) != patch_count:
        raise ValueError("patch_rects length must match logits patch count")
    source_height = feature_height * feature_stride
    source_width = feature_width * feature_stride
    if any(
        rect.width != source_width or rect.height != source_height
        for rect in patch_rects
    ):
        raise ValueError("patch rect size must match logits spatial size times feature_stride")

    probabilities = F.interpolate(
        torch.sigmoid(logits),
        size=(source_height, source_width),
        mode="bilinear",
        align_corners=False,
    )
    squared_differences: list[Tensor] = []
    for first_index, first in enumerate(patch_rects):
        for second_index in range(first_index + 1, patch_count):
            second = patch_rects[second_index]
            left = max(first.x, second.x)
            top = max(first.y, second.y)
            right = min(first.right, second.right)
            bottom = min(first.bottom, second.bottom)
            if left >= right or top >= bottom:
                continue
            first_crop = probabilities[
                first_index,
                :,
                top - first.y : bottom - first.y,
                left - first.x : right - first.x,
            ]
            second_crop = probabilities[
                second_index,
                :,
                top - second.y : bottom - second.y,
                left - second.x : right - second.x,
            ]
            squared_differences.append((first_crop - second_crop).square().reshape(-1))

    if not squared_differences:
        return logits.sum() * 0.0
    return torch.cat(squared_differences).mean()


__all__ = [
    "absent_class_hard_negative_loss",
    "derive_train_positive_class_weights",
    "overlap_consistency_loss",
    "positive_spatial_mil_loss",
]
