"""Losses for weakly supervised spatial logits."""

from collections.abc import Sequence
from math import ceil, isfinite, log

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


def _bag_max_logits(logits: Tensor) -> Tensor:
    if logits.ndim == 4:
        return torch.amax(logits, dim=(-2, -1))
    if logits.ndim == 5:
        return torch.amax(logits, dim=(1, -2, -1))
    raise ValueError("logits must be a 4D patch tensor or 5D bag tensor")


def positive_spatial_mil_loss(
    logits: Tensor,
    targets: Tensor,
    class_weights: Tensor | None = None,
) -> Tensor:
    """Reward one strong spatial response for each present class."""

    losses = F.softplus(-_bag_max_logits(logits))
    if class_weights is not None:
        losses = losses * class_weights.to(device=logits.device, dtype=logits.dtype)
    positive_logits = losses[targets == 1]
    if positive_logits.numel() == 0:
        return logits.sum() * 0.0
    return positive_logits.mean()


def absent_class_hard_negative_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Penalize the strongest spatial response for each absent class."""

    absent_logits = _bag_max_logits(logits)[targets == 0]
    if absent_logits.numel() == 0:
        return logits.sum() * 0.0
    return F.softplus(absent_logits).mean()


def positive_spatial_lse_loss(
    logits: Tensor,
    targets: Tensor,
    class_weights: Tensor | None = None,
    *,
    temperature: float = 0.5,
) -> Tensor:
    """Reward distributed high spatial evidence for every present class."""

    if not isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite number")
    flattened = _flatten_spatial_logits(logits)
    pooled = temperature * (
        torch.logsumexp(flattened / temperature, dim=-1) - log(flattened.shape[-1])
    )
    losses = F.softplus(-pooled)
    if class_weights is not None:
        losses = losses * class_weights.to(device=logits.device, dtype=logits.dtype)
    present = losses[targets == 1]
    if present.numel() == 0:
        return logits.sum() * 0.0
    return present.mean()


def positive_spatial_topk_loss(
    logits: Tensor,
    targets: Tensor,
    class_weights: Tensor | None = None,
    *,
    top_fraction: float = 0.01,
) -> Tensor:
    """Reward localized evidence in the highest-scoring present-class locations."""

    if not isfinite(top_fraction) or not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be finite in the range (0, 1]")
    flattened = _flatten_spatial_logits(logits)
    count = max(1, ceil(flattened.shape[-1] * top_fraction))
    pooled = torch.topk(flattened, count, dim=-1).values.mean(dim=-1)
    losses = F.softplus(-pooled)
    if class_weights is not None:
        losses = losses * class_weights.to(device=logits.device, dtype=logits.dtype)
    present = losses[targets == 1]
    if present.numel() == 0:
        return logits.sum() * 0.0
    return present.mean()


def dense_absent_class_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    hardest_fraction: float = 0.01,
) -> Tensor:
    """Suppress every absent-class location and equally weight its hardest tail."""

    if not isfinite(hardest_fraction) or not 0 < hardest_fraction <= 1:
        raise ValueError("hardest_fraction must be finite in the range (0, 1]")
    losses = F.softplus(_flatten_spatial_logits(logits))
    absent = losses[targets == 0]
    if absent.numel() == 0:
        return logits.sum() * 0.0
    hardest_count = max(1, ceil(absent.shape[-1] * hardest_fraction))
    hardest = torch.topk(absent, hardest_count, dim=-1).values
    return 0.5 * (absent.mean() + hardest.mean())


def present_sparse_budget_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    probability_budget: float = 0.01,
) -> Tensor:
    """Penalize present-class mean probability only above its sparse budget."""

    if not isfinite(probability_budget) or not 0 <= probability_budget <= 1:
        raise ValueError("probability_budget must be finite in the range [0, 1]")
    mean_probabilities = torch.sigmoid(_flatten_spatial_logits(logits)).mean(dim=-1)
    present = mean_probabilities[targets == 1]
    if present.numel() == 0:
        return logits.sum() * 0.0
    return F.relu(present - probability_budget).square().mean()


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


def _flatten_spatial_logits(logits: Tensor) -> Tensor:
    if logits.ndim == 4:
        return logits.flatten(start_dim=2)
    if logits.ndim == 5:
        return logits.permute(0, 2, 1, 3, 4).flatten(start_dim=2)
    raise ValueError("logits must be a 4D patch tensor or 5D bag tensor")


__all__ = [
    "absent_class_hard_negative_loss",
    "dense_absent_class_loss",
    "derive_train_positive_class_weights",
    "overlap_consistency_loss",
    "positive_spatial_lse_loss",
    "positive_spatial_mil_loss",
    "positive_spatial_topk_loss",
    "present_sparse_budget_loss",
]
