"""Losses for weakly supervised spatial logits."""

import torch
import torch.nn.functional as F
from torch import Tensor


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


__all__ = ["absent_class_hard_negative_loss", "positive_spatial_mil_loss"]
