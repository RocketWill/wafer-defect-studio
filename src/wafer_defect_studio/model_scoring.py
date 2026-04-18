"""Deterministic scoring of bundled samples with a project checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model_registry import load_project_checkpoint
from .training_input_bundle import TrainingInputBundle, TrainingInputBundleError
from .training_patch_dataset import TrainingPatchDataset


class ModelScoringError(ValueError):
    """Raised when a checkpoint cannot score a requested bundle split."""


@dataclass(frozen=True, slots=True)
class ModelScores:
    class_codes: tuple[str, ...]
    y_true: np.ndarray
    y_score: np.ndarray


def score_training_bundle(
    bundle_path: str | Path,
    checkpoint_path: str | Path,
    *,
    split: str,
    device: str | torch.device | None = "cpu",
    batch_size: int = 32,
) -> ModelScores:
    """Return ordered multi-label targets and sigmoid scores for one split."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ModelScoringError("batch_size must be a positive integer")
    try:
        bundle = TrainingInputBundle.from_json(
            Path(bundle_path).expanduser().resolve().read_text(encoding="utf-8")
        )
    except (OSError, TrainingInputBundleError) as error:
        raise ModelScoringError(f"unable to read training input bundle: {bundle_path}") from error
    try:
        model, checkpoint = load_project_checkpoint(
            checkpoint_path,
            device=device,
            expected_class_codes=bundle.class_codes,
        )
    except ValueError as error:
        raise ModelScoringError(str(error)) from error
    input_size = checkpoint["input_size"]
    sizes = {(sample.width, sample.height) for sample in bundle.samples}
    if sizes != {(input_size["width"], input_size["height"])}:
        raise ModelScoringError("bundle sample size does not match checkpoint input_size")
    try:
        dataset = TrainingPatchDataset(bundle, split)
    except ValueError as error:
        raise ModelScoringError(str(error)) from error
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    resolved_device = next(model.parameters()).device
    true_batches: list[np.ndarray] = []
    score_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs.to(resolved_device))
            true_batches.append(targets.cpu().numpy())
            score_batches.append(torch.sigmoid(logits).cpu().numpy())
    return ModelScores(
        tuple(checkpoint["class_codes"]),
        np.concatenate(true_batches, axis=0),
        np.concatenate(score_batches, axis=0),
    )


__all__ = ["ModelScores", "ModelScoringError", "score_training_bundle"]
