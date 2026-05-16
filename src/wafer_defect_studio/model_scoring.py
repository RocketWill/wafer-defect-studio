"""Deterministic scoring of bundled samples with a project checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model_registry import load_project_checkpoint, max_pool_patch_logits
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
    checkpoint_format = checkpoint["checkpoint_format"]
    is_patch_bag_checkpoint = checkpoint_format in {
        "wafer_defect_studio.resnet18.v3",
        "wafer_defect_studio.resnet18.v4",
    }
    is_spatial_checkpoint = checkpoint_format == "wafer_defect_studio.resnet18.v4"
    input_size = checkpoint["input_size"]
    if is_patch_bag_checkpoint:
        if bundle.version != 2:
            raise ModelScoringError(
                f"{checkpoint_format} checkpoint requires a Patch Bag bundle"
            )
        if any(
            rect.width != checkpoint["patch_size"] or rect.height != checkpoint["patch_size"]
            for bag in bundle.patch_bags
            for rect in bag.patches
        ):
            raise ModelScoringError("Patch Bag descriptors do not match checkpoint patch_size")
    else:
        sizes = {(sample.width, sample.height) for sample in bundle.samples}
        if sizes != {(input_size["width"], input_size["height"])}:
            raise ModelScoringError("bundle sample size does not match checkpoint input_size")
    try:
        dataset = TrainingPatchDataset(bundle, split)
    except ValueError as error:
        raise ModelScoringError(str(error)) from error
    resolved_device = next(model.parameters()).device
    true_batches: list[np.ndarray] = []
    score_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        if is_patch_bag_checkpoint:
            for index in range(len(dataset)):
                inputs, target = dataset[index]
                patch_logits = model(inputs.to(resolved_device))
                logits = (
                    torch.amax(patch_logits, dim=(0, 2, 3)).unsqueeze(0)
                    if is_spatial_checkpoint
                    else max_pool_patch_logits(patch_logits.unsqueeze(0))
                )
                true_batches.append(target.unsqueeze(0).numpy())
                score_batches.append(torch.sigmoid(logits).cpu().numpy())
        else:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
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
