"""Small, explicit deterministic transforms for model patch samples."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor


SPATIAL_MIL_V4_AUGMENTATION_POLICY = {
    "name": "spatial_mil_v4_defect_preserving_affine",
    "contrast": [0.9, 1.1],
    "brightness": [-0.03, 0.03],
    "seed": "run_seed",
    "seed_formula": "run_seed + epoch * 1_000_003 + bag_index",
}


class TrainingAugmentationError(ValueError):
    """Raised when a transform configuration or sample is invalid."""


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Editable transform values; all transforms are disabled by default.

    ``brightness`` is an additive offset in normalized model-input units and
    ``contrast`` is a multiplier around 0.5.  ``noise_std`` is Gaussian noise
    in the same units and uses ``seed`` for reproducibility.
    """

    horizontal_flip: bool = False
    vertical_flip: bool = False
    brightness: float = 0.0
    contrast: float = 1.0
    noise_std: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.horizontal_flip, bool) or not isinstance(self.vertical_flip, bool):
            raise TrainingAugmentationError("flip settings must be boolean")
        for name, value in (
            ("brightness", self.brightness),
            ("contrast", self.contrast),
            ("noise_std", self.noise_std),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
                raise TrainingAugmentationError(f"{name} must be a finite number")
        if self.contrast < 0:
            raise TrainingAugmentationError("contrast must be non-negative")
        if self.noise_std < 0:
            raise TrainingAugmentationError("noise_std must be non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TrainingAugmentationError("seed must be an integer")


@dataclass(frozen=True, slots=True)
class TransformPreview:
    """One transformed sample plus the immutable facts shown to the user."""

    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    config: AugmentationConfig
    transformed: Tensor


def apply_augmentation(
    sample: Tensor,
    config: AugmentationConfig | None = None,
) -> Tensor:
    """Apply only the transforms explicitly enabled in ``config``.

    Samples are native-resolution ``(C, H, W)`` CPU tensors from
    :func:`training_dataset.extract_model_patch`; no resize or implicit
    augmentation is performed.
    """

    _validate_sample(sample)
    config = _coerce_config(config)
    result = sample.clone()
    if config.horizontal_flip:
        result = torch.flip(result, dims=(-1,))
    if config.vertical_flip:
        result = torch.flip(result, dims=(-2,))
    if config.contrast != 1.0:
        result = (result - 0.5) * float(config.contrast) + 0.5
    if config.brightness != 0.0:
        result = result + float(config.brightness)
    if config.noise_std != 0.0:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        result = result + torch.randn(
            result.shape,
            generator=generator,
            dtype=result.dtype,
            device="cpu",
        ) * float(config.noise_std)
    if config != AugmentationConfig():
        result = result.clamp(0.0, 1.0)
    return result


def apply_spatial_mil_v4_augmentation(sample: Tensor, *, effective_seed: int) -> Tensor:
    """Apply the frozen train-only pointwise affine transform for one v4 bag."""

    _validate_sample(sample)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(effective_seed)
    contrast = 0.9 + 0.2 * float(torch.rand((), generator=generator))
    brightness = -0.03 + 0.06 * float(torch.rand((), generator=generator))
    return ((sample - 0.5) * contrast + 0.5 + brightness).clamp(0.0, 1.0)


def preview_transformed_sample(
    sample: Tensor,
    config: AugmentationConfig | None = None,
) -> TransformPreview:
    """Return one transformed sample and its input/output shape facts."""

    _validate_sample(sample)
    config = _coerce_config(config)
    transformed = apply_augmentation(sample, config)
    return TransformPreview(tuple(sample.shape), tuple(transformed.shape), config, transformed)


def _coerce_config(config: AugmentationConfig | None) -> AugmentationConfig:
    if config is None:
        return AugmentationConfig()
    if not isinstance(config, AugmentationConfig):
        raise TrainingAugmentationError("config must be an AugmentationConfig value")
    return config


def _validate_sample(sample: Tensor) -> None:
    if not isinstance(sample, Tensor):
        raise TrainingAugmentationError("sample must be a torch.Tensor")
    if sample.device.type != "cpu":
        raise TrainingAugmentationError("sample must be a CPU tensor")
    if sample.ndim != 3 or min(sample.shape) < 1:
        raise TrainingAugmentationError("sample must have non-empty (C, H, W) shape")
    if not torch.is_floating_point(sample):
        raise TrainingAugmentationError("sample must use a floating-point dtype")


__all__ = [
    "AugmentationConfig",
    "TrainingAugmentationError",
    "TransformPreview",
    "apply_augmentation",
    "apply_spatial_mil_v4_augmentation",
    "SPATIAL_MIL_V4_AUGMENTATION_POLICY",
    "preview_transformed_sample",
]
