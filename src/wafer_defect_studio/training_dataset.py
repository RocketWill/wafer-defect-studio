"""Native-resolution model patch samples for a Dataset Snapshot."""

from __future__ import annotations

from math import isfinite

import numpy as np
import torch

from .detection_windows import Rect
from .normalization import NormalizationBounds
from .training_protocol import TrainingConfig


class TrainingDatasetError(ValueError):
    """Raised when a native source or patch request is invalid."""


_NATIVE_RANGES = {
    "uint8": (np.dtype(np.uint8), 0, 255, torch.uint8),
    "uint16": (np.dtype(np.uint16), 0, 65535, torch.uint16),
}


def enumerate_model_patch_rects(
    grid_rect: Rect,
    config: TrainingConfig,
) -> tuple[Rect, ...]:
    """Cover one frozen Annotation Grid with ordered source-coordinate patches."""

    config.validate_patch_geometry(grid_rect.width, grid_rect.height)
    patch_size = int(config.patch_size)
    patch_stride = int(config.patch_stride)
    x_starts = _patch_starts(grid_rect.width, patch_size, patch_stride)
    y_starts = _patch_starts(grid_rect.height, patch_size, patch_stride)
    return tuple(
        Rect(
            grid_rect.x + left,
            grid_rect.y + top,
            patch_size,
            patch_size,
        )
        for top in y_starts
        for left in x_starts
    )


def _patch_starts(length: int, patch_size: int, patch_stride: int) -> tuple[int, ...]:
    starts = list(range(0, length - patch_size + 1, patch_stride))
    edge = length - patch_size
    if starts[-1] != edge:
        starts.append(edge)
    return tuple(starts)


def extract_model_patch(
    source: np.ndarray | torch.Tensor,
    bounds: NormalizationBounds,
    *,
    top: int,
    left: int,
    size: int | tuple[int, int],
) -> torch.Tensor:
    """Return one normalized, reflect-padded grayscale patch as ``(3, H, W)``.

    ``top`` and ``left`` are the source-image coordinates of the requested
    patch's top-left pixel.  Coordinates may be outside the image; reflection
    supplies those pixels without changing the source array or tensor.
    """

    source_array, source_dtype = _source_array(source)
    if source_array.ndim != 2 or source_array.shape[0] == 0 or source_array.shape[1] == 0:
        raise TrainingDatasetError("source must be a non-empty grayscale HxW image")
    if not isinstance(bounds, NormalizationBounds):
        raise TrainingDatasetError("bounds must be a NormalizationBounds value")
    if not isinstance(top, int) or isinstance(top, bool):
        raise TrainingDatasetError("top must be an integer")
    if not isinstance(left, int) or isinstance(left, bool):
        raise TrainingDatasetError("left must be an integer")
    if isinstance(size, int) and not isinstance(size, bool):
        patch_width = patch_height = size
    elif (
        isinstance(size, tuple)
        and len(size) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in size)
    ):
        patch_width, patch_height = size
    else:
        raise TrainingDatasetError("size must be a positive integer or (width, height)")
    if patch_width < 1 or patch_height < 1:
        raise TrainingDatasetError("size values must be positive integers")

    _, source_min, source_max, _ = _NATIVE_RANGES[source_dtype]
    if bounds.dtype != source_dtype:
        raise TrainingDatasetError(
            f"Normalization bounds dtype {bounds.dtype!r} does not match source {source_dtype!r}"
        )
    if bounds.source_min != source_min or bounds.source_max != source_max:
        raise TrainingDatasetError("Normalization bounds have invalid native range metadata")
    if (
        not isinstance(bounds.low, (int, float))
        or isinstance(bounds.low, bool)
        or not isinstance(bounds.high, (int, float))
        or isinstance(bounds.high, bool)
        or not isfinite(float(bounds.low))
        or not isfinite(float(bounds.high))
        or not source_min <= bounds.low < bounds.high <= source_max
    ):
        raise TrainingDatasetError("Normalization bounds must be ordered native values")

    row_indices = _reflect_indices(top, patch_height, source_array.shape[0])
    column_indices = _reflect_indices(left, patch_width, source_array.shape[1])
    native_patch = source_array[np.ix_(row_indices, column_indices)]
    normalized = native_patch.astype(np.float64, copy=True)
    normalized -= float(bounds.low)
    normalized /= float(bounds.high) - float(bounds.low)
    np.clip(normalized, 0.0, 1.0, out=normalized)
    sample = torch.from_numpy(normalized.astype(np.float32, copy=False)).contiguous()
    return sample.unsqueeze(0).expand(3, -1, -1).clone()


def _source_array(source: np.ndarray | torch.Tensor) -> tuple[np.ndarray, str]:
    if isinstance(source, np.ndarray):
        source_array = source
        dtype = source_array.dtype
        for name, (expected, *_rest) in _NATIVE_RANGES.items():
            if dtype == expected:
                return source_array, name
        raise TrainingDatasetError("source must use uint8 or uint16 dtype")
    if isinstance(source, torch.Tensor):
        for name, (*_rest, torch_dtype) in _NATIVE_RANGES.items():
            if source.dtype == torch_dtype:
                return source.detach().cpu().numpy(), name
        raise TrainingDatasetError("source must use torch.uint8 or torch.uint16 dtype")
    raise TrainingDatasetError("source must be a numpy array or torch tensor")


def _reflect_indices(start: int, size: int, length: int) -> np.ndarray:
    if length == 1:
        return np.zeros(size, dtype=np.int64)
    period = 2 * (length - 1)
    values = np.arange(start, start + size, dtype=np.int64)
    folded = np.mod(values, period)
    return np.where(folded < length, folded, period - folded)


__all__ = ["TrainingDatasetError", "enumerate_model_patch_rects", "extract_model_patch"]
