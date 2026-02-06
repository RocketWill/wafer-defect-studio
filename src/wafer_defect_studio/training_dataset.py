"""Native-resolution model patch samples for a Dataset Snapshot."""

from __future__ import annotations

from math import isfinite

import numpy as np
import torch

from .normalization import NormalizationBounds


class TrainingDatasetError(ValueError):
    """Raised when a native source or patch request is invalid."""


_NATIVE_RANGES = {
    "uint8": (np.dtype(np.uint8), 0, 255, torch.uint8),
    "uint16": (np.dtype(np.uint16), 0, 65535, torch.uint16),
}


def extract_model_patch(
    source: np.ndarray | torch.Tensor,
    bounds: NormalizationBounds,
    *,
    top: int,
    left: int,
    size: int,
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
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise TrainingDatasetError("size must be a positive integer")

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

    row_indices = _reflect_indices(top, size, source_array.shape[0])
    column_indices = _reflect_indices(left, size, source_array.shape[1])
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


__all__ = ["TrainingDatasetError", "extract_model_patch"]
