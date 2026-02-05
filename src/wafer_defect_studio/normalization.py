"""Deterministic percentile bounds for native grayscale source pixels."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Iterable


_SOURCE_RANGES = {"uint8": (0, 255), "uint16": (0, 65535)}


@dataclass(frozen=True)
class NativePixelSample:
    dtype: str
    pixels: Iterable[int]


@dataclass(frozen=True)
class NormalizationBounds:
    dtype: str
    source_min: int
    source_max: int
    low: float
    high: float
    low_percentile: float
    high_percentile: float


def compute_percentile_bounds(
    samples: Iterable[NativePixelSample],
    low_percentile: float,
    high_percentile: float,
) -> tuple[NormalizationBounds, ...]:
    """Return one fixed native-value bound per source dtype."""

    low_percentile = _percentile(low_percentile, "low_percentile")
    high_percentile = _percentile(high_percentile, "high_percentile")
    if low_percentile >= high_percentile:
        raise ValueError("low_percentile must be less than high_percentile")

    grouped: dict[str, list[int]] = {}
    for sample in samples:
        if not isinstance(sample, NativePixelSample) or sample.dtype not in _SOURCE_RANGES:
            raise ValueError("samples must contain uint8 or uint16 NativePixelSample values")
        source_min, source_max = _SOURCE_RANGES[sample.dtype]
        values = list(sample.pixels)
        if not values or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not source_min <= value <= source_max
            for value in values
        ):
            raise ValueError(f"{sample.dtype} pixels must be non-empty native integers")
        grouped.setdefault(sample.dtype, []).extend(values)
    if not grouped:
        raise ValueError("samples must not be empty")

    result = []
    for dtype in sorted(grouped, key=lambda item: _SOURCE_RANGES[item][1]):
        values = sorted(grouped[dtype])
        low = _linear_percentile(values, low_percentile)
        high = _linear_percentile(values, high_percentile)
        if low >= high:
            raise ValueError(f"{dtype} percentile bounds must be ordered")
        source_min, source_max = _SOURCE_RANGES[dtype]
        result.append(
            NormalizationBounds(
                dtype,
                source_min,
                source_max,
                low,
                high,
                low_percentile,
                high_percentile,
            )
        )
    return tuple(result)


def _percentile(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number from 0 to 100")
    result = float(value)
    if not isfinite(result) or not 0 <= result <= 100:
        raise ValueError(f"{name} must be a finite number from 0 to 100")
    return result


def _linear_percentile(values: list[int], percentile: float) -> float:
    rank = (len(values) - 1) * percentile / 100
    lower = floor(rank)
    fraction = rank - lower
    return float(values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * fraction)
