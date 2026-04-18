"""Deterministic overlap blending for per-class inference-window scores."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np

from .detection_windows import Window


@dataclass(frozen=True, slots=True)
class StitchedConfidenceMap:
    """Per-class confidence values and explicit source-pixel coverage.

    ``confidence`` has shape ``(source_height, source_width, class_count)``.
    Pixels with no contributing window are NaN and have zero ``coverage``;
    coverage otherwise counts the contributing source rectangles.
    """

    confidence: np.ndarray
    coverage: np.ndarray

    @property
    def values(self) -> np.ndarray:
        """Alias for callers that refer to the map values directly."""

        return self.confidence


def stitch_window_scores(
    windows: Sequence[Window],
    scores: Sequence[Sequence[float]] | np.ndarray,
    source_width: int,
    source_height: int,
    class_count: int,
    center_weight: str = "linear",
) -> StitchedConfidenceMap:
    """Blend scores from overlapping windows into a native source-space map.

    ``scores`` may contain one class vector per window, ``(N, C)``; those
    values are broadcast over each window's covered source rectangle.  A
    local score map, ``(N, window_height, window_width, C)``, is also accepted
    and is sampled at the source-coordinate transform supplied by ``Window``.
    Only ``source_rect`` contributes, so reflect padding never creates pixels
    outside the source image.
    """

    _positive_integer("source_width", source_width)
    _positive_integer("source_height", source_height)
    _positive_integer("class_count", class_count)
    resolved_windows = tuple(windows)
    if any(not isinstance(window, Window) for window in resolved_windows):
        raise ValueError("windows must contain Window values")
    for window in resolved_windows:
        if window.source_width != source_width or window.source_height != source_height:
            raise ValueError("window source dimensions must match the requested map")

    mode = _center_weight(center_weight)
    values = _scores(scores, len(resolved_windows), class_count)
    is_local = values.ndim == 4
    if is_local:
        expected_height = {window.height for window in resolved_windows}
        expected_width = {window.width for window in resolved_windows}
        if len(expected_height) > 1 or len(expected_width) > 1:
            raise ValueError("local scores require equal window dimensions")
        window_height = next(iter(expected_height), 0)
        window_width = next(iter(expected_width), 0)
        if values.shape[1:3] != (window_height, window_width):
            raise ValueError("local scores must match the model window dimensions")

    totals = np.zeros((source_height, source_width, class_count), dtype=np.float64)
    weights = np.zeros((source_height, source_width), dtype=np.float64)
    coverage = np.zeros((source_height, source_width), dtype=np.int32)
    for index, window in enumerate(resolved_windows):
        rect = window.source_rect
        source_y = slice(rect.y, rect.bottom)
        source_x = slice(rect.x, rect.right)
        if mode == "uniform":
            contribution_weights = 1.0
        else:
            x = np.arange(rect.x, rect.right, dtype=np.float64)
            y = np.arange(rect.y, rect.bottom, dtype=np.float64)
            half_width = max(rect.width / 2.0, 0.5)
            half_height = max(rect.height / 2.0, 0.5)
            x_distance = np.abs((x + 0.5) - window.center.x) / half_width
            y_distance = np.abs((y + 0.5) - window.center.y) / half_height
            distance = np.maximum(y_distance[:, None], x_distance[None, :])
            contribution_weights = 1.0 - 0.5 * np.minimum(1.0, distance)
        if is_local:
            local_y = rect.y - window.read_rect.y
            local_x = rect.x - window.read_rect.x
            sample = values[
                index,
                local_y : local_y + rect.height,
                local_x : local_x + rect.width,
                :,
            ]
        else:
            sample = values[index, :]
        totals[source_y, source_x, :] += sample * np.asarray(contribution_weights)[..., None]
        weights[source_y, source_x] += contribution_weights
        coverage[source_y, source_x] += 1

    confidence = np.full_like(totals, np.nan)
    covered = coverage > 0
    confidence[covered, :] = totals[covered, :] / weights[covered, None]
    return StitchedConfidenceMap(confidence=confidence, coverage=coverage)


def _scores(value: object, window_count: int, class_count: int) -> np.ndarray:
    try:
        scores = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("scores must be a numeric (N, C) or (N, H, W, C) array") from error
    if scores.ndim not in (2, 4) or scores.shape[0] != window_count:
        raise ValueError("scores must have one row per window and preserve the class axis")
    if scores.shape[-1] != class_count:
        raise ValueError("scores class axis must match class_count")
    if not np.all(np.isfinite(scores)) or not np.all((scores >= 0.0) & (scores <= 1.0)):
        raise ValueError("scores must contain finite values from 0 to 1")
    return scores


def _center_weight(value: object) -> str:
    if not isinstance(value, str) or value not in {"linear", "uniform"}:
        raise ValueError("center_weight must be 'linear' or 'uniform'")
    return value


def _pixel_weight(window: Window, x: int, y: int, mode: str) -> float:
    if mode == "uniform":
        return 1.0
    # Keep edge contributions non-zero while giving the geometric center the
    # greatest influence.  Normalization happens after all windows contribute.
    half_width = max(window.source_rect.width / 2.0, 0.5)
    half_height = max(window.source_rect.height / 2.0, 0.5)
    distance = max(
        abs((x + 0.5) - window.center.x) / half_width,
        abs((y + 0.5) - window.center.y) / half_height,
    )
    return 1.0 - 0.5 * min(1.0, distance)


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


__all__ = ["StitchedConfidenceMap", "stitch_window_scores"]
