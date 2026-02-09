"""Deterministic multi-label metrics for Grid Evaluation.

The evaluator intentionally treats every defect class as an independent
binary problem.  It consumes already materialized test-split labels and
scores; loading models, choosing thresholds, and persisting evaluations live
in later slices.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ClassMetrics:
    """Binary metrics and confusion counts for one defect class."""

    class_name: str
    precision: float
    recall: float
    f1: float
    support: int
    fpr: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    threshold: float = 0.5

    @property
    def tp(self) -> int:
        return self.true_positive

    @property
    def tn(self) -> int:
        return self.true_negative

    @property
    def fp(self) -> int:
        return self.false_positive

    @property
    def fn(self) -> int:
        return self.false_negative


@dataclass(frozen=True)
class GridMetrics:
    """Macro F1 and independent per-class results."""

    macro_f1: float
    per_class: tuple[ClassMetrics, ...]

    @property
    def by_class(self) -> dict[str, ClassMetrics]:
        """Return a convenient class-code lookup without changing the result."""

        return {metric.class_name: metric for metric in self.per_class}


def compute_grid_metrics(
    y_true: Sequence[Sequence[int]] | np.ndarray,
    y_score: Sequence[Sequence[float]] | np.ndarray,
    *,
    thresholds: float | Sequence[float] | np.ndarray = 0.5,
    class_names: Sequence[str] | None = None,
) -> GridMetrics:
    """Compute deterministic independent multi-label classification metrics.

    ``y_true`` and ``y_score`` have shape ``(sample_count, class_count)``.
    Each score is compared with that class's threshold (inclusive).  Empty
    positive or negative denominators use an explicit zero metric while the
    confusion counts retain the reason for that zero.
    """

    truth = _binary_matrix(y_true, "y_true")
    scores = _score_matrix(y_score)
    if truth.shape != scores.shape:
        raise ValueError("y_true and y_score must have the same 2-D shape")
    sample_count, class_count = truth.shape
    if sample_count == 0 or class_count == 0:
        raise ValueError("y_true and y_score must contain at least one sample and class")

    resolved_thresholds = _thresholds(thresholds, class_count)
    names = _class_names(class_names, class_count)
    metrics: list[ClassMetrics] = []
    for index, name in enumerate(names):
        actual = truth[:, index]
        predicted = scores[:, index] >= resolved_thresholds[index]
        true_positive = int(np.count_nonzero(actual & predicted))
        true_negative = int(np.count_nonzero(~actual & ~predicted))
        false_positive = int(np.count_nonzero(~actual & predicted))
        false_negative = int(np.count_nonzero(actual & ~predicted))
        support = true_positive + false_negative

        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, support)
        f1 = _ratio(2.0 * precision * recall, precision + recall)
        fpr = _ratio(false_positive, false_positive + true_negative)
        metrics.append(
            ClassMetrics(
                name,
                precision,
                recall,
                f1,
                support,
                fpr,
                true_positive,
                true_negative,
                false_positive,
                false_negative,
                resolved_thresholds[index],
            )
        )
    return GridMetrics(
        macro_f1=float(sum(metric.f1 for metric in metrics) / len(metrics)),
        per_class=tuple(metrics),
    )


def _binary_matrix(value: object, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric 2-D matrix") from error
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix")
    if matrix.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain only binary numeric values")
    if not np.all(np.isfinite(matrix)) or not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError(f"{name} must contain only 0 or 1 values")
    return matrix.astype(bool, copy=False)


def _score_matrix(value: object) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("y_score must be a numeric 2-D matrix") from error
    if matrix.ndim != 2:
        raise ValueError("y_score must be a 2-D matrix")
    if not np.all(np.isfinite(matrix)) or not np.all((matrix >= 0.0) & (matrix <= 1.0)):
        raise ValueError("y_score must contain finite values from 0 to 1")
    return matrix


def _thresholds(value: object, class_count: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("thresholds must be a number or one value per class")
    try:
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("thresholds must be a number or one value per class") from error
    if values.ndim == 0:
        values = np.repeat(values, class_count)
    elif values.ndim != 1 or values.shape[0] != class_count:
        raise ValueError("thresholds must be a scalar or match the class count")
    if not np.all(np.isfinite(values)) or not np.all((values >= 0.0) & (values <= 1.0)):
        raise ValueError("thresholds must contain finite values from 0 to 1")
    return tuple(float(item) for item in values)


def _class_names(value: Sequence[str] | None, class_count: int) -> tuple[str, ...]:
    if value is None:
        return tuple(f"class_{index}" for index in range(class_count))
    if isinstance(value, (str, bytes)):
        raise ValueError("class_names must contain one non-empty string per class")
    try:
        names = tuple(value)
    except TypeError as error:
        raise ValueError("class_names must contain one non-empty string per class") from error
    if len(names) != class_count or any(
        not isinstance(name, str) or not name.strip() for name in names
    ):
        raise ValueError("class_names must contain one non-empty string per class")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique")
    return names


def _ratio(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0:
        return 0.0
    result = float(numerator) / float(denominator)
    return result if isfinite(result) else 0.0


__all__ = ["ClassMetrics", "GridMetrics", "compute_grid_metrics"]
