"""Deterministic, independent Class Threshold selection for Grid Evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from typing import Mapping, Sequence

import numpy as np

from .grid_evaluation import ClassMetrics, compute_grid_metrics


DEFAULT_MIN_RECALL_TARGET = 0.95
DEFAULT_MAX_FPR_TARGET = 0.05


class ThresholdPolicy(str, Enum):
    """The metric constraint used to choose one class's threshold."""

    MAX_F1 = "max_f1"
    MIN_RECALL = "min_recall"
    MAX_FPR = "max_fpr"


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    """One independently selected Class Threshold and its resulting metrics."""

    class_name: str
    policy: ThresholdPolicy
    threshold: float
    metrics: ClassMetrics
    target: float | None = None
    target_satisfied: bool | None = None

    @property
    def target_label(self) -> str:
        """Describe a target as an optimization target, never a guarantee."""

        if self.target is None:
            return ""
        metric_name = "minimum recall" if self.policy is ThresholdPolicy.MIN_RECALL else "maximum FPR"
        return f"{metric_name} target {self.target:.0%} (optimization target, not guarantee)"

    @property
    def satisfied(self) -> bool | None:
        """Short alias used by evaluation views and persistence adapters."""

        return self.target_satisfied


@dataclass(frozen=True, slots=True)
class ThresholdOptimization:
    """All per-class selections; no Training Run or other state is modified."""

    per_class: tuple[ThresholdSelection, ...]
    min_recall_target: float = DEFAULT_MIN_RECALL_TARGET
    max_fpr_target: float = DEFAULT_MAX_FPR_TARGET

    @property
    def by_class(self) -> dict[str, ThresholdSelection]:
        return {selection.class_name: selection for selection in self.per_class}

    @property
    def thresholds(self) -> tuple[float, ...]:
        return tuple(selection.threshold for selection in self.per_class)


def optimize_thresholds(
    y_true: Sequence[Sequence[int]] | np.ndarray,
    y_score: Sequence[Sequence[float]] | np.ndarray,
    *,
    policy: ThresholdPolicy | str | Sequence[ThresholdPolicy | str] | Mapping[str, ThresholdPolicy | str] = ThresholdPolicy.MAX_F1,
    policies: ThresholdPolicy | str | Sequence[ThresholdPolicy | str] | Mapping[str, ThresholdPolicy | str] | None = None,
    min_recall_target: float = DEFAULT_MIN_RECALL_TARGET,
    max_fpr_target: float = DEFAULT_MAX_FPR_TARGET,
    class_names: Sequence[str] | None = None,
) -> ThresholdOptimization:
    """Choose deterministic Class Thresholds independently for each class.

    Candidate thresholds are every observed score plus the inclusive 0 and 1
    boundaries.  ``MAX_F1`` picks the highest F1, then the highest threshold;
    ``MIN_RECALL`` picks the highest threshold meeting its target; and
    ``MAX_FPR`` picks the greatest FPR not exceeding its target, then the
    lowest threshold.  If a target cannot be met, the best available candidate
    is returned and ``target_satisfied`` is explicitly false.
    """

    if policies is not None:
        if policy != ThresholdPolicy.MAX_F1:
            raise ValueError("pass either policy or policies, not both")
        policy = policies
    _validate_target(min_recall_target, "min_recall_target")
    _validate_target(max_fpr_target, "max_fpr_target")
    baseline = compute_grid_metrics(y_true, y_score, class_names=class_names)
    names = tuple(metric.class_name for metric in baseline.per_class)
    resolved_policies = _resolve_policies(policy, names)
    scores = np.asarray(y_score, dtype=float)
    selections: list[ThresholdSelection] = []
    for index, (name, chosen_policy) in enumerate(zip(names, resolved_policies)):
        candidates = _candidate_thresholds(scores[:, index])
        metrics = tuple(
            compute_grid_metrics(
                np.asarray(y_true)[:, index : index + 1],
                scores[:, index : index + 1],
                thresholds=(threshold,),
                class_names=(name,),
            ).per_class[0]
            for threshold in candidates
        )
        target = (
            min_recall_target
            if chosen_policy is ThresholdPolicy.MIN_RECALL
            else max_fpr_target
            if chosen_policy is ThresholdPolicy.MAX_FPR
            else None
        )
        selected, satisfied = _select(metrics, chosen_policy, target)
        selections.append(
            ThresholdSelection(
                name,
                chosen_policy,
                selected.threshold,
                selected,
                target,
                satisfied,
            )
        )
    return ThresholdOptimization(tuple(selections), min_recall_target, max_fpr_target)


def optimize_class_thresholds(*args: object, **kwargs: object) -> ThresholdOptimization:
    """Compatibility spelling for callers that name the per-class operation."""

    return optimize_thresholds(*args, **kwargs)  # type: ignore[arg-type]


def _candidate_thresholds(scores: np.ndarray) -> tuple[float, ...]:
    return tuple(sorted({0.0, 1.0, *(float(score) for score in np.unique(scores))}))


def _select(
    metrics: tuple[ClassMetrics, ...],
    policy: ThresholdPolicy,
    target: float | None,
) -> tuple[ClassMetrics, bool | None]:
    if policy is ThresholdPolicy.MAX_F1:
        return max(metrics, key=lambda item: (item.f1, item.threshold)), None
    if policy is ThresholdPolicy.MIN_RECALL:
        feasible = tuple(item for item in metrics if item.recall >= target)  # type: ignore[operator]
        if feasible:
            return max(feasible, key=lambda item: item.threshold), True
        best = max(metrics, key=lambda item: (item.recall, item.threshold))
        return best, False
    feasible = tuple(item for item in metrics if item.fpr <= target)  # type: ignore[operator]
    if feasible:
        return max(feasible, key=lambda item: (item.fpr, -item.threshold)), True
    best = min(metrics, key=lambda item: (item.fpr, -item.threshold))
    return best, False


def _resolve_policies(
    value: ThresholdPolicy | str | Sequence[ThresholdPolicy | str] | Mapping[str, ThresholdPolicy | str],
    names: tuple[str, ...],
) -> tuple[ThresholdPolicy, ...]:
    if isinstance(value, Mapping):
        return tuple(_coerce_policy(value.get(name, ThresholdPolicy.MAX_F1)) for name in names)
    if isinstance(value, (str, ThresholdPolicy)):
        chosen = _coerce_policy(value)
        return (chosen,) * len(names)
    try:
        values = tuple(value)
    except TypeError as error:
        raise ValueError("policy must be a policy or one policy per class") from error
    if len(values) != len(names):
        raise ValueError("policy sequence must contain one policy per class")
    return tuple(_coerce_policy(item) for item in values)


def _coerce_policy(value: object) -> ThresholdPolicy:
    if isinstance(value, ThresholdPolicy):
        return value
    if isinstance(value, str):
        aliases = {
            "f1": ThresholdPolicy.MAX_F1,
            "maximize_f1": ThresholdPolicy.MAX_F1,
            "recall": ThresholdPolicy.MIN_RECALL,
            "minimum_recall": ThresholdPolicy.MIN_RECALL,
            "fpr": ThresholdPolicy.MAX_FPR,
            "maximum_fpr": ThresholdPolicy.MAX_FPR,
        }
        try:
            if value in aliases:
                return aliases[value]
            return ThresholdPolicy(value)
        except ValueError as error:
            raise ValueError(f"unsupported threshold policy: {value!r}") from error
    raise ValueError(f"unsupported threshold policy: {value!r}")


def _validate_target(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number from 0 to 1")
    number = float(value)
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite number from 0 to 1")


__all__ = [
    "DEFAULT_MAX_FPR_TARGET",
    "DEFAULT_MIN_RECALL_TARGET",
    "ThresholdOptimization",
    "ThresholdPolicy",
    "ThresholdSelection",
    "optimize_class_thresholds",
    "optimize_thresholds",
]
