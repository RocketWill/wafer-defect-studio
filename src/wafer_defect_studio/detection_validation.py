"""Cell-level comparison of Detection Proposals with Grid Annotations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .annotation import GridAnnotation
from .grid_geometry import AnnotationGrid
from .proposal_conversion import preview_proposal_conversion
from .proposal_generation import DefectProposal


@dataclass(frozen=True, slots=True)
class ClassDetectionMetrics:
    """One-vs-rest counts and derived metrics for one Defect Class."""

    class_name: str
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def support(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def predicted_positive(self) -> int:
        return self.true_positive + self.false_positive

    @property
    def precision(self) -> float:
        return _ratio(self.true_positive, self.predicted_positive)

    @property
    def recall(self) -> float:
        return _ratio(self.true_positive, self.support)

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0.0:
            return 0.0
        return 2.0 * self.precision * self.recall / (self.precision + self.recall)

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "class_name": self.class_name,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "support": self.support,
            "predicted_positive": self.predicted_positive,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True, slots=True)
class DetectionAnnotationReport:
    """Cell-level Detection-versus-Annotation report."""

    evaluated_cells: int
    exact_cell_matches: int
    metrics: tuple[ClassDetectionMetrics, ...]

    @property
    def exact_match_rate(self) -> float:
        return _ratio(self.exact_cell_matches, self.evaluated_cells)

    @property
    def macro_f1(self) -> float:
        return sum(metric.f1 for metric in self.metrics) / len(self.metrics)

    def for_class(self, class_name: str) -> ClassDetectionMetrics:
        for metric in self.metrics:
            if metric.class_name == class_name:
                return metric
        raise KeyError(class_name)

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluated_cells": self.evaluated_cells,
            "exact_cell_matches": self.exact_cell_matches,
            "exact_match_rate": self.exact_match_rate,
            "macro_f1": self.macro_f1,
            "classes": [metric.to_dict() for metric in self.metrics],
        }


def compare_detection_proposals(
    annotations: Iterable[GridAnnotation],
    proposals: Iterable[DefectProposal],
    participating_grids: Sequence[AnnotationGrid],
    *,
    class_names: Sequence[str],
) -> DetectionAnnotationReport:
    """Compare proposal-overlapped cells with persisted multi-label annotations."""

    names = _class_names(class_names)
    grids = tuple(participating_grids)
    if not grids or any(not isinstance(grid, AnnotationGrid) for grid in grids):
        raise ValueError("participating_grids must contain at least one AnnotationGrid")
    grid_keys = {(grid.row, grid.column) for grid in grids}

    truth: dict[tuple[int, int], frozenset[str]] = {
        key: frozenset() for key in grid_keys
    }
    for annotation in annotations:
        if not isinstance(annotation, GridAnnotation):
            raise TypeError("annotations must contain GridAnnotation values")
        key = (annotation.row, annotation.column)
        if key not in grid_keys:
            raise ValueError("annotation is outside participating_grids")
        labels = frozenset(annotation.class_codes)
        if not labels <= set(names):
            raise ValueError("annotation class_codes must be in class_names")
        truth[key] = labels

    proposal_values = tuple(proposals)
    if any(not isinstance(proposal, DefectProposal) for proposal in proposal_values):
        raise TypeError("proposals must contain DefectProposal values")
    if len({proposal.proposal_id for proposal in proposal_values}) != len(proposal_values):
        raise ValueError("proposals must have unique proposal_id values")
    if any(proposal.class_name not in names for proposal in proposal_values):
        raise ValueError("proposal class_name must be in class_names")

    predicted: dict[tuple[int, int], frozenset[str]] = {
        key: frozenset() for key in grid_keys
    }
    if proposal_values:
        preview = preview_proposal_conversion(proposal_values, grids)
        predicted.update({
            (cell.row, cell.column): frozenset(cell.class_codes)
            for cell in preview.cells
        })

    exact_matches = sum(truth[key] == predicted[key] for key in grid_keys)
    metrics = tuple(
        _metrics_for_class(name, truth, predicted, grid_keys)
        for name in names
    )
    return DetectionAnnotationReport(len(grids), exact_matches, metrics)


def _metrics_for_class(
    class_name: str,
    truth: dict[tuple[int, int], frozenset[str]],
    predicted: dict[tuple[int, int], frozenset[str]],
    grid_keys: set[tuple[int, int]],
) -> ClassDetectionMetrics:
    true_positive = sum(class_name in truth[key] and class_name in predicted[key] for key in grid_keys)
    false_positive = sum(class_name not in truth[key] and class_name in predicted[key] for key in grid_keys)
    false_negative = sum(class_name in truth[key] and class_name not in predicted[key] for key in grid_keys)
    return ClassDetectionMetrics(class_name, true_positive, false_positive, false_negative)


def _class_names(value: Sequence[str]) -> tuple[str, ...]:
    names = tuple(value)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("class_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique")
    return names


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


__all__ = [
    "ClassDetectionMetrics",
    "DetectionAnnotationReport",
    "compare_detection_proposals",
]
