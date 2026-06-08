"""Deterministic instance-to-peak matching for Ticket 36.

This metric evaluates approximate localization proposals.  It does not create
or validate a segmentation mask: one 8-connected response component produces
exactly one peak, even when that component contains several local highs.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import hypot, isfinite
from numbers import Real

import numpy as np

from docs.demo.defect_oracle import Particle, Scratch, _point_segment_distance
from docs.demo.ticket35_development_corpus import DevelopmentInstance


_NEIGHBORS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def match_ticket36_instance_peaks(
    response: np.ndarray,
    instances: Sequence[DevelopmentInstance],
    *,
    threshold: float = 0.5,
    tolerance_pixels: float = 8,
) -> dict[str, object]:
    """Match thresholded response components to sparse instance truth.

    ``response`` is either a 2-D boolean response mask or a 2-D floating
    probability map.  Floating maps are thresholded with the fixed per-call
    ``threshold`` (the Ticket 36 contract uses ``0.5``), while the component's
    peak retains the highest floating response.  Coordinates are expressed as
    ``(x, y)`` pixels: numpy ``(row, column)`` cells are converted before
    distance calculations.

    Empty truth is a vacuous recall of ``1.0``; non-empty truth with no
    response has recall ``0.0``.  A maximum-cardinality one-to-one matching is
    used, with input truth order and row-major component order as tie-breakers.
    """

    _validate_response(response)
    threshold_value = _validate_threshold(threshold)
    tolerance_value = _validate_tolerance(tolerance_pixels)
    truth = _validate_instances(instances, response.shape[1], response.shape[0])
    active = response if response.dtype == np.bool_ else response >= threshold_value
    components = _components(response, active)

    adjacency: list[list[int]] = []
    for item in truth:
        adjacency.append(
            [
                component["component_id"]
                for component in components
                if _distance_to_defect(component["peak"], item.defect) <= tolerance_value
            ]
        )

    matched_component_to_instance = _maximum_matching(adjacency, len(components))
    matched_pairs = []
    matched_instance_indexes: set[int] = set()
    for component_id, instance_index in enumerate(matched_component_to_instance):
        if instance_index < 0:
            continue
        matched_instance_indexes.add(instance_index)
        matched_pairs.append(
            {
                "instance_id": truth[instance_index].instance_id,
                "component_id": component_id,
                "distance_pixels": float(
                    _distance_to_defect(components[component_id]["peak"], truth[instance_index].defect)
                ),
            }
        )
    matched_pairs.sort(key=lambda item: next(
        index for index, instance in enumerate(truth) if instance.instance_id == item["instance_id"]
    ))

    merged_component_ids = []
    for component in components:
        reachable_truth_count = sum(
            any(
                _distance_to_defect(cell, item.defect) <= tolerance_value
                for cell in component["_cells"]
            )
            for item in truth
        )
        if reachable_truth_count > 1:
            merged_component_ids.append(component["component_id"])
    matched = len(matched_pairs)
    instance_count = len(truth)
    return {
        "instance_count": instance_count,
        "component_count": len(components),
        "matched": matched,
        "unmatched_instance_ids": [
            item.instance_id
            for index, item in enumerate(truth)
            if index not in matched_instance_indexes
        ],
        "unmatched_component_ids": [
            component_id
            for component_id, instance_index in enumerate(matched_component_to_instance)
            if instance_index < 0
        ],
        "recall": 1.0 if instance_count == 0 else matched / instance_count,
        "merged_component_ids": merged_component_ids,
        "matched_pairs": matched_pairs,
        "component_peaks": [
            {
                "component_id": component["component_id"],
                "peak": list(component["peak"]),
                "area": component["area"],
            }
            for component in components
        ],
    }


def _validate_response(response: object) -> None:
    if not isinstance(response, np.ndarray):
        raise ValueError("response must be a numpy ndarray")
    if response.ndim != 2:
        raise ValueError(f"response must be 2-D; actual ndim={response.ndim}")
    if response.shape[0] == 0 or response.shape[1] == 0:
        raise ValueError(f"response must have non-empty 2-D shape; actual={response.shape!r}")
    if response.dtype != np.bool_ and not np.issubdtype(response.dtype, np.floating):
        raise ValueError("response dtype must be boolean or floating")
    if np.issubdtype(response.dtype, np.floating) and not bool(np.isfinite(response).all()):
        raise ValueError("response floating values must be finite")


def _validate_threshold(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("threshold must be a finite real number in [0, 1]")
    converted = float(value)
    if not isfinite(converted) or not 0 <= converted <= 1:
        raise ValueError("threshold must be a finite real number in [0, 1]")
    return converted


def _validate_tolerance(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("tolerance_pixels must be a finite non-negative real number")
    converted = float(value)
    if not isfinite(converted) or converted < 0:
        raise ValueError("tolerance_pixels must be a finite non-negative real number")
    return converted


def _validate_instances(
    instances: Sequence[DevelopmentInstance],
    width: int,
    height: int,
) -> tuple[DevelopmentInstance, ...]:
    try:
        resolved = tuple(instances)
    except TypeError as error:
        raise ValueError("instances must be a sequence of DevelopmentInstance values") from error
    seen_ids: set[str] = set()
    class_code: str | None = None
    for index, instance in enumerate(resolved):
        if not isinstance(instance, DevelopmentInstance):
            raise ValueError(f"instances[{index}] must be a DevelopmentInstance")
        if not isinstance(instance.instance_id, str) or not instance.instance_id.strip():
            raise ValueError(f"instances[{index}] instance_id must be non-empty text")
        if instance.instance_id in seen_ids:
            raise ValueError(f"duplicate instance_id: {instance.instance_id!r}")
        seen_ids.add(instance.instance_id)
        defect = instance.defect
        if not isinstance(defect, (Particle, Scratch)):
            raise ValueError(f"instances[{index}] defect must be Particle or Scratch")
        expected_class = "particle" if isinstance(defect, Particle) else "scratch"
        if defect.class_code != expected_class:
            raise ValueError(
                f"instances[{index}] defect class/kind mismatch: expected {expected_class!r}"
            )
        if class_code is None:
            class_code = defect.class_code
        elif class_code != defect.class_code:
            raise ValueError("all instances must share the same class")
        if isinstance(defect, Particle):
            _validate_point(
                defect.center,
                f"instances[{index}].defect.center",
                width,
                height,
            )
        else:
            if not isinstance(defect.points, Sequence) or len(defect.points) < 2:
                raise ValueError(f"instances[{index}].defect.points must contain at least two points")
            for point_index, point in enumerate(defect.points):
                _validate_point(
                    point,
                    f"instances[{index}].defect.points[{point_index}]",
                    width,
                    height,
                )
    return resolved


def _validate_point(value: object, context: str, width: int, height: int) -> None:
    if (
        not isinstance(value, Sequence)
        or len(value) != 2
        or any(
            isinstance(coordinate, (bool, np.bool_))
            or not isinstance(coordinate, Real)
            or not isfinite(float(coordinate))
            for coordinate in value
        )
    ):
        raise ValueError(f"{context} must be a finite numeric (x, y) point")
    x, y = float(value[0]), float(value[1])
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(
            f"{context} must lie within response extent width={width} height={height}; "
            f"actual={tuple(value)!r}"
        )


def _components(response: np.ndarray, active: np.ndarray) -> list[dict[str, object]]:
    height, width = active.shape
    visited = np.zeros(active.shape, dtype=bool)
    components: list[dict[str, object]] = []
    for row in range(height):
        for column in range(width):
            if not active[row, column] or visited[row, column]:
                continue
            stack = [(row, column)]
            visited[row, column] = True
            cells: list[tuple[int, int]] = []
            while stack:
                current_row, current_column = stack.pop()
                cells.append((current_row, current_column))
                for row_delta, column_delta in _NEIGHBORS_8:
                    next_row = current_row + row_delta
                    next_column = current_column + column_delta
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and active[next_row, next_column]
                        and not visited[next_row, next_column]
                    ):
                        visited[next_row, next_column] = True
                        stack.append((next_row, next_column))
            cells.sort()
            if response.dtype == np.bool_:
                peak_row, peak_column = cells[0]
            else:
                peak_row, peak_column = min(
                    cells,
                    key=lambda cell: (-float(response[cell]), cell[0], cell[1]),
                )
            components.append(
                {
                    "component_id": len(components),
                    "peak": (float(peak_column), float(peak_row)),
                    "area": len(cells),
                    "_cells": tuple((column, row) for row, column in cells),
                }
            )
    return components


def _distance_to_defect(peak: tuple[float, float], defect: Particle | Scratch) -> float:
    if isinstance(defect, Particle):
        return hypot(peak[0] - defect.center[0], peak[1] - defect.center[1])
    return min(
        _point_segment_distance(peak, start, end)
        for start, end in zip(defect.points, defect.points[1:])
    )


def _maximum_matching(adjacency: Sequence[Sequence[int]], component_count: int) -> list[int]:
    """Return component-owner indices after deterministic Kuhn augmentation."""

    component_owner = [-1] * component_count

    def augment(instance_index: int, visited: set[int]) -> bool:
        for component_id in adjacency[instance_index]:
            if component_id in visited:
                continue
            visited.add(component_id)
            owner = component_owner[component_id]
            if owner < 0 or augment(owner, visited):
                component_owner[component_id] = instance_index
                return True
        return False

    for instance_index in range(len(adjacency)):
        augment(instance_index, set())
    return component_owner


__all__ = ["match_ticket36_instance_peaks"]
