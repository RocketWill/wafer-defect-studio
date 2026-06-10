"""Rendered truth and Defect Proposal evaluation for Ticket 37."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil, sqrt
from numbers import Integral

import numpy as np

from docs.demo.defect_oracle import DefectOracle, Particle, Scratch
from docs.demo.ticket30_evidence_corpus import render_ticket30_evidence_pixels
from docs.demo.ticket35_development_corpus import DevelopmentInstance
from docs.demo.ticket36_instance_matching import (
    _components,
    _maximum_matching,
    _validate_instances,
    _validate_response,
    _validate_threshold,
    _validate_tolerance,
)


SCHEMA = "ticket37-proposal-evaluation.v1"
_BACKGROUND_VALUE = 128
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


def build_rendered_truth_components(
    instances: Sequence[DevelopmentInstance],
    response_shape: tuple[int, int],
) -> dict[str, object]:
    """Render truth support and join touching support into components."""

    height, width = _validate_shape(response_shape)
    validated = _validate_instances(instances, width, height)
    for index, instance in enumerate(validated):
        _validate_radius(instance.defect.radius, f"instances[{index}].defect.radius")

    groups: dict[tuple[object, ...], dict[str, object]] = {}
    for instance in sorted(validated, key=lambda item: item.instance_id):
        descriptor = _descriptor_key(instance.defect)
        group = groups.setdefault(
            descriptor,
            {
                "instance_ids": [],
                "support": _render_support(instance.defect, width, height),
            },
        )
        group["instance_ids"].append(instance.instance_id)  # type: ignore[union-attr]

    truths: list[dict[str, object]] = []
    for truth_id, group in enumerate(groups.values()):
        cells = sorted(group["support"], key=lambda cell: (cell[1], cell[0]))  # type: ignore[index]
        truths.append(
            {
                "truth_id": truth_id,
                "instance_ids": sorted(group["instance_ids"]),  # type: ignore[index]
                "pixels": [[x, y] for x, y in cells],
                "area": len(cells),
            }
        )

    truth_components = _join_truth_components(truths, height, width)
    raw_count = len(validated)
    unique_count = len(truths)
    return {
        "schema": SCHEMA,
        "response_shape": [height, width],
        "raw_instance_count": raw_count,
        "unique_truth_count": unique_count,
        "duplicate_instance_count": raw_count - unique_count,
        "truth_component_count": len(truth_components),
        "unique_truths": truths,
        "truth_components": truth_components,
    }


def match_proposals_to_rendered_truth(
    response: np.ndarray,
    rendered_truth: Mapping[str, object],
    *,
    threshold: float = 0.5,
    tolerance_pixels: float = 8,
) -> dict[str, object]:
    """Match response proposals one-to-one against rendered truth components."""

    _validate_response(response)
    threshold_value = _validate_threshold(threshold)
    tolerance_value = _validate_tolerance(tolerance_pixels)
    truth = _validate_truth_report(rendered_truth, response.shape)
    truths = truth["unique_truths"]
    components = truth["truth_components"]

    active = response if response.dtype == np.bool_ else response >= threshold_value
    proposals = _components(response, active)
    touch_distances: list[dict[int, float]] = []
    for proposal in proposals:
        distances: dict[int, float] = {}
        for truth_id, item in enumerate(truths):
            distance = _distance_within_tolerance(
                proposal["_cells"], item["_cells"], tolerance_value
            )
            if distance is not None:
                distances[truth_id] = distance
        touch_distances.append(distances)
    touch_truth = [set(distances) for distances in touch_distances]
    truth_to_component = {
        truth_id: component_id
        for component_id, component in enumerate(components)
        for truth_id in component["truth_ids"]
    }
    touch_components = [
        {truth_to_component[truth_id] for truth_id in touched}
        for touched in touch_truth
    ]
    adjacency = [
        [
            proposal_id
            for proposal_id, touched in enumerate(touch_components)
            if component_id in touched
        ]
        for component_id in range(len(components))
    ]
    proposal_owner = _maximum_matching(adjacency, len(proposals))

    matched_pairs: list[dict[str, object]] = []
    matched_components: set[int] = set()
    matched_proposals: set[int] = set()
    for proposal_id, component_id in enumerate(proposal_owner):
        if component_id < 0:
            continue
        matched_components.add(component_id)
        matched_proposals.add(proposal_id)
        matched_pairs.append(
            {
                "truth_component_id": component_id,
                "proposal_id": proposal_id,
                "distance_pixels": min(
                    touch_distances[proposal_id][truth_id]
                    for truth_id in components[component_id]["truth_ids"]
                    if truth_id in touch_distances[proposal_id]
                ),
            }
        )
    matched_pairs.sort(key=lambda pair: pair["truth_component_id"])

    truth_count = len(components)
    proposal_count = len(proposals)
    matched_count = len(matched_pairs)
    touched_truth_count = len({truth_id for touched in touch_truth for truth_id in touched})
    return {
        "schema": SCHEMA,
        "truth_component_count": truth_count,
        "proposal_count": proposal_count,
        "matched_count": matched_count,
        "proposal_precision": 1.0 if proposal_count == 0 else matched_count / proposal_count,
        "proposal_recall": 1.0 if truth_count == 0 else matched_count / truth_count,
        "unique_truth_touch_recall": (
            1.0
            if not truths
            else touched_truth_count / len(truths)
        ),
        "unmatched_truth_component_ids": [
            component_id for component_id in range(truth_count) if component_id not in matched_components
        ],
        "unmatched_proposal_ids": [
            proposal_id for proposal_id in range(proposal_count) if proposal_id not in matched_proposals
        ],
        "cross_component_merge_proposal_ids": [
            proposal_id for proposal_id, touched in enumerate(touch_components) if len(touched) > 1
        ],
        "matched_pairs": matched_pairs,
        "proposal_components": [
            {
                "proposal_id": proposal["component_id"],
                "peak": list(proposal["peak"]),
                "area": proposal["area"],
                "truth_component_ids": sorted(touch_components[proposal_id]),
            }
            for proposal_id, proposal in enumerate(proposals)
        ],
    }


def _validate_shape(response_shape: tuple[int, int]) -> tuple[int, int]:
    try:
        shape = tuple(response_shape)
    except TypeError as error:
        raise ValueError("response_shape must be a two-item positive integer shape") from error
    if len(shape) != 2 or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0
        for value in shape
    ):
        raise ValueError("response_shape must be a two-item positive integer shape")
    return int(shape[0]), int(shape[1])


def _validate_radius(value: object, context: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")


def _descriptor_key(defect: Particle | Scratch) -> tuple[object, ...]:
    if isinstance(defect, Particle):
        return ("particle", defect.class_code, tuple(defect.center), defect.radius)
    return (
        "scratch",
        defect.class_code,
        tuple(tuple(point) for point in defect.points),
        defect.radius,
    )


def _render_support(defect: Particle | Scratch, width: int, height: int) -> frozenset[tuple[int, int]]:
    pixels = render_ticket30_evidence_pixels(DefectOracle(width, height, (defect,)))
    rows, columns = np.nonzero(pixels != _BACKGROUND_VALUE)
    return frozenset((int(column), int(row)) for row, column in zip(rows, columns))


def _join_truth_components(
    truths: Sequence[Mapping[str, object]], height: int, width: int
) -> list[dict[str, object]]:
    truth_cells: dict[tuple[int, int], set[int]] = {}
    for truth_id, truth in enumerate(truths):
        for x, y in truth["pixels"]:  # type: ignore[misc]
            truth_cells.setdefault((x, y), set()).add(truth_id)
    remaining = set(truth_cells)
    components: list[dict[str, object]] = []
    for row in range(height):
        for column in range(width):
            start = (column, row)
            if start not in remaining:
                continue
            remaining.remove(start)
            stack = [start]
            cells: list[tuple[int, int]] = []
            while stack:
                cell = stack.pop()
                cells.append(cell)
                x, y = cell
                for row_delta, column_delta in _NEIGHBORS_8:
                    neighbor = (x + column_delta, y + row_delta)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
            cells.sort(key=lambda cell: (cell[1], cell[0]))
            truth_ids = sorted({truth_id for cell in cells for truth_id in truth_cells[cell]})
            instance_ids = sorted(
                instance_id
                for truth_id in truth_ids
                for instance_id in truths[truth_id]["instance_ids"]  # type: ignore[misc]
            )
            components.append(
                {
                    "component_id": len(components),
                    "truth_ids": truth_ids,
                    "instance_ids": instance_ids,
                    "pixels": [[x, y] for x, y in cells],
                    "area": len(cells),
                }
            )
    return components


def _validate_truth_report(
    report: Mapping[str, object], response_shape: tuple[int, int]
) -> dict[str, list[dict[str, object]]]:
    if not isinstance(report, Mapping) or report.get("schema") != SCHEMA:
        raise ValueError("rendered_truth must be a Ticket 37 truth report")
    if tuple(report.get("response_shape", ())) != tuple(response_shape):
        raise ValueError("rendered truth response_shape must match response")
    truths = _validate_records(report.get("unique_truths"), "unique_truths")
    components = _validate_records(report.get("truth_components"), "truth_components")
    if report.get("unique_truth_count") != len(truths):
        raise ValueError("unique_truth_count does not match unique_truths")
    if report.get("truth_component_count") != len(components):
        raise ValueError("truth_component_count does not match truth_components")
    for index, truth in enumerate(truths):
        if truth.get("truth_id") != index:
            raise ValueError("unique truth IDs must be contiguous")
        _validate_pixels(truth, response_shape, f"unique_truths[{index}]")
    assigned_truth_ids: list[int] = []
    for index, component in enumerate(components):
        if component.get("component_id") != index:
            raise ValueError("truth component IDs must be contiguous")
        _validate_pixels(component, response_shape, f"truth_components[{index}]")
        ids = component.get("truth_ids")
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
            raise ValueError(f"truth_components[{index}] truth_ids must be sorted")
        ids = list(ids)
        if any(not isinstance(truth_id, Integral) for truth_id in ids) or ids != sorted(ids):
            raise ValueError(f"truth_components[{index}] truth_ids must be sorted")
        if any(not isinstance(truth_id, Integral) or truth_id < 0 or truth_id >= len(truths) for truth_id in ids):
            raise ValueError(f"truth_components[{index}] truth_ids out of range")
        if not ids:
            raise ValueError(f"truth_components[{index}] truth IDs must not be empty")
        assigned_truth_ids.extend(int(truth_id) for truth_id in ids)
    if sorted(assigned_truth_ids) != list(range(len(truths))):
        raise ValueError("truth IDs must be assigned exactly once to components")
    return {"unique_truths": truths, "truth_components": components}


def _validate_records(value: object, name: str) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{name} entries must be objects")
    return [dict(item) for item in value]


def _validate_pixels(item: Mapping[str, object], shape: tuple[int, int], context: str) -> None:
    pixels = item.get("pixels")
    if not isinstance(pixels, Sequence) or not pixels:
        raise ValueError(f"{context}.pixels must be non-empty")
    height, width = shape
    cells = []
    for pixel in pixels:
        if (
            not isinstance(pixel, Sequence)
            or len(pixel) != 2
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in pixel)
        ):
            raise ValueError(f"{context}.pixels must contain integer points")
        x, y = int(pixel[0]), int(pixel[1])
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"{context}.pixels lies outside response extent")
        cells.append((x, y))
    item["_cells"] = tuple(cells)  # type: ignore[index]


def _distance_within_tolerance(
    left: Sequence[tuple[int, int]],
    right: Sequence[tuple[int, int]],
    tolerance: float,
) -> float | None:
    if not left or not right:
        return None
    left_bbox = _bbox(left)
    right_bbox = _bbox(right)
    gap_x = max(left_bbox[0] - right_bbox[2], right_bbox[0] - left_bbox[2], 0)
    gap_y = max(left_bbox[1] - right_bbox[3], right_bbox[1] - left_bbox[3], 0)
    limit = tolerance * tolerance
    if gap_x * gap_x + gap_y * gap_y > limit:
        return None

    if len(left) <= len(right):
        source, target = left, set(right)
    else:
        source, target = right, set(left)
    for offset_x, offset_y, distance_squared in _neighbor_offsets(tolerance):
        for x, y in source:
            if (x + offset_x, y + offset_y) in target:
                return sqrt(distance_squared)
    return None


def _bbox(cells: Sequence[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [cell[0] for cell in cells]
    ys = [cell[1] for cell in cells]
    return min(xs), min(ys), max(xs), max(ys)


def _neighbor_offsets(tolerance: float) -> tuple[tuple[int, int, float], ...]:
    radius = ceil(tolerance)
    limit = tolerance * tolerance
    offsets = [
        (offset_x, offset_y, float(offset_x * offset_x + offset_y * offset_y))
        for offset_y in range(-radius, radius + 1)
        for offset_x in range(-radius, radius + 1)
        if offset_x * offset_x + offset_y * offset_y <= limit
    ]
    offsets.sort(key=lambda item: (item[2], item[1], item[0]))
    return tuple(offsets)


__all__ = [
    "build_rendered_truth_components",
    "match_proposals_to_rendered_truth",
]
