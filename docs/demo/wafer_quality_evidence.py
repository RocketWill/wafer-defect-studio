"""Generated-wafer spatial quality evidence in source coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Mapping, Sequence

import numpy

from docs.demo.defect_oracle import DefectOracle, Particle, Scratch, defect_intersects_rectangle
from wafer_defect_studio.grid_geometry import AnnotationGrid


@dataclass(frozen=True)
class WaferEvidenceCase:
    filename: str
    split: str
    oracle: DefectOracle
    grids: tuple[AnnotationGrid, ...]
    absolute_maps: numpy.ndarray


def compute_wafer_quality_evidence(
    cases: Sequence[WaferEvidenceCase],
    class_codes: Sequence[str],
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    totals = {
        code: {"instances": 0, "covered": 0, "tp": 0, "fp": 0, "fn": 0, "normal": 0, "occupancies": []}
        for code in class_codes
    }
    for case in cases:
        maps = numpy.asarray(case.absolute_maps)
        expected = (case.oracle.image_height, case.oracle.image_width, len(class_codes))
        if maps.shape != expected:
            if maps.ndim >= 1 and maps.shape[-1] != len(class_codes):
                raise ValueError(
                    f"{case.filename}: class map count expected {len(class_codes)}, actual {maps.shape[-1]}"
                )
            raise ValueError(f"{case.filename}: map shape expected {expected}, actual {maps.shape}")
        truth = case.oracle.grid_truth(case.grids)
        for class_index, code in enumerate(class_codes):
            class_map = maps[:, :, class_index]
            retained = numpy.isfinite(class_map) & (class_map >= thresholds[code])
            class_totals = totals[code]
            for defect in case.oracle.defects:
                if defect.class_code != code:
                    continue
                class_totals["instances"] += 1
                left, top, right, bottom = _defect_pixel_bounds(
                    defect, case.oracle.image_width, case.oracle.image_height
                )
                local = retained[top:bottom, left:right]
                ys, xs = numpy.nonzero(local)
                if any(
                    defect_intersects_rectangle(
                        defect, int(x + left), int(y + top), int(x + left + 1), int(y + top + 1)
                    )
                    for y, x in zip(ys, xs)
                ):
                    class_totals["covered"] += 1
            for grid in case.grids:
                left, top = max(0, grid.x), max(0, grid.y)
                right = min(case.oracle.image_width, grid.x + grid.width)
                bottom = min(case.oracle.image_height, grid.y + grid.height)
                clipped_area = (right - left) * (bottom - top)
                if clipped_area == 0:
                    continue
                selected = retained[top:bottom, left:right]
                asserted_pixels = int(selected.sum())
                predicted = asserted_pixels > 0
                actual = code in truth[(grid.row, grid.column)]
                if actual and predicted:
                    class_totals["tp"] += 1
                elif actual:
                    class_totals["fn"] += 1
                elif predicted:
                    class_totals["fp"] += 1
                if not actual:
                    class_totals["normal"] += 1
                if actual:
                    class_totals["occupancies"].append(asserted_pixels / clipped_area)

    per_class = {}
    for code, values in totals.items():
        precision_denominator = values["tp"] + values["fp"]
        recall_denominator = values["tp"] + values["fn"]
        occupancies = sorted(values["occupancies"])
        per_class[code] = {
            "defect_instances": values["instances"],
            "covered_defect_instances": values["covered"],
            "defect_coverage_recall": _ratio(values["covered"], values["instances"]),
            "grid_tp": values["tp"],
            "grid_fp": values["fp"],
            "grid_fn": values["fn"],
            "grid_precision": _ratio(values["tp"], precision_denominator),
            "grid_recall": _ratio(values["tp"], recall_denominator),
            "normal_grid_leaks": values["fp"],
            "normal_grids": values["normal"],
            "normal_grid_leak_rate": _ratio(values["fp"], values["normal"]),
            "asserted_grid_occupancy_p95": (
                occupancies[ceil(0.95 * len(occupancies)) - 1] if occupancies else None
            ),
        }
    return {"case_count": len(cases), "per_class": per_class}


def quality_relevant_threshold_candidates(
    cases: Sequence[WaferEvidenceCase], class_index: int
) -> tuple[float, ...]:
    """Return inclusive thresholds where Grid prediction or defect coverage changes."""

    candidates = {0.0, 1.0}
    for case in cases:
        class_map = numpy.asarray(case.absolute_maps)[:, :, class_index]
        for grid in case.grids:
            left, top = max(0, grid.x), max(0, grid.y)
            right = min(case.oracle.image_width, grid.x + grid.width)
            bottom = min(case.oracle.image_height, grid.y + grid.height)
            _add_finite_max(candidates, class_map[top:bottom, left:right])
        for defect in case.oracle.defects:
            left, top, right, bottom = _defect_pixel_bounds(
                defect, case.oracle.image_width, case.oracle.image_height
            )
            support_values = (
                class_map[y, x]
                for y in range(top, bottom)
                for x in range(left, right)
                if defect_intersects_rectangle(defect, x, y, x + 1, y + 1)
            )
            finite_values = (float(value) for value in support_values if numpy.isfinite(value))
            maximum = max(finite_values, default=None)
            if maximum is not None:
                candidates.add(maximum)
    return tuple(sorted(candidates))


def _add_finite_max(candidates: set[float], values: numpy.ndarray) -> None:
    finite = values[numpy.isfinite(values)]
    if finite.size:
        candidates.add(float(finite.max()))


def _defect_pixel_bounds(
    defect: Scratch | Particle, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    if isinstance(defect, Particle):
        xs, ys = (defect.center[0],), (defect.center[1],)
    else:
        xs, ys = zip(*defect.points)
    return (
        max(0, min(xs) - defect.radius - 1),
        max(0, min(ys) - defect.radius - 1),
        min(image_width, max(xs) + defect.radius + 1),
        min(image_height, max(ys) + defect.radius + 1),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
