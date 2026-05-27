"""Validation-only thresholds for Ticket 30 model-specific map domains."""

from __future__ import annotations

import json
from math import inf, isfinite
from numbers import Real
from typing import Mapping, Sequence

import numpy

from docs.demo.spatial_thresholds import CANDIDATE_POLICY, TARGETS
from docs.demo.wafer_quality_evidence import (
    WaferEvidenceCase,
    _defect_pixel_bounds,
    quality_relevant_threshold_candidates,
)
from docs.demo.defect_oracle import defect_intersects_rectangle


SCHEMA = "ticket30-map-thresholds.v1"
SCORE_DOMAINS = ("normalized_window_cam", "patch_sigmoid", "absolute_spatial_probability")


def calibrate_map_thresholds(
    validation_cases: Sequence[WaferEvidenceCase],
    class_codes: Sequence[str],
    score_domain: str,
    *,
    source_split: str = "validation",
) -> dict[str, object]:
    _require_domain(score_domain)
    if source_split != "validation":
        raise ValueError(f"map thresholds require validation source split, got {source_split!r}")
    for case in validation_cases:
        if case.split != "validation":
            raise ValueError(f"{case.filename}: map threshold case split must be validation, got {case.split!r}")

    codes = tuple(class_codes)
    selected = {}
    for class_index, code in enumerate(codes):
        finite_ranges = [
            values[numpy.isfinite(values)]
            for case in validation_cases
            for values in (numpy.asarray(case.absolute_maps)[:, :, class_index],)
        ]
        if any(values.size and (values.min() < 0 or values.max() > 1) for values in finite_ranges):
            raise ValueError(f"{code}: map threshold candidates must be in [0, 1]")
        metric_index = _build_metric_index(validation_cases, class_index, code)
        evaluated = []
        for threshold in quality_relevant_threshold_candidates(validation_cases, class_index):
            metrics = _indexed_metrics(metric_index, threshold)
            evaluated.append((threshold, metrics, _target_satisfied(metrics)))
        feasible = [entry for entry in evaluated if entry[2]]
        winner = max(feasible, key=lambda entry: entry[0]) if feasible else max(
            evaluated, key=lambda entry: _fallback_key(entry[1], entry[0])
        )
        selected[code] = {
            "threshold": winner[0],
            "metrics": winner[1],
            "target_satisfied": winner[2],
        }

    return {
        "schema": SCHEMA,
        "score_domain": score_domain,
        "source_split": "validation",
        "class_codes": list(codes),
        "candidate_policy": CANDIDATE_POLICY,
        "targets": dict(TARGETS),
        "selected": selected,
    }


def _build_metric_index(cases, class_index: int, code: str):
    defect_maxima = []
    grids = []
    for case in cases:
        class_map = numpy.asarray(case.absolute_maps)[:, :, class_index]
        truth = case.oracle.grid_truth(case.grids)
        for defect in case.oracle.defects:
            if defect.class_code != code:
                continue
            left, top, right, bottom = _defect_pixel_bounds(
                defect, case.oracle.image_width, case.oracle.image_height
            )
            values = (
                float(class_map[y, x])
                for y in range(top, bottom)
                for x in range(left, right)
                if defect_intersects_rectangle(defect, x, y, x + 1, y + 1)
                and numpy.isfinite(class_map[y, x])
            )
            defect_maxima.append(max(values, default=-inf))
        for grid in case.grids:
            left, top = max(0, grid.x), max(0, grid.y)
            right = min(case.oracle.image_width, grid.x + grid.width)
            bottom = min(case.oracle.image_height, grid.y + grid.height)
            area = (right - left) * (bottom - top)
            if not area:
                continue
            finite = class_map[top:bottom, left:right]
            finite = finite[numpy.isfinite(finite)]
            grids.append((
                code in truth[(grid.row, grid.column)],
                float(finite.max()) if finite.size else -inf,
                numpy.sort(finite) if code in truth[(grid.row, grid.column)] else None,
                area,
            ))
    return defect_maxima, grids


def _indexed_metrics(index, threshold: float) -> dict[str, object]:
    defect_maxima, grids = index
    covered = sum(value >= threshold for value in defect_maxima)
    tp = fp = fn = normal = 0
    occupancies = []
    for actual, maximum, sorted_values, area in grids:
        predicted = maximum >= threshold
        if actual and predicted:
            tp += 1
        elif actual:
            fn += 1
        elif predicted:
            fp += 1
        if not actual:
            normal += 1
        else:
            asserted = len(sorted_values) - int(
                numpy.searchsorted(sorted_values, threshold, side="left")
            )
            occupancies.append(asserted / area)
    occupancies.sort()
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    return {
        "defect_instances": len(defect_maxima),
        "covered_defect_instances": covered,
        "defect_coverage_recall": covered / len(defect_maxima) if defect_maxima else None,
        "grid_tp": tp,
        "grid_fp": fp,
        "grid_fn": fn,
        "grid_precision": tp / precision_denominator if precision_denominator else None,
        "grid_recall": tp / recall_denominator if recall_denominator else None,
        "normal_grid_leaks": fp,
        "normal_grids": normal,
        "normal_grid_leak_rate": fp / normal if normal else None,
        "asserted_grid_occupancy_p95": (
            occupancies[(95 * len(occupancies) + 99) // 100 - 1]
            if occupancies else None
        ),
    }


def resolve_map_thresholds(
    payload: Mapping[str, object],
    class_codes: Sequence[str],
    score_domain: str,
    validation_cases: Sequence[WaferEvidenceCase],
) -> dict[str, float]:
    _require_domain(score_domain)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"map threshold schema must be {SCHEMA}")
    if payload.get("score_domain") != score_domain:
        raise ValueError(f"map threshold score domain must be {score_domain}")
    codes = tuple(class_codes)
    expected = calibrate_map_thresholds(validation_cases, codes, score_domain)
    if payload != expected:
        raise ValueError("map threshold artifact does not match validation calibration")
    resolved = {}
    for code in codes:
        threshold = payload["selected"][code]["threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, Real) or not isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"{code}: map threshold must be a finite number in [0, 1]")
        resolved[code] = float(threshold)
    return resolved


def canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require_domain(score_domain: str) -> None:
    if score_domain not in SCORE_DOMAINS:
        raise ValueError(f"unsupported map threshold score domain: {score_domain!r}")


def _target_satisfied(metrics: Mapping[str, float | int | None]) -> bool:
    return (
        metrics["defect_coverage_recall"] is not None
        and metrics["defect_coverage_recall"] >= TARGETS["defect_coverage_recall"]
        and metrics["grid_precision"] is not None
        and metrics["grid_precision"] >= TARGETS["grid_precision"]
        and metrics["grid_recall"] is not None
        and metrics["grid_recall"] >= TARGETS["grid_recall"]
        and metrics["normal_grid_leak_rate"] is not None
        and metrics["normal_grid_leak_rate"] <= TARGETS["normal_grid_leak_rate"]
        and metrics["asserted_grid_occupancy_p95"] is not None
        and metrics["asserted_grid_occupancy_p95"] <= TARGETS["asserted_grid_occupancy_p95"]
    )


def _fallback_key(metrics: Mapping[str, float | int | None], threshold: float) -> tuple[float, ...]:
    value = lambda name: -inf if metrics[name] is None else float(metrics[name])
    precision = value("grid_precision")
    recall = value("grid_recall")
    leakage = value("normal_grid_leak_rate")
    occupancy = value("asserted_grid_occupancy_p95")
    return (
        value("defect_coverage_recall"),
        min(precision, recall),
        precision,
        recall,
        -leakage if leakage != -inf else -inf,
        -occupancy if occupancy != -inf else -inf,
        threshold,
    )
