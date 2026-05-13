"""Validation-only calibration for absolute spatial probability maps."""

from __future__ import annotations

import json
from math import inf, isfinite
from numbers import Real
from typing import Mapping, Sequence

import numpy

from docs.demo.wafer_quality_evidence import WaferEvidenceCase, compute_wafer_quality_evidence


SCHEMA = "spatial-map-thresholds.v1"
SCORE_DOMAIN = "absolute_spatial_probability"
CANDIDATE_POLICY = "sorted_unique_finite_map_values_plus_0_and_1; map>=threshold"
TARGETS = {
    "defect_coverage_recall": 1.0,
    "grid_precision": 0.95,
    "grid_recall": 0.95,
    "normal_grid_leak_rate": 0.05,
    "asserted_grid_occupancy_p95": 0.25,
}


def calibrate_spatial_thresholds(
    validation_cases: Sequence[WaferEvidenceCase],
    class_codes: Sequence[str],
    *,
    source_split: str = "validation",
) -> dict[str, object]:
    if source_split != "validation":
        raise ValueError(f"spatial thresholds require validation source split, got {source_split!r}")
    for case in validation_cases:
        if case.split != "validation":
            raise ValueError(f"{case.filename}: spatial threshold case split must be validation, got {case.split!r}")

    codes = tuple(class_codes)
    selected = {}
    for class_index, code in enumerate(codes):
        candidates = sorted(
            {0.0, 1.0}.union(
                float(value)
                for case in validation_cases
                for value in numpy.asarray(case.absolute_maps)[:, :, class_index].flat
                if numpy.isfinite(value)
            )
        )
        evaluated = []
        for threshold in candidates:
            thresholds = {candidate_code: (threshold if candidate_code == code else 1.0) for candidate_code in codes}
            metrics = compute_wafer_quality_evidence(validation_cases, codes, thresholds)["per_class"][code]
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
        "score_domain": SCORE_DOMAIN,
        "source_split": "validation",
        "class_codes": list(codes),
        "candidate_policy": CANDIDATE_POLICY,
        "targets": dict(TARGETS),
        "selected": selected,
    }


def resolve_spatial_thresholds(
    payload: Mapping[str, object],
    class_codes: Sequence[str],
    score_domain: str = SCORE_DOMAIN,
) -> dict[str, float]:
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"spatial threshold schema must be {SCHEMA}")
    if score_domain != SCORE_DOMAIN or payload.get("score_domain") != SCORE_DOMAIN:
        raise ValueError(f"spatial threshold score domain must be {SCORE_DOMAIN}")
    if payload.get("source_split") != "validation":
        raise ValueError("spatial threshold source split must be validation")
    if payload.get("candidate_policy") != CANDIDATE_POLICY:
        raise ValueError("spatial threshold candidate policy does not match calibration contract")
    targets = payload.get("targets")
    if (
        not isinstance(targets, Mapping)
        or set(targets) != set(TARGETS)
        or any(type(targets[name]) is not float or targets[name] != expected for name, expected in TARGETS.items())
    ):
        raise ValueError("spatial threshold targets do not match calibration contract")
    codes = list(class_codes)
    if payload.get("class_codes") != codes:
        raise ValueError("spatial threshold class order does not match requested classes")
    selected = payload.get("selected")
    if not isinstance(selected, Mapping) or set(selected) != set(codes):
        raise ValueError("spatial threshold selected classes must be complete")
    resolved = {}
    for code in codes:
        entry = selected[code]
        if not isinstance(entry, Mapping):
            raise ValueError(f"{code}: spatial threshold selected entry must be a mapping")
        threshold = entry.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, Real) or not isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"{code}: spatial threshold must be a finite number in [0, 1]")
        metrics = entry.get("metrics")
        if not isinstance(metrics, Mapping) or not set(TARGETS).issubset(metrics):
            raise ValueError(f"{code}: spatial threshold metrics are incomplete")
        if type(entry.get("target_satisfied")) is not bool:
            raise ValueError(f"{code}: spatial threshold target_satisfied must be boolean")
        resolved[code] = float(threshold)
    return resolved


def spatial_thresholds_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
