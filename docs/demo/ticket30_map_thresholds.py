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
    compute_wafer_quality_evidence,
    quality_relevant_threshold_candidates,
)


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
        evaluated = []
        for threshold in quality_relevant_threshold_candidates(validation_cases, class_index):
            thresholds = {candidate: (threshold if candidate == code else 1.0) for candidate in codes}
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
        "score_domain": score_domain,
        "source_split": "validation",
        "class_codes": list(codes),
        "candidate_policy": CANDIDATE_POLICY,
        "targets": dict(TARGETS),
        "selected": selected,
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
