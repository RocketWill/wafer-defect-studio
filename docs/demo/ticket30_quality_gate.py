"""Pure quality-gate evaluation for Ticket 30 Spatial MIL evidence."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real
from typing import Any


SCHEMA = "ticket30-quality-gate.v1"
MODEL = "spatial_mil_v4"
RECOMMENDED_FALLBACK = "cam_v2"
SEEDS = (17, 42, 91)
CLASS_CODES = ("scratch", "particle")
METRIC_KEYS = (
    "defect_instances",
    "defect_coverage_recall",
    "grid_precision",
    "grid_recall",
    "normal_grid_leak_rate",
    "asserted_grid_occupancy_p95",
)
CRITERIA = {
    "defect_instances": {"minimum": 150},
    "defect_coverage_recall": {"minimum": 1.0},
    "grid_precision": {"minimum": 0.95},
    "grid_recall": {"minimum": 0.95},
    "normal_grid_leak_rate": {"maximum": 0.05},
    "asserted_grid_occupancy_p95": {"maximum": 0.25},
}


def evaluate_ticket30_spatial_quality(
    manifest_sha256: str,
    metrics_by_seed: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[str, object]:
    """Evaluate every frozen seed/class independently against Ticket 30 targets."""

    _validate_manifest_sha256(manifest_sha256)
    if not isinstance(metrics_by_seed, Mapping):
        raise ValueError("metrics_by_seed must be a mapping")
    _require_membership(metrics_by_seed, SEEDS, "seed")

    per_seed: dict[int, dict[str, object]] = {}
    all_pass = True
    for seed in SEEDS:
        classes = metrics_by_seed[seed]
        if not isinstance(classes, Mapping):
            raise ValueError(f"seed={seed}: class metrics must be a mapping")
        _require_membership(classes, CLASS_CODES, f"seed={seed} class")
        per_class: dict[str, object] = {}
        for class_code in CLASS_CODES:
            metrics = _validate_metrics(seed, class_code, classes[class_code])
            criterion_results = _evaluate_criteria(metrics)
            class_pass = all(criterion_results.values())
            all_pass = all_pass and class_pass
            per_class[class_code] = {
                "metrics": metrics,
                "criteria": criterion_results,
                "overall": "PASS" if class_pass else "FAIL",
            }
        per_seed[seed] = per_class

    overall = "PASS" if all_pass else "FAIL"
    return {
        "schema": SCHEMA,
        "model": MODEL,
        "criteria": {name: dict(bounds) for name, bounds in CRITERIA.items()},
        "manifest_sha256": manifest_sha256,
        "per_seed": per_seed,
        "overall": overall,
        "recommendation": MODEL if overall == "PASS" else RECOMMENDED_FALLBACK,
    }


def _validate_manifest_sha256(value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("manifest SHA-256 must be 64 lowercase hexadecimal characters")


def _require_membership(mapping: Mapping[object, object], expected: tuple[object, ...], label: str) -> None:
    actual = tuple(mapping.keys())
    if set(actual) != set(expected) or len(actual) != len(expected) or any(
        type(value) is not type(expected[0]) for value in actual
    ):
        raise ValueError(f"{label} membership must be exactly {expected!r}; actual={actual!r}")


def _validate_metrics(seed: int, class_code: str, value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"seed={seed} class={class_code}: metrics must be a mapping")
    _require_membership(value, METRIC_KEYS, f"seed={seed} class={class_code} metric fields")

    instances = value["defect_instances"]
    if isinstance(instances, bool):
        raise ValueError(f"seed={seed} class={class_code}: defect_instances must not be boolean")
    if not isinstance(instances, Integral):
        raise ValueError(f"seed={seed} class={class_code}: defect_instances must be an integer")

    normalized: dict[str, object] = {"defect_instances": int(instances)}
    for name in METRIC_KEYS[1:]:
        metric = value[name]
        if metric is None:
            normalized[name] = None
            continue
        if isinstance(metric, bool):
            raise ValueError(f"seed={seed} class={class_code}: {name} must not be boolean")
        if not isinstance(metric, Real):
            raise ValueError(f"seed={seed} class={class_code}: {name} must be numeric or None")
        numeric = float(metric)
        if not isfinite(numeric):
            raise ValueError(f"seed={seed} class={class_code}: {name} must be finite")
        normalized[name] = numeric
    return normalized


def _evaluate_criteria(metrics: Mapping[str, object]) -> dict[str, bool]:
    instances = metrics["defect_instances"]
    coverage = metrics["defect_coverage_recall"]
    precision = metrics["grid_precision"]
    recall = metrics["grid_recall"]
    leakage = metrics["normal_grid_leak_rate"]
    occupancy = metrics["asserted_grid_occupancy_p95"]
    return {
        "defect_instances": instances >= 150,
        "defect_coverage_recall": coverage is not None and coverage == 1.0,
        "grid_precision": precision is not None and precision >= 0.95,
        "grid_recall": recall is not None and recall >= 0.95,
        "normal_grid_leak_rate": leakage is not None and leakage <= 0.05,
        "asserted_grid_occupancy_p95": occupancy is not None and occupancy <= 0.25,
    }


__all__ = ["evaluate_ticket30_spatial_quality"]
