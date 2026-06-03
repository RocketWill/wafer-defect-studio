"""Frozen density/extent diagnosis for Ticket 34 final evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path


SCHEMA = "ticket35-density-diagnosis.v1"
TICKET34_REPORT_SHA256 = (
    "bad029716b6798c0c88e641f9ac011f210c1b663f7d2f601604b177ac899f562"
)
TICKET31_CORPUS_SOURCE_SHA256 = (
    "c087e1be7b236862d54fec1a4f7cb1e2d0d2b4030e041e5c9c9371d7a261e070"
)
FINAL_MEMBERS = (
    "ticket30-evidence-17.png",
    "ticket30-evidence-42.png",
    "ticket30-evidence-91.png",
)
FINAL_SEEDS = (17, 42, 91)
CLASS_CODES = ("scratch", "particle")
DEVELOPMENT_METADATA = {
    "development_seeds": [101, 211, 307, 401, 503],
    "development_splits": ["train", "validation"],
    "development_cases_per_seed_split": 24,
    "development_max_instances_per_image_per_class": 2,
    "development_max_instances_per_image": 4,
    "development_case_count": 240,
}
FINAL_INSTANCES_PER_SEED_CLASS = 150
OCCUPANCY_FAILURE_BOUNDARY = 0.25


def build_ticket35_density_diagnosis(
    report_text: str,
    development_metadata: Mapping[str, object],
    *,
    ticket34_report_path: str = "docs/demo/ticket34-final-gate.json",
    ticket31_corpus_source_path: str = "docs/demo/ticket31_development_corpus.py",
) -> dict[str, object]:
    """Build the canonical diagnosis from frozen report and corpus metadata.

    The final report is consumed as already-published evidence.  This seam
    computes no maps, thresholds, model scores, or new metrics.
    """

    report = validate_ticket34_final_report(report_text)
    metadata = _validate_development_metadata(development_metadata)
    per_seed: dict[str, object] = {}
    all_margins_positive = True

    for seed, filename in zip(FINAL_SEEDS, FINAL_MEMBERS, strict=True):
        report_seed = report["per_seed"][str(seed)]
        classes = report_seed["per_class"]
        per_class: dict[str, object] = {}
        for class_code in CLASS_CODES:
            row = classes[class_code]
            margin = row["score_separation_margin"]
            all_margins_positive = all_margins_positive and margin > 0.0
            per_class[class_code] = {
                **{key: row[key] for key in _EVIDENCE_FIELDS},
                "diagnostic": _diagnostic_fields(row, class_code),
            }
        per_seed[str(seed)] = {
            "filename": filename,
            "seed": seed,
            "per_class": per_class,
        }

    ratio = FINAL_INSTANCES_PER_SEED_CLASS // metadata[
        "development_max_instances_per_image_per_class"
    ]
    return {
        "schema": SCHEMA,
        "ticket34_report_sha256": TICKET34_REPORT_SHA256,
        "ticket31_corpus_source_sha256": TICKET31_CORPUS_SOURCE_SHA256,
        "sources": {
            "ticket34_final_report": ticket34_report_path,
            "ticket31_development_corpus": ticket31_corpus_source_path,
        },
        "decision": {
            "overall": report["overall"],
            "recommendation": report["recommendation"],
            "cam_default": "cam_v2",
            "final_calibration": "forbidden",
            "rerun_or_tuning": "forbidden",
        },
        "final_boundary": {
            "members": list(FINAL_MEMBERS),
            "status": "diagnostic_only_consumed_final",
            "promotion_reuse": "forbidden",
        },
        "development_metadata": metadata,
        "density_shift": {
            "development_max_instances_per_image_per_class": metadata[
                "development_max_instances_per_image_per_class"
            ],
            "final_instances_per_seed_class": FINAL_INSTANCES_PER_SEED_CLASS,
            "ratio_final_to_development": ratio,
            "comparison": "final held-out density is 75x the development per-image per-class maximum",
        },
        "separation": {
            "all_margins_positive": all_margins_positive,
            "separation_failure": not all_margins_positive,
            "interpretation": "positive separation margins are not the observed failure",
        },
        "hypothesis": {
            "density_composition_shift": {
                "status": "primary_hypothesis",
                "relationship": "correlation",
                "validation_slices": ["35.02", "35.03", "35.04", "35.05", "35.06"],
                "causal_proof": False,
            }
        },
        "per_seed": per_seed,
    }


def validate_ticket34_final_report(report_text: str) -> dict[str, object]:
    """Validate the exact published Ticket 34 report before reading metrics."""

    actual_sha256 = _sha256_text(report_text)
    if actual_sha256 != TICKET34_REPORT_SHA256:
        raise ValueError(
            "Ticket 34 final report SHA-256 mismatch: "
            f"expected={TICKET34_REPORT_SHA256} actual={actual_sha256}"
        )
    try:
        report = json.loads(report_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Ticket 34 final report: {error}") from error
    if not isinstance(report, Mapping):
        raise ValueError("Ticket 34 final report must be a JSON object")
    if report.get("schema") != "ticket34-final-gate.v1":
        raise ValueError("Ticket 34 final report schema drift")
    if report.get("overall") != "FAIL" or report.get("recommendation") != "cam_v2":
        raise ValueError("Ticket 34 final decision drift")
    if report.get("final_members") != list(FINAL_MEMBERS):
        raise ValueError("Ticket 34 final member order drift")
    if report.get("class_codes") != ["scratch", "particle"]:
        raise ValueError("Ticket 34 final class order drift")

    seeds = report.get("per_seed")
    if not isinstance(seeds, Mapping) or tuple(seeds) != tuple(str(seed) for seed in FINAL_SEEDS):
        raise ValueError("Ticket 34 final seed membership/order drift")
    for seed in FINAL_SEEDS:
        result = seeds[str(seed)]
        if not isinstance(result, Mapping) or result.get("filename") != f"ticket30-evidence-{seed}.png":
            raise ValueError(f"Ticket 34 final filename drift: seed={seed}")
        classes = result.get("per_class")
        if not isinstance(classes, Mapping) or tuple(classes) != ("particle", "scratch"):
            raise ValueError(f"Ticket 34 final class membership/order drift: seed={seed}")
        for class_code in ("particle", "scratch"):
            row = classes[class_code]
            if not isinstance(row, Mapping) or row.get("overall") != "FAIL":
                raise ValueError(f"Ticket 34 final row decision drift: seed={seed} class={class_code}")
            _validate_evidence_row(row, seed, class_code)
    return dict(report)


def read_ticket31_development_metadata(source_path: Path) -> dict[str, object]:
    """Read corpus metadata without importing or rendering the corpus."""

    try:
        source_text = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read Ticket 31 corpus source: {source_path}: {error}") from error
    actual_sha256 = _sha256_text(source_text)
    if actual_sha256 != TICKET31_CORPUS_SOURCE_SHA256:
        raise ValueError(
            "Ticket 31 corpus source SHA-256 mismatch: "
            f"expected={TICKET31_CORPUS_SOURCE_SHA256} actual={actual_sha256}"
        )
    required_markers = (
        'for split_index, split in enumerate(("train", "validation")):',
        "for index in range(24):",
        "scratch_count = (",
        "particle_count = (",
        '1 + variant % 2 if composition in {"scratch", "both"}',
        '1 + variant % 2 if composition in {"particle", "both"}',
    )
    if any(marker not in source_text for marker in required_markers):
        raise ValueError("Ticket 31 development corpus metadata markers drift")
    return {
        "source_sha256": TICKET31_CORPUS_SOURCE_SHA256,
        **DEVELOPMENT_METADATA,
    }


def canonical_density_diagnosis_json(payload: Mapping[str, object]) -> str:
    """Serialize a diagnosis deterministically for tracking and comparison."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--corpus-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report_text = _read_text(args.report, "Ticket 34 final report")
        metadata = read_ticket31_development_metadata(args.corpus_source)
        diagnosis = build_ticket35_density_diagnosis(
            report_text,
            metadata,
            ticket34_report_path=args.report.as_posix(),
            ticket31_corpus_source_path=args.corpus_source.as_posix(),
        )
        output = args.output.resolve()
        if output.exists():
            raise ValueError(f"Ticket 35 diagnosis output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_density_diagnosis_json(diagnosis) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"Ticket 35 density diagnosis failed: {error}", file=sys.stderr)
        return 1
    print(f"Ticket 35 density diagnosis: {output}", flush=True)
    return 0


_EVIDENCE_FIELDS = (
    "asserted_grid_occupancy_p95",
    "covered_defect_instances",
    "defect_coverage_recall",
    "defect_instances",
    "grid_fn",
    "grid_fp",
    "grid_precision",
    "grid_recall",
    "grid_tp",
    "normal_grid_leak_rate",
    "normal_grid_leaks",
    "normal_grids",
    "overall",
    "score_separation_margin",
)


def _diagnostic_fields(row: Mapping[str, object], class_code: str) -> dict[str, object]:
    missed = row["defect_instances"] - row["covered_defect_instances"]
    return {
        "missed_defect_instances": missed if class_code == "scratch" else 0,
        "grid_false_positives": row["grid_fp"],
        "normal_grid_leaks": row["normal_grid_leaks"],
        "extent_failure": row["asserted_grid_occupancy_p95"] > OCCUPANCY_FAILURE_BOUNDARY,
        "separation_failure": row["score_separation_margin"] <= 0.0,
    }


def _validate_evidence_row(row: Mapping[str, object], seed: int, class_code: str) -> None:
    missing = set(_EVIDENCE_FIELDS) - set(row)
    if missing:
        raise ValueError(
            f"Ticket 34 final evidence field missing: seed={seed} class={class_code} "
            + ", ".join(sorted(missing))
        )
    for field in _EVIDENCE_FIELDS:
        value = row[field]
        if field == "overall":
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"Ticket 34 final evidence value invalid: seed={seed} class={class_code} field={field}")
    if row["defect_instances"] != FINAL_INSTANCES_PER_SEED_CLASS:
        raise ValueError(f"Ticket 34 final instance count drift: seed={seed} class={class_code}")
    if row["covered_defect_instances"] < 0 or row["covered_defect_instances"] > row["defect_instances"]:
        raise ValueError(f"Ticket 34 final coverage count invalid: seed={seed} class={class_code}")
    if row["score_separation_margin"] <= 0.0:
        raise ValueError(f"Ticket 34 final separation margin is not positive: seed={seed} class={class_code}")


def _validate_development_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    expected = {"source_sha256": TICKET31_CORPUS_SOURCE_SHA256, **DEVELOPMENT_METADATA}
    if not isinstance(metadata, Mapping) or dict(metadata) != expected:
        raise ValueError("Ticket 31 development corpus metadata drift")
    return dict(metadata)


def _read_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}: {error}") from error


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "CLASS_CODES",
    "FINAL_MEMBERS",
    "SCHEMA",
    "TICKET31_CORPUS_SOURCE_SHA256",
    "TICKET34_REPORT_SHA256",
    "build_ticket35_density_diagnosis",
    "canonical_density_diagnosis_json",
    "main",
    "read_ticket31_development_metadata",
    "validate_ticket34_final_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
