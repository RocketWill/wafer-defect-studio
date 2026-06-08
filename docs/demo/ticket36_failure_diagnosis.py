"""Evidence-only diagnosis of the failed Ticket 35 density micro-overfit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path


SCHEMA = "ticket36-failure-diagnosis.v1"
REPORT_SCHEMA = "ticket35-density-micro-overfit.v1"
ARTIFACT_INVENTORY_SCHEMA = "ticket35-density-micro-overfit-artifacts.v1"
TICKET35_REPORT_SHA256 = (
    "8406e9eb4033f8da12913dc8ca1234d116f77a615e70b0f19a76292283e9494a"
)
MISSING_SPATIAL_ARTIFACTS = (
    "checkpoint",
    "maps",
    "per_instance_matching",
)
CASE_IDS = (
    "ticket35-density-101-train-particle-000-001-06",
    "ticket35-density-101-train-particle-000-150-10",
    "ticket35-density-101-train-scratch-001-000-01",
    "ticket35-density-101-train-scratch-150-000-05",
)
FINAL_MEMBERS = (
    "ticket30-evidence-17.png",
    "ticket30-evidence-42.png",
    "ticket30-evidence-91.png",
)
RESERVED_FINAL_MEMBERS = (
    "ticket35-final-member-01",
    "ticket35-final-member-02",
    "ticket35-final-member-03",
)
FORBIDDEN_ARTIFACT_TOKENS = (
    "ticket34-final",
    "ticket30-evidence-",
    "ticket35-final-member-",
    *RESERVED_FINAL_MEMBERS,
)
_CASE_FIELDS = (
    "case_id",
    "class_code",
    "composition",
    "filename",
    "instance_count",
    "instance_recall",
    "normal_grid_leak_rate",
    "asserted_grid_occupancy_p95",
    "outside_declared_extent_cells",
    "outside_declared_extent_response",
    "overall",
    "threshold",
)


def read_ticket35_density_micro_overfit_report(path: Path) -> dict[str, object]:
    """Read the tracked report only after verifying its immutable source SHA."""

    try:
        report_bytes = Path(path).read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read Ticket 35 density report: {path}: {error}") from error
    report_text = _decode(report_bytes, "Ticket 35 density report")
    return _validate_report_text(report_text)


def build_ticket36_failure_diagnosis(
    report_text: str,
    artifact_inventory: Mapping[str, object],
    *,
    report_path: str = "docs/demo/ticket35-density-micro-overfit.json",
    inventory_path: str = "temp/ticket35-density-micro-overfit/artifact-inventory.json",
) -> dict[str, object]:
    """Build a canonical diagnosis from published metrics and explicit inventory.

    No model, map, threshold, component, or instance inference occurs here.
    """

    report = _validate_report_text(report_text)
    inventory = _validate_artifact_inventory(artifact_inventory)
    rows = []
    for case_id in CASE_IDS:
        row = dict(report["per_case"][case_id])
        row["covered_defect_instances"] = _inferred_covered_instances(row)
        row["missed_defect_instances"] = row["instance_count"] - row["covered_defect_instances"]
        row["covered_defect_instances_inferred"] = True
        rows.append(row)
    outside_values = [float(row["outside_declared_extent_response"]) for row in rows]
    particle150 = next(
        row
        for row in rows
        if row["case_id"] == "ticket35-density-101-train-particle-000-150-10"
    )
    missing = list(MISSING_SPATIAL_ARTIFACTS)
    unavailable = {
        "status": "unavailable_missing_spatial_artifacts",
        "missing_artifacts": missing,
    }
    return {
        "schema": SCHEMA,
        "sources": {
            "ticket35_density_micro_overfit_report": {
                "path": report_path,
                "sha256": TICKET35_REPORT_SHA256,
            },
            "artifact_inventory": {
                "path": inventory_path,
                "sha256": _sha256_text(canonical_artifact_inventory_json(inventory)),
            },
        },
        "artifact_inventory": {
            "root": inventory["root"],
            "source_pngs": [
                {"case_id": item["case_id"], "path": item["path"]}
                for item in inventory["source_pngs"]
            ],
            "source_png_count": len(inventory["source_pngs"]),
            "missing": missing,
        },
        "per_case": rows,
        "observed_failure": {
            "outside_declared_extent_response": {
                "min": min(outside_values),
                "max": max(outside_values),
                "case_count": len(outside_values),
            },
            "particle150": {
                "covered_defect_instances": particle150["covered_defect_instances"],
                "defect_instances": particle150["instance_count"],
                "missed_defect_instances": particle150["instance_count"]
                - particle150["covered_defect_instances"],
                "instance_recall": particle150["instance_recall"],
            },
        },
        "root_symptom": {
            "far_activation": "dominant",
            "particle150_recall_gap": {
                "covered": particle150["covered_defect_instances"],
                "total": particle150["instance_count"],
                "missed": particle150["instance_count"]
                - particle150["covered_defect_instances"],
            },
            "causal_claim": False,
        },
        "spatial_analysis": {
            "component_analysis": dict(unavailable),
            "miss_location": dict(unavailable),
            "boundary_classification": dict(unavailable),
            "merge_classification": dict(unavailable),
        },
        "decision": {
            "overall": "BLOCKED",
            "recommendation": "cam_v2",
            "next_slice": "freeze_localization_and_artifact_contract",
            "ticket35_06": "blocked",
            "final_calibration": "forbidden",
            "rerun_or_tuning": "forbidden",
        },
        "boundary": {
            "ticket34_final_members": list(FINAL_MEMBERS),
            "ticket34_final_use": "forbidden",
            "ticket35_reserved_final_members": list(RESERVED_FINAL_MEMBERS),
            "ticket35_reserved_final_use": "forbidden",
            "final_members_used": False,
            "spatial_mil_v7_status": "experimental",
            "cam_default": "cam_v2",
        },
        "claims": [
            "aggregate diagnosis only",
            "not segmentation",
            "not Neurocle equivalence",
            "not production accuracy",
        ],
    }


def canonical_ticket36_failure_diagnosis_json(payload: Mapping[str, object]) -> str:
    """Serialize a diagnosis deterministically for the tracked artifact."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_artifact_inventory_json(payload: Mapping[str, object]) -> str:
    """Serialize an explicit temporary-output inventory deterministically."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--artifact-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report_path = args.report
        inventory_path = args.artifact_inventory
        report_text = report_path.read_text(encoding="utf-8")
        try:
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid Ticket 35 artifact inventory: {error}") from error
        diagnosis = build_ticket36_failure_diagnosis(
            report_text,
            inventory,
            report_path=report_path.as_posix(),
            inventory_path=inventory_path.as_posix(),
        )
        output = args.output.resolve()
        if output.exists():
            raise ValueError(f"Ticket 36 diagnosis output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            canonical_ticket36_failure_diagnosis_json(diagnosis) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"Ticket 36 failure diagnosis failed: {error}", file=sys.stderr)
        return 1
    print(f"Ticket 36 failure diagnosis: {output}", flush=True)
    return 0


def _validate_report_text(report_text: str) -> dict[str, object]:
    if not isinstance(report_text, str):
        raise ValueError("Ticket 35 density report must be text")
    actual_sha256 = _sha256_text(report_text)
    if actual_sha256 != TICKET35_REPORT_SHA256:
        raise ValueError(
            "Ticket 35 density micro-overfit report SHA-256 mismatch: "
            f"expected={TICKET35_REPORT_SHA256} actual={actual_sha256}"
        )
    try:
        report = json.loads(report_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Ticket 35 density report: {error}") from error
    if not isinstance(report, Mapping):
        raise ValueError("Ticket 35 density report must be an object")
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("Ticket 35 density report schema drift")
    if report.get("overall") != "FAIL" or report.get("recommendation") != "cam_v2":
        raise ValueError("Ticket 35 density report decision drift")
    if report.get("spatial_mil_v7_status") != "experimental":
        raise ValueError("Ticket 35 density report status drift")
    if report.get("case_count") != 4 or report.get("threshold") != 0.5:
        raise ValueError("Ticket 35 density report case/threshold drift")
    if report.get("calibration") != "forbidden":
        raise ValueError("Ticket 35 density report calibration drift")
    per_case = report.get("per_case")
    if not isinstance(per_case, Mapping) or set(per_case) != set(CASE_IDS):
        raise ValueError("Ticket 35 density report case membership drift")
    for case_id in CASE_IDS:
        row = per_case[case_id]
        if not isinstance(row, Mapping) or set(_CASE_FIELDS) - set(row):
            raise ValueError(f"Ticket 35 density report row malformed: {case_id}")
        if row.get("case_id") != case_id or row.get("overall") != "FAIL":
            raise ValueError(f"Ticket 35 density report row identity drift: {case_id}")
        for field in _CASE_FIELDS:
            value = row[field]
            if field in {"case_id", "class_code", "composition", "filename", "overall"}:
                if not isinstance(value, str):
                    raise ValueError(f"Ticket 35 density report text field invalid: {case_id} {field}")
            elif not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"Ticket 35 density report metric invalid: {case_id} {field}")
        if _inferred_covered_instances(row) > row["instance_count"]:
            raise ValueError(f"Ticket 35 density report coverage invalid: {case_id}")
    particle150 = per_case[CASE_IDS[1]]
    if _inferred_covered_instances(particle150) != 143 or particle150["instance_count"] != 150:
        raise ValueError("Ticket 35 particle-150 coverage drift")
    outside = [per_case[case_id]["outside_declared_extent_response"] for case_id in CASE_IDS]
    if min(outside) != 0.9273584905660377 or max(outside) != 0.9737548147164298:
        raise ValueError("Ticket 35 outside-response range drift")
    return dict(report)


def _validate_artifact_inventory(inventory: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(inventory, Mapping):
        raise ValueError("Ticket 35 artifact inventory must be an object")
    expected_keys = {"schema", "root", "source_pngs", "missing"}
    if set(inventory) != expected_keys:
        raise ValueError("Ticket 35 artifact inventory fields drift")
    if inventory.get("schema") != ARTIFACT_INVENTORY_SCHEMA:
        raise ValueError("Ticket 35 artifact inventory schema drift")
    if not isinstance(inventory.get("root"), str) or not inventory["root"]:
        raise ValueError("Ticket 35 artifact inventory root invalid")
    _reject_forbidden_artifact_token(inventory["root"], "root")
    source_pngs = inventory.get("source_pngs")
    if not isinstance(source_pngs, list) or len(source_pngs) != len(CASE_IDS):
        raise ValueError("Ticket 35 artifact inventory source PNG count drift")
    by_case: dict[str, dict[str, object]] = {}
    for item in source_pngs:
        if not isinstance(item, Mapping) or set(item) != {"path", "case_id", "status"}:
            raise ValueError("Ticket 35 artifact inventory source PNG row malformed")
        case_id = item.get("case_id")
        _reject_forbidden_artifact_token(case_id, f"source PNG case_id={case_id}")
        if case_id not in CASE_IDS or case_id in by_case:
            raise ValueError("Ticket 35 artifact inventory source PNG membership drift")
        if item.get("status") != "present" or not isinstance(item.get("path"), str):
            raise ValueError(f"Ticket 35 artifact inventory source PNG status invalid: {case_id}")
        _reject_forbidden_artifact_token(item["path"], f"source PNG case_id={case_id}")
        expected_name = f"{case_id}.png"
        if not item["path"].replace("\\", "/").endswith(expected_name):
            raise ValueError(f"Ticket 35 artifact inventory source PNG path drift: {case_id}")
        by_case[case_id] = dict(item)
    if set(by_case) != set(CASE_IDS):
        raise ValueError("Ticket 35 artifact inventory source PNG membership drift")
    missing = inventory.get("missing")
    if missing != list(MISSING_SPATIAL_ARTIFACTS):
        raise ValueError("Ticket 35 artifact inventory missing set drift")
    return {
        "schema": inventory["schema"],
        "root": inventory["root"],
        "source_pngs": [by_case[case_id] for case_id in CASE_IDS],
        "missing": list(MISSING_SPATIAL_ARTIFACTS),
    }


def _decode(payload: bytes, label: str) -> str:
    try:
        return payload.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8: {error}") from error


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _inferred_covered_instances(row: Mapping[str, object]) -> int:
    """Derive a count from aggregate recall; no per-instance matching is implied."""

    return int(round(float(row["instance_recall"]) * int(row["instance_count"])))


def _reject_forbidden_artifact_token(value: object, label: str) -> None:
    if isinstance(value, str) and any(
        token.lower() in value.lower() for token in FORBIDDEN_ARTIFACT_TOKENS
    ):
        raise ValueError(f"Ticket 35 artifact inventory forbidden final token: {label}")


__all__ = [
    "ARTIFACT_INVENTORY_SCHEMA",
    "CASE_IDS",
    "FORBIDDEN_ARTIFACT_TOKENS",
    "SCHEMA",
    "TICKET35_REPORT_SHA256",
    "build_ticket36_failure_diagnosis",
    "canonical_artifact_inventory_json",
    "canonical_ticket36_failure_diagnosis_json",
    "main",
    "read_ticket35_density_micro_overfit_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
