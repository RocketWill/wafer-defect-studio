"""Seal and one-time execution seam for the Ticket 34 final gate.

The seal only reads source/report/artifact bytes.  The final runner receives a
corpus provider, but it cannot call that provider until every sealed input has
passed content-address and contract validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from docs.demo.ticket31_contract import DEVELOPMENT_SEEDS
from docs.demo.ticket34_contract import (
    FINAL_MEMBERS,
    FINAL_TARGETS,
    FROZEN_CORPUS_SHA256,
    SCORE_SEPARATION_MARGIN_MINIMUM,
    TICKET33_REPORT_SHA256,
    V6_POLICY,
    build_ticket34_contract,
    canonical_contract_json,
    validate_ticket33_development_report,
)
from docs.demo.wafer_quality_evidence import (
    WaferEvidenceCase,
    compute_wafer_quality_evidence,
)
from wafer_defect_studio.grid_geometry import annotation_grids


SCHEMA = "ticket34-final-seal.v1"
FINAL_GATE_SCHEMA = "ticket34-final-gate.v1"
FINAL_GATE_CLASS_CODES = ("scratch", "particle")
CANDIDATE_ARTIFACT_ROLES = (
    "checkpoint",
    "configuration",
    "training_loss",
    "validation_maps",
    "validation_metrics",
    "validation_thresholds",
)
SOURCE_ROLES = (
    "ticket30_corpus_source",
    "ticket33_development_report",
    "ticket34_contract",
    "ticket34_development_candidate_report",
    "ticket34_development_candidate_source",
    "ticket34_final_gate_source",
)
_REQUIRED_ROLES = SOURCE_ROLES + CANDIDATE_ARTIFACT_ROLES
_ENVIRONMENT_FIELDS = ("python", "torch", "cuda", "gpu")
_FINAL_GATE_MEMBERS = {
    seed: filename for seed, filename in zip((17, 42, 91), FINAL_MEMBERS, strict=True)
}


def _development_members() -> dict[str, list[str]]:
    return {
        split: [
            f"ticket31-{seed}-{split}-{index:02d}.png"
            for seed in DEVELOPMENT_SEEDS
            for index in range(24)
        ]
        for split in ("train", "validation")
    }


EXPECTED_DEVELOPMENT_MEMBERS = _development_members()


def build_ticket34_final_contract() -> dict[str, object]:
    """Build the tracked contract from the exact 120/120 development names."""

    return build_ticket34_contract(
        EXPECTED_DEVELOPMENT_MEMBERS["train"],
        EXPECTED_DEVELOPMENT_MEMBERS["validation"],
    )


def canonical_final_contract_json(contract: Mapping[str, object]) -> str:
    return canonical_contract_json(contract)


def build_ticket34_final_seal(
    *,
    root: Path,
    contract_path: Path,
    ticket33_report_path: Path,
    candidate_report_path: Path,
    ticket30_source_path: Path,
    candidate_runner_path: Path,
    final_gate_runner_path: Path,
    candidate_artifact_paths: Mapping[str, Path],
    git_commit: str,
) -> dict[str, object]:
    """Build a content-addressed seal without opening final corpus pixels."""

    root = Path(root).resolve()
    contract_file = Path(contract_path).resolve()
    report_file = Path(ticket33_report_path).resolve()
    candidate_file = Path(candidate_report_path).resolve()
    corpus_file = Path(ticket30_source_path).resolve()
    candidate_source = Path(candidate_runner_path).resolve()
    final_source = Path(final_gate_runner_path).resolve()

    contract_text = _read_text(contract_file, "Ticket 34 final contract")
    contract = _validate_contract_text(contract_text)
    report_text = _read_text(report_file, "Ticket 33 development report")
    validate_ticket33_development_report(report_text)
    candidate_text = _read_text(candidate_file, "Ticket 34 development candidate report")
    candidate = _validate_candidate_report(candidate_text, contract)
    _validate_corpus_source(_read_text(corpus_file, "Ticket 30 corpus source"), contract)

    paths = {name: Path(path).resolve() for name, path in candidate_artifact_paths.items()}
    if tuple(paths) != CANDIDATE_ARTIFACT_ROLES:
        raise ValueError(
            "candidate artifact roles must be exactly "
            + ", ".join(CANDIDATE_ARTIFACT_ROLES)
        )
    artifact_entries = {
        name: _file_entry(name, path, root) for name, path in paths.items()
    }
    _validate_candidate_artifact_links(candidate, artifact_entries, root)

    files = [
        _file_entry("ticket34_contract", contract_file, root),
        _file_entry("ticket33_development_report", report_file, root),
        _file_entry("ticket34_development_candidate_report", candidate_file, root),
        _file_entry("ticket30_corpus_source", corpus_file, root),
        _file_entry("ticket34_development_candidate_source", candidate_source, root),
        _file_entry("ticket34_final_gate_source", final_source, root),
        *artifact_entries.values(),
    ]
    files.sort(key=lambda entry: (entry["role"], entry["path"]))
    seal: dict[str, object] = {
        "schema": SCHEMA,
        "git_commit": _validate_git_commit(git_commit),
        "git_commit_role": "pre_seal_base",
        "contract_sha256": _sha256(contract_text.encode("utf-8")),
        "ticket30_manifest_sha256": contract["ticket30_manifest_sha256"],
        "ticket33_report_sha256": TICKET33_REPORT_SHA256,
        "final_corpus_sha256": contract["final_corpus_sha256"],
        "final_members": list(FINAL_MEMBERS),
        "final_targets": dict(FINAL_TARGETS),
        "score_separation_margin": {"operator": ">", "minimum": SCORE_SEPARATION_MARGIN_MINIMUM},
        "v6_policy": dict(V6_POLICY),
        "checkpoint_selection_sources": ["validation"],
        "default_until_final_pass": "cam_v2",
        "candidate": {
            "report_sha256": _sha256(candidate_text.encode("utf-8")),
            "overall": candidate["overall"],
            "final_gate_eligible": candidate["final_gate_eligible"],
            "training_seed": candidate["training_seed"],
            "epochs": candidate["epochs"],
            "candidate_count": candidate["candidate_count"],
            "candidate_epoch": candidate["candidate_epoch"],
            "selection_source": candidate["selection_source"],
            "environment": dict(candidate["environment"]),
            "validation_threshold_source": candidate["validation_thresholds"]["source_split"],
        },
        "runner": {
            "candidate_source_role": "ticket34_development_candidate_source",
            "final_gate_source_role": "ticket34_final_gate_source",
            "final_calibration": "forbidden",
            "corpus_provider": "after_seal_validation_only",
        },
        "required_roles": list(_REQUIRED_ROLES),
        "files": files,
    }
    seal["seal_sha256"] = hash_ticket34_final_seal(seal)
    return seal


def validate_ticket34_final_seal(
    seal: Mapping[str, object], *, root: Path
) -> dict[str, object]:
    """Validate the seal and every declared byte before final corpus access."""

    if not isinstance(seal, Mapping) or seal.get("schema") != SCHEMA:
        raise ValueError("invalid Ticket 34 final seal schema")
    if not isinstance(seal.get("seal_sha256"), str):
        raise ValueError("Ticket 34 final seal SHA-256 is missing")
    if hash_ticket34_final_seal(seal) != seal["seal_sha256"]:
        raise ValueError("seal manifest hash mismatch")
    expected_keys = {
        "schema",
        "git_commit",
        "git_commit_role",
        "contract_sha256",
        "ticket30_manifest_sha256",
        "ticket33_report_sha256",
        "final_corpus_sha256",
        "final_members",
        "final_targets",
        "score_separation_margin",
        "v6_policy",
        "checkpoint_selection_sources",
        "default_until_final_pass",
        "candidate",
        "runner",
        "required_roles",
        "files",
        "seal_sha256",
    }
    if set(seal) != expected_keys:
        raise ValueError("Ticket 34 final seal fields drift")
    _validate_git_commit(seal["git_commit"])
    if seal["git_commit_role"] != "pre_seal_base":
        raise ValueError("Ticket 34 final seal commit role drift")
    if seal["required_roles"] != list(_REQUIRED_ROLES):
        raise ValueError("Ticket 34 final seal required roles drift")
    if seal["final_members"] != list(FINAL_MEMBERS):
        raise ValueError("Ticket 34 final seal final membership drift")
    if seal["final_targets"] != FINAL_TARGETS:
        raise ValueError("Ticket 34 final seal final targets drift")
    if seal["score_separation_margin"] != {
        "operator": ">",
        "minimum": SCORE_SEPARATION_MARGIN_MINIMUM,
    }:
        raise ValueError("Ticket 34 final seal separation policy drift")
    if seal["v6_policy"] != V6_POLICY or seal["checkpoint_selection_sources"] != ["validation"]:
        raise ValueError("Ticket 34 final seal v6 policy drift")
    if seal["default_until_final_pass"] != "cam_v2":
        raise ValueError("Ticket 34 final seal default policy drift")
    candidate = seal["candidate"]
    if not isinstance(candidate, Mapping):
        raise ValueError("Ticket 34 final seal candidate metadata is invalid")
    _validate_candidate_metadata(candidate)
    runner = seal["runner"]
    if not isinstance(runner, Mapping) or runner.get("final_calibration") != "forbidden":
        raise ValueError("Ticket 34 final seal runner policy drift")
    if runner.get("corpus_provider") != "after_seal_validation_only":
        raise ValueError("Ticket 34 final seal corpus-provider policy drift")

    entries = seal["files"]
    if not isinstance(entries, list):
        raise ValueError("Ticket 34 final seal files must be a list")
    if entries != sorted(entries, key=lambda entry: (entry["role"], entry["path"])):
        raise ValueError("Ticket 34 final seal file order drift")
    roles: list[str] = []
    by_role: dict[str, Mapping[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"role", "path", "bytes", "sha256"}:
            raise ValueError("Ticket 34 final seal file entry is invalid")
        role = entry["role"]
        if not isinstance(role, str) or role in by_role:
            raise ValueError("Ticket 34 final seal file roles are duplicated")
        if not isinstance(entry["path"], str) or not entry["path"]:
            raise ValueError(f"Ticket 34 final seal file path is invalid: {role}")
        if not isinstance(entry["bytes"], int) or entry["bytes"] < 0:
            raise ValueError(f"Ticket 34 final seal file byte count is invalid: {role}")
        if not _is_hex(entry["sha256"], 64):
            raise ValueError(f"Ticket 34 final seal file SHA-256 is invalid: {role}")
        roles.append(role)
        by_role[role] = entry
        _verify_file_entry(entry, root)
    if tuple(roles) != tuple(sorted(roles)) or set(roles) != set(_REQUIRED_ROLES):
        raise ValueError("Ticket 34 final seal file role membership drift")

    contract_text = _read_entry_text(by_role["ticket34_contract"], root)
    contract = _validate_contract_text(contract_text)
    if _sha256(contract_text.encode("utf-8")) != seal["contract_sha256"]:
        raise ValueError("Ticket 34 final contract SHA-256 drift")
    if seal["ticket30_manifest_sha256"] != contract["ticket30_manifest_sha256"]:
        raise ValueError("Ticket 34 final seal Ticket 30 manifest drift")
    if seal["final_corpus_sha256"] != contract["final_corpus_sha256"]:
        raise ValueError("Ticket 34 final seal frozen corpus drift")

    report_text = _read_entry_text(by_role["ticket33_development_report"], root)
    try:
        validate_ticket33_development_report(report_text)
    except ValueError as error:
        raise ValueError(f"Ticket 33 development report validation failed: {error}") from error
    if seal["ticket33_report_sha256"] != _sha256(report_text.encode("utf-8")):
        raise ValueError("Ticket 33 report SHA-256 drift")

    candidate_text = _read_entry_text(by_role["ticket34_development_candidate_report"], root)
    candidate_report = _validate_candidate_report(candidate_text, contract)
    candidate_sha = _sha256(candidate_text.encode("utf-8"))
    if candidate["report_sha256"] != candidate_sha:
        raise ValueError("Ticket 34 candidate report SHA-256 drift")
    if candidate["environment"] != candidate_report["environment"]:
        raise ValueError("Ticket 34 candidate environment drift")
    _validate_corpus_source(
        _read_entry_text(by_role["ticket30_corpus_source"], root), contract
    )
    _validate_candidate_artifact_links(candidate_report, by_role, root)
    return dict(seal)


def canonical_final_seal_json(seal: Mapping[str, object]) -> str:
    return json.dumps(seal, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hash_ticket34_final_seal(seal: Mapping[str, object]) -> str:
    payload = dict(seal)
    payload.pop("seal_sha256", None)
    return _sha256(canonical_final_seal_json(payload).encode("utf-8"))


def run_final_gate(
    seal: Mapping[str, object],
    *,
    seal_root: Path,
    output_root: Path,
    corpus_provider: Callable[[], Sequence[object]],
    model_loader: Callable[[Path], object],
    scorer: Callable[[object, object, np.ndarray], np.ndarray],
    renderer: Callable[[object], np.ndarray],
    bounds: object | None = None,
) -> dict[str, object]:
    """Run the frozen final gate using thresholds sealed from validation.

    ``corpus_provider`` is deliberately called only after seal validation.
    This function never calibrates thresholds from final cases.
    """

    validated = validate_ticket34_final_seal(seal, root=Path(seal_root).resolve())
    root = Path(seal_root).resolve()
    entries = {entry["role"]: entry for entry in validated["files"]}
    threshold_payload = _read_json_entry(entries["validation_thresholds"], root)
    thresholds = _sealed_thresholds(threshold_payload)
    checkpoint = _resolve_entry_path(entries["checkpoint"], root)
    model = model_loader(checkpoint)

    # The provider is the only boundary that may expose final cases.
    cases = tuple(corpus_provider())
    _validate_final_membership(cases)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_seed: dict[str, object] = {}
    map_artifacts: dict[str, object] = {}
    for case in cases:
        seed = int(case.seed)
        source = np.asarray(renderer(case))
        absolute_map = np.asarray(scorer(model, bounds, source), dtype=np.float32)
        expected_shape = (
            int(case.oracle.image_height),
            int(case.oracle.image_width),
            len(FINAL_GATE_CLASS_CODES),
        )
        if absolute_map.shape != expected_shape:
            raise ValueError(
                f"{case.filename}: final map shape expected {expected_shape}, actual {absolute_map.shape}"
            )
        if not np.isfinite(absolute_map).all() or absolute_map.min() < 0 or absolute_map.max() > 1:
            raise ValueError(f"{case.filename}: final map values must be finite in [0, 1]")
        grids = annotation_grids(case.oracle.image_width, case.oracle.image_height, 512, 512)
        evidence = WaferEvidenceCase(case.filename, "test", case.oracle, grids, absolute_map)
        metrics = compute_wafer_quality_evidence((evidence,), FINAL_GATE_CLASS_CODES, thresholds)
        separation = _separation_rows(case, absolute_map, grids)
        rows = {}
        for code in FINAL_GATE_CLASS_CODES:
            row = dict(metrics["per_class"][code])
            row["score_separation_margin"] = separation[code]["margin"]
            row["overall"] = "PASS" if _final_row_passes(row) else "FAIL"
            rows[code] = row
        per_seed[str(seed)] = {"seed": seed, "filename": case.filename, "per_class": rows}

        map_path = output / f"seed-{seed}-maps.npz"
        if map_path.exists():
            raise ValueError(f"final map output already exists: {map_path}")
        np.savez_compressed(
            map_path,
            seed=np.asarray(seed),
            filename=np.asarray(case.filename),
            class_codes=np.asarray(FINAL_GATE_CLASS_CODES),
            maps=absolute_map.astype(np.float16),
            split=np.asarray("test"),
        )
        map_artifacts[str(seed)] = {"seed": seed, **_artifact_for_output(map_path, output)}

    overall = "PASS" if all(
        row["overall"] == "PASS"
        for seed in per_seed.values()
        for row in seed["per_class"].values()
    ) else "FAIL"
    report: dict[str, object] = {
        "schema": FINAL_GATE_SCHEMA,
        "overall": overall,
        "recommendation": "spatial_mil_v6" if overall == "PASS" else "cam_v2",
        "seal_sha256": validated["seal_sha256"],
        "final_members": list(FINAL_MEMBERS),
        "class_codes": list(FINAL_GATE_CLASS_CODES),
        "threshold_source": "sealed_validation_artifact",
        "final_calibration": "forbidden",
        "per_seed": per_seed,
        "map_artifacts": map_artifacts,
        "claims": [
            "frozen final held-out evidence",
            "approximate localization",
            "not segmentation",
            "not Neurocle equivalence",
            "not production accuracy",
        ],
    }
    report_path = output / "ticket34-final-gate.json"
    if report_path.exists():
        raise ValueError(f"final gate report already exists: {report_path}")
    report_path.write_text(canonical_final_seal_json(report) + "\n", encoding="utf-8")
    return report


def _validate_contract_text(text: str) -> dict[str, object]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Ticket 34 final contract: {error}") from error
    expected = build_ticket34_final_contract()
    if text not in (canonical_contract_json(payload), canonical_contract_json(payload) + "\n"):
        raise ValueError("Ticket 34 final contract serialization is not canonical")
    if payload != expected:
        raise ValueError("Ticket 34 final contract fields drift")
    return dict(payload)


def _validate_candidate_report(text: str, contract: Mapping[str, object]) -> dict[str, object]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Ticket 34 development candidate report: {error}") from error
    if not isinstance(payload, Mapping) or payload.get("schema") != "ticket34-development-candidate.v1":
        raise ValueError("Ticket 34 development candidate schema drift")
    required = {
        "schema", "overall", "final_gate_eligible", "training_seed", "epochs",
        "candidate_count", "candidate_epoch", "selection_source", "membership",
        "configuration", "validation_thresholds", "validation_metrics", "environment",
        "artifacts",
    }
    if not required.issubset(payload):
        raise ValueError("Ticket 34 development candidate fields are incomplete")
    if payload["overall"] != "PASS" or payload["final_gate_eligible"] is not True:
        raise ValueError("Ticket 34 development candidate is not final-gate eligible")
    if (
        payload["training_seed"] != 101
        or payload["epochs"] != 30
        or payload["candidate_count"] != 1
        or payload["candidate_epoch"] != 30
        or payload["selection_source"] != "validation_spatial_metrics"
    ):
        raise ValueError("Ticket 34 development candidate selection is not frozen")
    membership = payload["membership"]
    if not isinstance(membership, Mapping) or (
        membership.get("train") != contract["development_members"]["train"]
        or membership.get("validation") != contract["development_members"]["validation"]
    ):
        raise ValueError("Ticket 34 development candidate membership drift")
    configuration = payload["configuration"]
    config_membership = configuration.get("membership") if isinstance(configuration, Mapping) else None
    if not isinstance(config_membership, Mapping) or (
        config_membership.get("train") != membership["train"]
        or config_membership.get("validation") != membership["validation"]
    ):
        raise ValueError("Ticket 34 development candidate configuration membership drift")
    thresholds = payload["validation_thresholds"]
    metrics = payload["validation_metrics"]
    if not isinstance(thresholds, Mapping) or thresholds.get("source_split") != "validation":
        raise ValueError("Ticket 34 candidate thresholds are not validation-only")
    if not isinstance(metrics, Mapping) or (
        metrics.get("source_split") != "validation"
        or metrics.get("selection_source") != "validation_spatial_metrics"
    ):
        raise ValueError("Ticket 34 candidate metrics are not validation-only")
    _validate_candidate_metadata({
        "overall": payload["overall"],
        "final_gate_eligible": payload["final_gate_eligible"],
        "training_seed": payload["training_seed"],
        "epochs": payload["epochs"],
        "candidate_count": payload["candidate_count"],
        "candidate_epoch": payload["candidate_epoch"],
        "selection_source": payload["selection_source"],
        "environment": payload["environment"],
        "validation_threshold_source": thresholds["source_split"],
    })
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(CANDIDATE_ARTIFACT_ROLES):
        raise ValueError("Ticket 34 candidate artifact membership drift")
    return dict(payload)


def _validate_candidate_metadata(candidate: Mapping[str, object]) -> None:
    if (
        candidate.get("overall") != "PASS"
        or candidate.get("final_gate_eligible") is not True
        or candidate.get("training_seed") != 101
        or candidate.get("epochs") != 30
        or candidate.get("candidate_count") != 1
        or candidate.get("candidate_epoch") != 30
        or candidate.get("selection_source") != "validation_spatial_metrics"
        or candidate.get("validation_threshold_source") != "validation"
    ):
        raise ValueError("Ticket 34 final seal candidate metadata drift")
    environment = candidate.get("environment")
    if not isinstance(environment, Mapping) or set(environment) != set(_ENVIRONMENT_FIELDS):
        raise ValueError("Ticket 34 final seal environment fields drift")
    if any(not isinstance(environment[field], str) or not environment[field] for field in _ENVIRONMENT_FIELDS):
        raise ValueError("Ticket 34 final seal environment is incomplete")
    if environment["gpu"] != "NVIDIA GeForce RTX 3090":
        raise ValueError("Ticket 34 final seal GPU identity drift")


def _validate_candidate_artifact_links(
    candidate: Mapping[str, object], entries: Mapping[str, Mapping[str, object]], root: Path
) -> None:
    artifacts = candidate["artifacts"]
    for role in CANDIDATE_ARTIFACT_ROLES:
        entry = entries.get(role)
        if entry is None:
            raise ValueError(f"Ticket 34 candidate artifact role missing: {role}")
        reference = artifacts[role]
        if not isinstance(reference, Mapping) or set(reference) != {"bytes", "path", "sha256"}:
            raise ValueError(f"Ticket 34 candidate artifact metadata invalid: {role}")
        if reference["bytes"] != entry["bytes"] or reference["sha256"] != entry["sha256"]:
            raise ValueError(f"Ticket 34 candidate artifact integrity drift: {role}")
        _resolve_entry_path(entry, root)


def _validate_corpus_source(text: str, contract: Mapping[str, object]) -> None:
    match = re.search(r"(?m)^FROZEN_CORPUS_SHA256\s*=\s*[\"']([0-9a-f]{64})[\"']\s*$", text)
    if match is None or match.group(1) != contract["final_corpus_sha256"] or match.group(1) != FROZEN_CORPUS_SHA256:
        raise ValueError("Ticket 30 corpus source frozen SHA-256 drift")


def _validate_final_membership(cases: Sequence[object]) -> None:
    if len(cases) != len(_FINAL_GATE_MEMBERS):
        raise ValueError("final corpus case count drift")
    actual: list[tuple[int, str]] = []
    for case in cases:
        seed = getattr(case, "seed", None)
        filename = getattr(case, "filename", None)
        if type(seed) is not int or filename != _FINAL_GATE_MEMBERS.get(seed):
            raise ValueError(f"final corpus member drift: seed={seed!r} filename={filename!r}")
        if getattr(case, "split", None) != "test":
            raise ValueError(f"final corpus split drift: {filename}")
        actual.append((seed, filename))
    if tuple(seed for seed, _ in actual) != (17, 42, 91):
        raise ValueError("final corpus seed order drift")


def _sealed_thresholds(payload: Mapping[str, object]) -> dict[str, float]:
    if payload.get("source_split") != "validation" or payload.get("schema") != "ticket30-map-thresholds.v1":
        raise ValueError("sealed thresholds are not validation-only")
    if payload.get("class_codes") != list(FINAL_GATE_CLASS_CODES):
        raise ValueError("sealed threshold class order drift")
    selected = payload.get("selected")
    if not isinstance(selected, Mapping) or set(selected) != set(FINAL_GATE_CLASS_CODES):
        raise ValueError("sealed threshold class membership drift")
    thresholds = {}
    for code in FINAL_GATE_CLASS_CODES:
        entry = selected[code]
        value = entry.get("threshold") if isinstance(entry, Mapping) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"sealed threshold is invalid: {code}")
        thresholds[code] = float(value)
    return thresholds


def _separation_rows(
    case: object, confidence_map: np.ndarray, grids: Sequence[object]
) -> dict[str, dict[str, float | None]]:
    rows: dict[str, dict[str, float | None]] = {}
    truth = case.oracle.grid_truth(grids)
    for class_index, code in enumerate(FINAL_GATE_CLASS_CODES):
        asserted: list[float] = []
        normal: list[float] = []
        for grid in grids:
            values = confidence_map[
                grid.y : grid.y + grid.height,
                grid.x : grid.x + grid.width,
                class_index,
            ].reshape(-1)
            if not values.size:
                continue
            count = max(1, math.ceil(values.size * 0.01))
            score = float(np.partition(values, values.size - count)[-count:].mean())
            if code in truth[(grid.row, grid.column)]:
                asserted.append(score)
            elif not truth[(grid.row, grid.column)]:
                normal.append(score)
        if not asserted or not normal:
            rows[code] = {"margin": None}
        else:
            rows[code] = {"margin": min(asserted) - max(normal)}
    return rows


def _final_row_passes(row: Mapping[str, object]) -> bool:
    return (
        row.get("defect_instances", 0) >= FINAL_TARGETS["defect_instances"]
        and row.get("defect_coverage_recall") is not None
        and row["defect_coverage_recall"] >= FINAL_TARGETS["defect_coverage_recall"]
        and row.get("grid_precision") is not None
        and row["grid_precision"] >= FINAL_TARGETS["grid_precision"]
        and row.get("grid_recall") is not None
        and row["grid_recall"] >= FINAL_TARGETS["grid_recall"]
        and row.get("normal_grid_leak_rate") is not None
        and row["normal_grid_leak_rate"] <= FINAL_TARGETS["normal_grid_leak_rate"]
        and row.get("asserted_grid_occupancy_p95") is not None
        and row["asserted_grid_occupancy_p95"] <= FINAL_TARGETS["asserted_grid_occupancy_p95"]
        and row.get("score_separation_margin") is not None
        and row["score_separation_margin"] > SCORE_SEPARATION_MARGIN_MINIMUM
    )


def _file_entry(role: str, path: Path, root: Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    relative = _relative_path(resolved, root)
    payload = resolved.read_bytes()
    return {"role": role, "path": relative, "bytes": len(payload), "sha256": _sha256(payload)}


def _artifact_for_output(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _verify_file_entry(entry: Mapping[str, object], root: Path) -> None:
    path = _resolve_entry_path(entry, root)
    payload = path.read_bytes()
    if len(payload) != entry["bytes"] or _sha256(payload) != entry["sha256"]:
        raise ValueError(f"artifact integrity mismatch: {entry['role']}")


def _resolve_entry_path(entry: Mapping[str, object], root: Path) -> Path:
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"sealed artifact path escapes root: {entry['role']}")
    if relative.name in FINAL_MEMBERS:
        raise ValueError(f"final evidence member is forbidden in seal: {relative.name}")
    path = (Path(root) / relative).resolve()
    try:
        path.relative_to(Path(root).resolve())
    except ValueError as error:
        raise ValueError(f"sealed artifact path escapes root: {entry['role']}") from error
    return path


def _read_entry_text(entry: Mapping[str, object], root: Path) -> str:
    return _resolve_entry_path(entry, root).read_bytes().decode("utf-8")


def _read_json_entry(entry: Mapping[str, object], root: Path) -> dict[str, object]:
    try:
        payload = json.loads(_read_entry_text(entry, root))
    except json.JSONDecodeError as error:
        raise ValueError(f"sealed JSON artifact is invalid: {entry['role']}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"sealed JSON artifact must be an object: {entry['role']}")
    return dict(payload)


def _read_text(path: Path, label: str) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}: {error}") from error


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"sealed artifact must be inside root: {path}") from error


def _validate_git_commit(value: object) -> str:
    if not _is_hex(value, 40):
        raise ValueError("invalid git commit")
    return value


def _is_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--ticket33-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--ticket30-source", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--final-gate-source", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    artifacts = {
        role: args.artifact_root / filename
        for role, filename in {
            "checkpoint": "checkpoint.pt",
            "configuration": "candidate-config.json",
            "training_loss": "training-loss.json",
            "validation_maps": "validation-maps.npz",
            "validation_metrics": "validation-metrics.json",
            "validation_thresholds": "validation-thresholds.json",
        }.items()
    }
    seal = build_ticket34_final_seal(
        root=args.repo_root,
        contract_path=args.contract,
        ticket33_report_path=args.ticket33_report,
        candidate_report_path=args.candidate_report,
        ticket30_source_path=args.ticket30_source,
        candidate_runner_path=args.candidate_source,
        final_gate_runner_path=args.final_gate_source,
        candidate_artifact_paths=artifacts,
        git_commit=args.git_commit,
    )
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"final seal output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_final_seal_json(seal) + "\n", encoding="utf-8")
    print(f"Ticket 34 final seal: {output}", flush=True)
    return 0


__all__ = [
    "CANDIDATE_ARTIFACT_ROLES",
    "EXPECTED_DEVELOPMENT_MEMBERS",
    "FINAL_GATE_CLASS_CODES",
    "FINAL_GATE_SCHEMA",
    "SCHEMA",
    "build_ticket34_final_contract",
    "build_ticket34_final_seal",
    "canonical_final_contract_json",
    "canonical_final_seal_json",
    "hash_ticket34_final_seal",
    "run_final_gate",
    "validate_ticket34_final_seal",
]


if __name__ == "__main__":
    raise SystemExit(main())
