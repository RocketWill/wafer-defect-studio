"""Frozen final held-out boundary and v6 policy for Ticket 34."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from docs.demo.ticket30_evidence_corpus import FROZEN_CORPUS_SHA256
from docs.demo.ticket31_contract import FINAL_MEMBERS, FINAL_TARGETS, TICKET30_MANIFEST_SHA256


SCHEMA = "ticket34-final-contract.v1"
TICKET33_REPORT_SHA256 = "c8c54e09148974cce683c47087da5b328af55e98b8fac612920515c3e704d2d2"
SCORE_SEPARATION_MARGIN_MINIMUM = 0.0
V6_POLICY = {
    "positive_spatial_evidence": "top_1_percent",
    "absent_class_suppression": "dense",
    "normal_grid_ranking": "same_image",
}


def validate_ticket33_development_report(report_text: str) -> dict[str, object]:
    """Require the published Ticket 33 development PASS before final use."""

    actual_sha256 = hashlib.sha256(report_text.encode("utf-8")).hexdigest()
    if actual_sha256 != TICKET33_REPORT_SHA256:
        raise ValueError(
            "Ticket 33 development report SHA-256 mismatch: "
            f"expected={TICKET33_REPORT_SHA256} actual={actual_sha256}"
        )
    try:
        report = json.loads(report_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Ticket 33 development report: {error}") from error
    if (
        report.get("schema") != "ticket33-development-gate.v1"
        or report.get("overall") != "PASS"
        or report.get("recommendation") != "run_final_held_out_gate"
        or report.get("spatial_mil_v6_status") != "development_pass"
    ):
        raise ValueError("Ticket 33 development decision is not a frozen PASS")
    seeds = report.get("per_seed")
    if not isinstance(seeds, Mapping) or tuple(seeds) != ("101", "211", "307", "401", "503"):
        raise ValueError("Ticket 33 development seed membership/order drift")
    for seed, result in seeds.items():
        classes = result.get("per_class") if isinstance(result, Mapping) else None
        if not isinstance(classes, Mapping) or tuple(classes) != ("particle", "scratch"):
            raise ValueError(f"Ticket 33 development class membership/order drift: seed={seed}")
        for class_code, row in classes.items():
            if not isinstance(row, Mapping) or row.get("overall") != "PASS":
                raise ValueError(f"Ticket 33 development row is not PASS: seed={seed} class={class_code}")
            margin = row.get("score_separation_margin")
            if not isinstance(margin, (int, float)) or margin <= SCORE_SEPARATION_MARGIN_MINIMUM:
                raise ValueError(
                    f"Ticket 33 development margin is not positive: seed={seed} class={class_code}"
                )
    return {
        "overall": "PASS",
        "recommendation": "run_final_held_out_gate",
        "spatial_mil_v6_status": "development_pass",
    }


def build_ticket34_contract(
    train_members: Sequence[str], validation_members: Sequence[str]
) -> dict[str, object]:
    """Build the canonical v6 final held-out boundary."""

    train = _members(train_members, "train")
    validation = _members(validation_members, "validation")
    overlap = set(train) & set(validation)
    if overlap:
        raise ValueError("train and validation members overlap: " + ", ".join(sorted(overlap)))
    final_overlap = (set(train) | set(validation)) & set(FINAL_MEMBERS)
    if final_overlap:
        raise ValueError(
            "development contains final evidence member: " + ", ".join(sorted(final_overlap))
        )
    return {
        "schema": SCHEMA,
        "ticket30_manifest_sha256": TICKET30_MANIFEST_SHA256,
        "ticket33_report_sha256": TICKET33_REPORT_SHA256,
        "development_members": {"train": list(train), "validation": list(validation)},
        "checkpoint_selection_sources": ["validation"],
        "final_members": list(FINAL_MEMBERS),
        "final_corpus_sha256": FROZEN_CORPUS_SHA256,
        "final_targets": dict(FINAL_TARGETS),
        "score_separation_margin": {"operator": ">", "minimum": SCORE_SEPARATION_MARGIN_MINIMUM},
        "default_until_final_pass": "cam_v2",
        "v6_policy": dict(V6_POLICY),
    }


def canonical_contract_json(contract: Mapping[str, object]) -> str:
    return json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _members(values: Sequence[str], split: str) -> tuple[str, ...]:
    members = tuple(values)
    if not members or any(not isinstance(value, str) or not value for value in members):
        raise ValueError(f"{split} members must be non-empty strings")
    if len(set(members)) != len(members):
        raise ValueError(f"{split} members must be unique")
    return members


__all__ = [
    "FINAL_MEMBERS",
    "FINAL_TARGETS",
    "FROZEN_CORPUS_SHA256",
    "SCORE_SEPARATION_MARGIN_MINIMUM",
    "SCHEMA",
    "TICKET30_MANIFEST_SHA256",
    "TICKET33_REPORT_SHA256",
    "V6_POLICY",
    "build_ticket34_contract",
    "canonical_contract_json",
    "validate_ticket33_development_report",
]
