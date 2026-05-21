"""Frozen evidence boundary and v5 policy for Ticket 31."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from docs.demo.ticket30_evidence_corpus import FROZEN_CORPUS_SHA256


SCHEMA = "ticket31-contract.v1"
DEVELOPMENT_SEEDS = (101, 211, 307, 401, 503)
FINAL_MEMBERS = tuple(f"ticket30-evidence-{seed}.png" for seed in (17, 42, 91))
TICKET30_REPORT_SHA256 = "83ed9fb3da660789bc30be984bf016ff2c0370dc997a368f519cd81280cfb4f4"
TICKET30_MANIFEST_SHA256 = "c726559838cffde843e28662528c0f5ffff175dac39b6ac9b884891ef594c922"
DEVELOPMENT_TARGETS = {
    "defect_coverage_recall": 1.0,
    "grid_precision": 0.97,
    "grid_recall": 0.97,
    "normal_grid_leak_rate": 0.03,
    "asserted_grid_occupancy_p95": 0.20,
}
FINAL_TARGETS = {
    "defect_instances": 150,
    "defect_coverage_recall": 1.0,
    "grid_precision": 0.95,
    "grid_recall": 0.95,
    "normal_grid_leak_rate": 0.05,
    "asserted_grid_occupancy_p95": 0.25,
}
V5_POLICY = {
    "feature_stride": 2,
    "positive_pooling": "normalized_logsumexp",
    "negative_dense_hardest_fraction": 0.01,
    "sparse_probability_budget": 0.01,
    "sparse_loss_weight": 0.25,
    "overlap_loss_weight": 0.10,
    "optimizer": "adamw",
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "epochs": 30,
    "gradient_clip_norm": 5.0,
    "effective_batch_size": 4,
}


def validate_ticket30_baseline(report_text: str) -> dict[str, object]:
    """Resolve the exact published v4 failure report used by Ticket 31."""

    actual_sha256 = hashlib.sha256(report_text.encode("utf-8")).hexdigest()
    if actual_sha256 != TICKET30_REPORT_SHA256:
        raise ValueError(
            "Ticket 30 quality report SHA-256 mismatch: "
            f"expected={TICKET30_REPORT_SHA256} actual={actual_sha256}"
        )
    try:
        report = json.loads(report_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Ticket 30 quality report: {error}") from error
    if (
        report.get("schema") != "ticket30-quality-gate.v1"
        or report.get("manifest_sha256") != TICKET30_MANIFEST_SHA256
        or report.get("overall") != "FAIL"
        or report.get("recommendation") != "cam_v2"
    ):
        raise ValueError("Ticket 30 quality decision does not match the frozen v4 baseline")
    rows = report.get("per_seed")
    if not isinstance(rows, Mapping) or tuple(rows) != ("17", "42", "91"):
        raise ValueError("Ticket 30 baseline seed membership/order drift")
    failed_rows = []
    for seed in ("17", "42", "91"):
        classes = rows[seed]
        if not isinstance(classes, Mapping) or tuple(classes) != ("particle", "scratch"):
            raise ValueError(f"Ticket 30 baseline class membership/order drift: seed={seed}")
        for class_code in ("particle", "scratch"):
            row = classes[class_code]
            if not isinstance(row, Mapping) or row.get("overall") != "FAIL":
                raise ValueError(
                    f"Ticket 30 baseline row must remain failed: seed={seed} class={class_code}"
                )
            failed_rows.append({"seed": int(seed), "class_code": class_code})
    return {
        "overall": "FAIL",
        "recommendation": "cam_v2",
        "failed_rows": failed_rows,
    }


def build_ticket31_contract(
    train_members: Sequence[str], validation_members: Sequence[str]
) -> dict[str, object]:
    """Build the canonical v5 development/final evidence boundary."""

    train = _members(train_members, "train")
    validation = _members(validation_members, "validation")
    overlap = set(train) & set(validation)
    if overlap:
        raise ValueError("train and validation members overlap: " + ", ".join(sorted(overlap)))
    final_overlap = (set(train) | set(validation)) & set(FINAL_MEMBERS)
    if final_overlap:
        raise ValueError("development contains final evidence member: " + ", ".join(sorted(final_overlap)))
    return {
        "schema": SCHEMA,
        "ticket30_report_sha256": TICKET30_REPORT_SHA256,
        "ticket30_manifest_sha256": TICKET30_MANIFEST_SHA256,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "development_members": {"train": list(train), "validation": list(validation)},
        "development_targets": dict(DEVELOPMENT_TARGETS),
        "checkpoint_selection_sources": ["validation"],
        "final_members": list(FINAL_MEMBERS),
        "final_corpus_sha256": FROZEN_CORPUS_SHA256,
        "final_targets": dict(FINAL_TARGETS),
        "default_until_final_pass": "cam_v2",
        "v5_policy": dict(V5_POLICY),
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
    "DEVELOPMENT_SEEDS",
    "DEVELOPMENT_TARGETS",
    "FINAL_MEMBERS",
    "FINAL_TARGETS",
    "V5_POLICY",
    "build_ticket31_contract",
    "canonical_contract_json",
    "validate_ticket30_baseline",
]
