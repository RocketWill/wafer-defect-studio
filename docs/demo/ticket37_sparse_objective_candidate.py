"""Ticket 37 sparse-objective candidate runner and sealed-map evaluator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models import ResNet18_Weights
from torch import Tensor

from docs.demo.ticket35_development_corpus import DevelopmentCase, render_ticket35_development_pixels
from docs.demo.ticket36_single_image_cnn_micro_overfit import (
    _build_batch_supervision,
    _global_tolerance_union,
    _reset_case_seed,
    _score_source_model,
    build_ticket36_cases,
)
from docs.demo.ticket37_proposal_evaluation import (
    build_rendered_truth_components,
    match_proposals_to_rendered_truth,
)
from docs.demo.ticket37_sparse_objective_contract import (
    ARTIFACT_FILENAMES,
    ARTIFACT_ROLES,
    CASE_IDS,
    CASE_SPECS,
    CANDIDATE_ARTIFACT_ROOT,
    CLASS_ORDER,
    FEATURE_STRIDE,
    HASH_POLICY,
    SOURCE_DEPENDENCIES,
    TOLERANCE_PIXELS,
    build_ticket37_sparse_objective_contract,
    canonical_ticket37_sparse_objective_contract_json,
    validate_ticket37_sparse_objective_contract,
)
from docs.demo.wafer_quality_evidence import WaferEvidenceCase, compute_wafer_quality_evidence
from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.grid_geometry import annotation_grids
from wafer_defect_studio.model_registry import (
    WeightsPolicy,
    create_resnet18_spatial_logits_v5,
    set_spatial_transfer_training_mode,
)
from wafer_defect_studio.normalization import NormalizationBounds
from wafer_defect_studio.spatial_mil import (
    dense_absent_class_loss,
    overlap_consistency_loss,
    same_image_grid_ranking_loss,
    sparse_instance_localization_loss,
)
from wafer_defect_studio.training_dataset import enumerate_model_patch_rects, extract_model_patch
from wafer_defect_studio.training_protocol import TrainingConfig


SCHEMA = "ticket37-sparse-objective-candidate.v1"
CHECKPOINT_SCHEMA = "ticket37-sparse-objective-candidate.checkpoint.v1"
MATCHING_SCHEMA = "ticket37-sparse-objective-candidate.matching.v1"
CONFIG_SCHEMA = "ticket37-sparse-objective-candidate.configuration.v1"
REPORT_PATH = Path("docs/demo/ticket37-sparse-objective-candidate.json")
IMAGE_WIDTH = 1536
IMAGE_HEIGHT = 1536
PATCH_SIZE = 128
PATCH_STRIDE = 64
EPOCHS = 30
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP_NORM = 5.0
HARDEST_FRACTION = 0.01
BATCH_NORM_POLICY = "frozen_running_stats"
WEIGHTS_ID = "ResNet18_Weights.IMAGENET1K_V1"
WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
DETERMINISTIC_POLICY = {
    "cublas_workspace_config": ":4096:8",
    "torch_deterministic_algorithms": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
}
RUNTIME_CASE_IDS = (
    "ticket35-density-101-train-scratch-001-000-01",
    "ticket35-density-101-train-scratch-150-000-05",
    "ticket35-density-101-train-particle-000-001-06",
    "ticket35-density-101-train-particle-000-150-10",
)
RUNTIME_CASE_SPECS = (("scratch", 1, 0), ("scratch", 150, 0), ("particle", 0, 1), ("particle", 0, 150))
CONTEXT_WEIGHTS = {
    "dense_absent_class_loss": 0.5,
    "same_image_grid_ranking_loss": 0.25,
    "overlap_consistency_loss": 0.10,
}


def compute_ticket37_objective(
    logits: Tensor,
    targets: Tensor,
    image_group_ids: Sequence[str],
    instance_masks: Tensor,
    instance_batch_indices: Tensor,
    instance_class_indices: Tensor,
    far_negative_masks: Tensor,
    far_batch_indices: Tensor,
    far_class_indices: Tensor,
    patch_rect_groups: Sequence[Sequence[Rect | tuple[int, int, int, int]]],
    *,
    hardest_fraction: float = HARDEST_FRACTION,
) -> Tensor:
    """Return the contracted sparse instance objective with context terms."""

    if not isinstance(logits, Tensor) or logits.ndim != 4:
        raise ValueError("logits must have shape (patches, classes, height, width)")
    if not isinstance(targets, Tensor) or targets.ndim != 2:
        raise ValueError("targets must have shape (bags, classes)")
    if targets.shape[1] != logits.shape[1]:
        raise ValueError("targets class count must match logits")
    if len(image_group_ids) != targets.shape[0]:
        raise ValueError("image_group_ids length must match targets")
    groups = tuple(tuple(_coerce_rect(value) for value in group) for group in patch_rect_groups)
    if len(groups) != targets.shape[0] or any(not group for group in groups):
        raise ValueError("patch_rect_groups must contain one non-empty group per target row")
    if sum(len(group) for group in groups) != logits.shape[0]:
        raise ValueError("patch_rect_groups must cover every logit row")
    sparse = sparse_instance_localization_loss(
        logits,
        instance_masks,
        instance_batch_indices,
        instance_class_indices,
        far_negative_masks,
        far_batch_indices,
        far_class_indices,
        hardest_fraction=hardest_fraction,
    )
    patch_count = len(groups[0])
    if any(len(group) != patch_count for group in groups):
        raise ValueError("all patch groups must have the same patch count")
    bag_logits = logits.reshape(len(groups), patch_count, *logits.shape[1:])
    dense = dense_absent_class_loss(
        bag_logits,
        targets,
        hardest_fraction=hardest_fraction,
    )
    ranking = same_image_grid_ranking_loss(
        bag_logits,
        targets,
        tuple(image_group_ids),
        top_fraction=hardest_fraction,
    )
    overlap_terms = [
        overlap_consistency_loss(
            logits[start : start + patch_count],
            group,
            feature_stride=FEATURE_STRIDE,
        )
        for start, group in zip(range(0, logits.shape[0], patch_count), groups)
    ]
    overlap = torch.stack(overlap_terms).mean()
    return sparse + CONTEXT_WEIGHTS["dense_absent_class_loss"] * dense + CONTEXT_WEIGHTS[
        "same_image_grid_ranking_loss"
    ] * ranking + CONTEXT_WEIGHTS["overlap_consistency_loss"] * overlap


def evaluate_ticket37_response(
    response: np.ndarray,
    rendered_truth: Mapping[str, object],
    *,
    normal_grid_leaks: int = 0,
    normal_grid_leak_rate: float = 0.0,
    asserted_grid_occupancy_p95: float = 0.0,
    remote_response: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate one stored class response; remote extent is diagnostic only."""

    matching = match_proposals_to_rendered_truth(
        response,
        rendered_truth,
        threshold=0.5,
        tolerance_pixels=TOLERANCE_PIXELS,
    )
    if isinstance(normal_grid_leaks, bool) or not isinstance(normal_grid_leaks, int) or normal_grid_leaks < 0:
        raise ValueError("normal_grid_leaks must be a non-negative integer")
    for value, name in (
        (normal_grid_leak_rate, "normal_grid_leak_rate"),
        (asserted_grid_occupancy_p95, "asserted_grid_occupancy_p95"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"{name} must be a finite non-negative number")
    remote = dict(remote_response or {"numerator": 0, "denominator": 0, "ratio": 0.0})
    _validate_remote(remote)
    row = {
        "proposal_precision": float(matching["proposal_precision"]),
        "proposal_recall": float(matching["proposal_recall"]),
        "unique_truth_touch_recall": float(matching["unique_truth_touch_recall"]),
        "unmatched_truth_component_ids": list(matching["unmatched_truth_component_ids"]),
        "unmatched_proposal_ids": list(matching["unmatched_proposal_ids"]),
        "cross_component_merge_proposal_ids": list(matching["cross_component_merge_proposal_ids"]),
        "normal_grid_leaks": normal_grid_leaks,
        "normal_grid_leak_rate": float(normal_grid_leak_rate),
        "asserted_grid_occupancy_p95": float(asserted_grid_occupancy_p95),
        "remote_response": remote,
    }
    row["overall"] = "PASS" if _row_passes(row) else "FAIL"
    return row, matching


def validate_ticket37_confidence_map(value: object) -> np.ndarray:
    """Require model scoring output to already be float32 probability maps."""

    if not isinstance(value, np.ndarray):
        raise ValueError("Ticket 37 confidence map must be a numpy array")
    if (
        value.dtype != np.float32
        or value.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, len(CLASS_ORDER))
        or not np.isfinite(value).all()
        or np.any(value < 0.0)
        or np.any(value > 1.0)
    ):
        raise ValueError("Ticket 37 confidence map must be finite float32 1536x1536x2 in [0,1]")
    return value


def evaluate_ticket37_stored_maps(
    maps: Mapping[str, np.ndarray],
    cases: Sequence[DevelopmentCase] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compute all gate rows strictly from re-read float32 maps."""

    if not isinstance(maps, Mapping) or tuple(maps) != CASE_IDS:
        raise ValueError("confidence map keys/order must match Ticket 37 case order")
    selected = tuple(build_ticket36_cases() if cases is None else cases)
    if tuple(case.case_id for case in selected) != CASE_IDS:
        raise ValueError("Ticket 37 map cases must match frozen case order")
    rows: list[dict[str, object]] = []
    matching: dict[str, object] = {}
    grids = annotation_grids(IMAGE_WIDTH, IMAGE_HEIGHT, 512, 512)
    for case in selected:
        confidence_map = np.asarray(maps[case.case_id])
        if confidence_map.dtype != np.float32 or confidence_map.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, len(CLASS_ORDER)):
            raise ValueError("Ticket 37 confidence maps must be finite float32 1536x1536x2 arrays")
        if (
            not np.isfinite(confidence_map).all()
            or np.any(confidence_map < 0.0)
            or np.any(confidence_map > 1.0)
        ):
            raise ValueError("Ticket 37 confidence maps must be finite probabilities")
        class_code = case.composition
        class_index = CLASS_ORDER.index(class_code)
        rendered = build_rendered_truth_components(
            tuple(item for item in case.instances if item.defect.class_code == class_code),
            (IMAGE_HEIGHT, IMAGE_WIDTH),
        )
        response = confidence_map[..., class_index]
        evidence = compute_wafer_quality_evidence(
            (WaferEvidenceCase(case.filename, case.split, case.oracle, grids, confidence_map),),
            CLASS_ORDER,
            {code: 0.5 for code in CLASS_ORDER},
        )["per_class"][class_code]
        allowed = _global_tolerance_union(case, class_code)
        active = response >= 0.5
        remote_numerator = int((active & ~allowed).sum())
        remote_denominator = int(active.sum())
        row, case_matching = evaluate_ticket37_response(
            response,
            rendered,
            normal_grid_leaks=int(evidence["normal_grid_leaks"]),
            normal_grid_leak_rate=float(evidence["normal_grid_leak_rate"] or 0.0),
            asserted_grid_occupancy_p95=float(evidence["asserted_grid_occupancy_p95"] or 0.0),
            remote_response={
                "numerator": remote_numerator,
                "denominator": remote_denominator,
                "ratio": remote_numerator / remote_denominator if remote_denominator else 0.0,
            },
        )
        row = {"case_id": case.case_id, "class_code": class_code, **row}
        rows.append(row)
        matching[case.case_id] = {class_code: case_matching}
    return rows, {"schema": MATCHING_SCHEMA, "case_order": list(CASE_IDS), "per_case": matching}


def seal_ticket37_artifacts(
    candidate_root: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Hash the four generated role files without permitting pending values."""

    root = Path(candidate_root).expanduser().resolve()
    base = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root).expanduser().resolve()
    try:
        relative_root = root.relative_to(base).as_posix()
    except ValueError as error:
        raise ValueError("Ticket 37 candidate root must be inside repo_root") from error
    if relative_root != CANDIDATE_ARTIFACT_ROOT:
        raise ValueError("Ticket 37 candidate root drift")
    roles: list[dict[str, object]] = []
    for role in ARTIFACT_ROLES:
        path = root / ARTIFACT_FILENAMES[role]
        if not path.is_file():
            raise ValueError(f"Ticket 37 artifact is missing: {role}")
        payload = path.read_bytes()
        roles.append(
            {
                "role": role,
                "path": f"{CANDIDATE_ARTIFACT_ROOT}/{ARTIFACT_FILENAMES[role]}",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "root": CANDIDATE_ARTIFACT_ROOT,
        "hash_policy": HASH_POLICY,
        "seal_status": "sealed_sha256",
        "roles": roles,
    }


def verify_ticket37_artifact_seal(
    seal: Mapping[str, object],
    *,
    repo_root: Path | None = None,
) -> None:
    """Re-read every role, verify bytes/SHA, and validate generated contents."""

    base = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root).expanduser().resolve()
    _verify_ticket37_artifact_bytes(seal, base)
    _validate_checkpoint_artifact(base / f"{CANDIDATE_ARTIFACT_ROOT}/{ARTIFACT_FILENAMES['checkpoint']}")
    _validate_maps_artifact(base / f"{CANDIDATE_ARTIFACT_ROOT}/{ARTIFACT_FILENAMES['confidence_maps']}")
    _validate_matching_artifact(base / f"{CANDIDATE_ARTIFACT_ROOT}/{ARTIFACT_FILENAMES['proposal_matching']}")
    _validate_configuration_artifact(base / f"{CANDIDATE_ARTIFACT_ROOT}/{ARTIFACT_FILENAMES['configuration']}")


def _verify_ticket37_artifact_bytes(seal: Mapping[str, object], base: Path) -> None:
    """Verify only role bytes/SHA for a focused seal fixture."""

    roles = _validate_seal_shape(seal)
    for role, expected_path, expected_bytes, expected_sha in roles:
        path = base / expected_path
        if not path.is_file():
            raise ValueError(f"Ticket 37 sealed artifact is missing: {role}")
        payload = path.read_bytes()
        if len(payload) != expected_bytes or hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ValueError(f"Ticket 37 sealed artifact bytes/SHA mismatch: {role}")


def validate_ticket37_maps_artifact(path: Path) -> None:
    """Validate one stored confidence-map artifact independently."""

    _validate_maps_artifact(Path(path).expanduser().resolve())


def validate_ticket37_checkpoint_artifact(path: Path) -> None:
    """Validate one stored candidate checkpoint independently."""

    _validate_checkpoint_artifact(Path(path).expanduser().resolve())


def preflight_ticket37_sparse_objective_candidate(
    *,
    repo_root: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, object]:
    """Validate tracked contract, outputs, worktree, then CUDA availability."""

    root = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root).expanduser().resolve()
    report = _resolve_fixed_report_path(root, report_path)
    contract_path = root / "docs/demo/ticket37-sparse-objective-contract.json"
    try:
        tracked_text = contract_path.read_text(encoding="utf-8")
        contract = json.loads(tracked_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Ticket 37 tracked contract is unreadable: {error}") from error
    if tracked_text != canonical_ticket37_sparse_objective_contract_json(contract) + "\n":
        raise ValueError("Ticket 37 tracked contract is not canonical")
    validate_ticket37_sparse_objective_contract(contract, repo_root=root)
    _validate_runtime_contract(contract)
    imagenet_weights = _validate_cached_imagenet_weights(contract["recipe"])
    deterministic_policy = _establish_deterministic_policy(contract["deterministic_policy"])
    _validate_output_targets(root, report)
    head, status = _git_state(root)
    if status:
        raise RuntimeError("Ticket 37 preflight requires a clean committed git worktree")
    if not torch.cuda.is_available():
        raise RuntimeError("Ticket 37 runner requires CUDA")
    return {
        "contract": contract,
        "git_head": head,
        "repo_root": root,
        "report_path": report,
        "imagenet_weights": imagenet_weights,
        "deterministic_policy": deterministic_policy,
    }


def _validate_runtime_contract(contract: Mapping[str, object]) -> None:
    """Ensure runner literals and live builders still implement the contract."""

    membership = contract.get("membership")
    expected_membership = {
        "source": "ticket36_single_image_cnn_micro_overfit",
        "seed": 101,
        "split": "train",
        "case_specs": [list(spec) for spec in RUNTIME_CASE_SPECS],
        "case_ids": list(RUNTIME_CASE_IDS),
    }
    if membership != expected_membership or tuple(CASE_IDS) != RUNTIME_CASE_IDS or tuple(CASE_SPECS) != RUNTIME_CASE_SPECS:
        raise ValueError("Ticket 37 runtime case contract drift")
    cases = build_ticket36_cases()
    if tuple(case.case_id for case in cases) != RUNTIME_CASE_IDS or any(
        case.seed != 101 or case.split != "train" for case in cases
    ):
        raise ValueError("Ticket 37 runtime case builder drift")
    expected_recipe = {
        "architecture": "resnet18_spatial_logits_v5",
        "weights_policy": "imagenet",
        "weights_id": WEIGHTS_ID,
        "weights_url": WEIGHTS_URL,
        "weights_sha256": WEIGHTS_SHA256,
        "batch_norm_policy": BATCH_NORM_POLICY,
        "epochs": EPOCHS,
        "optimizer": "adamw",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "patch_size": PATCH_SIZE,
        "patch_stride": PATCH_STRIDE,
        "feature_stride": FEATURE_STRIDE,
        "threshold": 0.5,
        "tolerance_pixels": TOLERANCE_PIXELS,
        "calibration": "forbidden",
    }
    if contract.get("recipe") != expected_recipe:
        raise ValueError("Ticket 37 runtime recipe contract drift")
    if contract.get("deterministic_policy") != DETERMINISTIC_POLICY:
        raise ValueError("Ticket 37 deterministic policy contract drift")
    expected_objective = {
        "loss": "sparse_instance_localization_loss",
        "terms": ["per_instance_coverage", "all_row_hardest_tail_far", "local_full_far"],
        "hardest_fraction": HARDEST_FRACTION,
        "context_weights": dict(CONTEXT_WEIGHTS),
        "forbidden_terms": ["positive_spatial_topk_loss", "present_sparse_budget_loss"],
    }
    if contract.get("objective") != expected_objective:
        raise ValueError("Ticket 37 runtime objective contract drift")
    expected_evaluation = {
        "method": "rendered_truth_components",
        "gates": {
            "proposal_precision_min": 1.0,
            "proposal_recall_min": 1.0,
            "unique_truth_touch_recall_min": 1.0,
            "unmatched_truth_max": 0,
            "unmatched_proposal_max": 0,
            "cross_component_merge_max": 0,
            "normal_grid_leaks_max": 0,
            "asserted_grid_occupancy_p95_max": 0.25,
        },
        "remote_response": {
            "gating": "non_gating_diagnostic",
            "claim": "sparse_tolerance_not_segmentation_mask",
        },
    }
    if contract.get("evaluation") != expected_evaluation:
        raise ValueError("Ticket 37 runtime evaluation contract drift")
    if ResNet18_Weights.DEFAULT is not ResNet18_Weights.IMAGENET1K_V1:
        raise RuntimeError("Ticket 37 ImageNet default weight enum drift")


def run_ticket37_sparse_objective_candidate(
    *,
    repo_root: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, object]:
    """Execute the fixed four-case CUDA candidate exactly once."""

    preflight = preflight_ticket37_sparse_objective_candidate(repo_root=repo_root, report_path=report_path)
    root = Path(preflight["repo_root"])
    report_file = Path(preflight["report_path"])
    contract = preflight["contract"]
    candidate_root = root / CANDIDATE_ARTIFACT_ROOT
    candidate_root.mkdir(parents=True)
    cases = build_ticket36_cases()
    states: dict[str, dict[str, Tensor]] = {}
    histories: dict[str, list[float]] = {}
    maps: dict[str, np.ndarray] = {}
    for case in cases:
        _reset_case_seed()
        source = render_ticket35_development_pixels(case)
        model, state, history = _train_ticket37_case(case, source)
        states[case.case_id] = state
        histories[case.case_id] = history
        maps[case.case_id] = validate_ticket37_confidence_map(_score_source_model(model, source))
        del model
        torch.cuda.empty_cache()
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "case_order": list(CASE_IDS),
        "class_codes": list(CLASS_ORDER),
        "recipe": copy.deepcopy(contract["recipe"]),
        "objective": copy.deepcopy(contract["objective"]),
        "batch_norm_policy": BATCH_NORM_POLICY,
        "state_dicts": states,
    }
    torch.save(checkpoint, candidate_root / ARTIFACT_FILENAMES["checkpoint"])
    maps_path = candidate_root / ARTIFACT_FILENAMES["confidence_maps"]
    np.savez_compressed(maps_path, **{case_id: maps[case_id] for case_id in CASE_IDS})
    with np.load(maps_path, allow_pickle=False) as loaded:
        if tuple(loaded.files) != CASE_IDS:
            raise ValueError("Ticket 37 confidence map keys drift after write")
        reread_maps = {case_id: np.asarray(loaded[case_id]) for case_id in CASE_IDS}
    rows, matching = evaluate_ticket37_stored_maps(reread_maps, cases)
    _write_json(candidate_root / ARTIFACT_FILENAMES["proposal_matching"], matching)
    configuration = _build_configuration(
        contract,
        preflight["git_head"],
        histories,
        imagenet_weights=preflight["imagenet_weights"],
        deterministic_policy=preflight["deterministic_policy"],
    )
    _write_json(candidate_root / ARTIFACT_FILENAMES["configuration"], configuration)
    seal = seal_ticket37_artifacts(candidate_root, repo_root=root)
    report = _build_report(
        contract,
        rows,
        matching,
        seal,
        deterministic_policy=preflight["deterministic_policy"],
    )
    report_file.parent.mkdir(parents=True, exist_ok=True)
    canonical_ticket37_sparse_objective_candidate_json(report, repo_root=root)
    _write_json(report_file, report)
    return report


def validate_ticket37_sparse_objective_candidate_report(
    report: Mapping[str, object],
    *,
    repo_root: Path | None = None,
) -> None:
    """Validate report gates against the sealed matching/configuration files."""

    if not isinstance(report, Mapping) or report.get("schema") != SCHEMA:
        raise ValueError("Ticket 37 candidate report schema drift")
    required = {
        "schema", "contract_sha256", "case_order", "per_case", "overall", "recommendation",
        "artifact_seal", "execution", "metrics_source", "recipe", "objective", "evaluation", "boundary", "claims",
        "deterministic_policy",
    }
    if set(report) != required:
        raise ValueError("Ticket 37 candidate report fields drift")
    contract = build_ticket37_sparse_objective_contract()
    expected_contract_sha = _contract_sha256(contract)
    if report.get("contract_sha256") != expected_contract_sha:
        raise ValueError("Ticket 37 candidate report contract SHA drift")
    if report.get("case_order") != list(CASE_IDS):
        raise ValueError("Ticket 37 candidate report case order drift")
    if report.get("recipe") != contract["recipe"] or report.get("objective") != contract["objective"]:
        raise ValueError("Ticket 37 candidate report recipe/objective drift")
    if report.get("deterministic_policy") != contract["deterministic_policy"]:
        raise ValueError("Ticket 37 candidate report deterministic policy drift")
    if report.get("evaluation") != contract["evaluation"] or report.get("boundary") != contract["boundary"] or report.get("claims") != contract["claims"]:
        raise ValueError("Ticket 37 candidate report evaluation drift")
    if report.get("metrics_source") != "sealed_float32_re_read":
        raise ValueError("Ticket 37 candidate report metrics source drift")
    execution = report.get("execution")
    if not isinstance(execution, Mapping) or execution.get("formal_run_count") != 1 or execution.get("rerun") is not False or execution.get("final_epoch_only") is not True:
        raise ValueError("Ticket 37 candidate report execution drift")
    boundary = report.get("boundary")
    if not isinstance(boundary, Mapping) or boundary.get("cam_default") != "cam_v2" or boundary.get("spatial_mil_status") != "experimental":
        raise ValueError("Ticket 37 candidate report boundary drift")
    rows = report.get("per_case")
    if not isinstance(rows, list) or tuple(row.get("case_id") for row in rows if isinstance(row, Mapping)) != CASE_IDS:
        raise ValueError("Ticket 37 candidate report rows drift")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Ticket 37 candidate report row must be an object")
        _validate_row(row)
    expected_overall = "PASS" if all(row["overall"] == "PASS" for row in rows) else "FAIL"
    if report.get("overall") != expected_overall:
        raise ValueError("Ticket 37 candidate report overall drift")
    expected_recommendation = "continue_development" if expected_overall == "PASS" else "cam_v2"
    if report.get("recommendation") != expected_recommendation:
        raise ValueError("Ticket 37 candidate report recommendation drift")
    base = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root).expanduser().resolve()
    seal = report.get("artifact_seal")
    verify_ticket37_artifact_seal(seal, repo_root=base)
    matching = _read_json_artifact(base / f"{CANDIDATE_ARTIFACT_ROOT}/{ARTIFACT_FILENAMES['proposal_matching']}")
    maps = _read_float32_maps_artifact(
        base / f"{CANDIDATE_ARTIFACT_ROOT}/{ARTIFACT_FILENAMES['confidence_maps']}"
    )
    expected_rows, expected_matching = evaluate_ticket37_stored_maps(maps)
    if list(rows) != expected_rows or matching != expected_matching:
        raise ValueError("Ticket 37 report/sealed maps or matching drift")
    configuration = _read_json_artifact(base / f"{CANDIDATE_ARTIFACT_ROOT}/{ARTIFACT_FILENAMES['configuration']}")
    if configuration.get("contract_sha256") != expected_contract_sha or configuration.get("recipe") != contract["recipe"] or configuration.get("objective") != contract["objective"] or configuration.get("metrics_source") != "sealed_float32_re_read":
        raise ValueError("Ticket 37 configuration contract drift")


def canonical_ticket37_sparse_objective_candidate_json(
    report: Mapping[str, object], *, repo_root: Path | None = None
) -> str:
    validate_ticket37_sparse_objective_candidate_report(report, repo_root=repo_root)
    return json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_ticket37_sparse_objective_candidate(report_path=args.report)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Ticket 37 sparse objective candidate failed: {error}", file=sys.stderr)
        return 1
    print(f"Ticket 37 sparse objective candidate: {report['overall']}", flush=True)
    return 0


def _train_ticket37_case(case: DevelopmentCase, source: np.ndarray) -> tuple[torch.nn.Module, dict[str, Tensor], list[float]]:
    model = create_resnet18_spatial_logits_v5(
        len(CLASS_ORDER), weights=WeightsPolicy.IMAGENET, device="cuda"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    bounds = NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 0.0, 100.0)
    config = TrainingConfig(
        "ticket37-sparse-objective-candidate",
        case.case_id,
        len(CLASS_ORDER),
        EPOCHS,
        2,
        device="cuda",
        seed=101,
        learning_rate=LEARNING_RATE,
        weights_policy="imagenet",
        patch_size=PATCH_SIZE,
        patch_stride=PATCH_STRIDE,
        training_policy="spatial_mil_v5",
    )
    grids = annotation_grids(IMAGE_WIDTH, IMAGE_HEIGHT, 512, 512)
    truth = case.oracle.grid_truth(grids)
    asserted: list[tuple[tuple[Rect, ...], Tensor]] = []
    normal: list[tuple[tuple[Rect, ...], Tensor]] = []
    for grid in grids:
        rects = enumerate_model_patch_rects(Rect(grid.x, grid.y, grid.width, grid.height), config)
        target = torch.tensor([int(code in truth[(grid.row, grid.column)]) for code in CLASS_ORDER], dtype=torch.float32)
        (asserted if bool(target.any().item()) else normal).append((rects, target))
    if not asserted:
        raise ValueError(f"Ticket 37 case has no asserted grid: {case.case_id}")
    pairs = tuple((bag, normal[index % len(normal)] if normal else None) for index, bag in enumerate(asserted))
    initial_bn = _snapshot_batch_norm(model)
    history: list[float] = []
    for epoch in range(EPOCHS):
        epoch_losses: list[float] = []
        for (asserted_rects, asserted_target), normal_bag in pairs:
            groups = [(asserted_rects, asserted_target)]
            if normal_bag is not None:
                groups.append(normal_bag)
            rects = tuple(rect for group, _target in groups for rect in group)
            inputs = torch.stack([
                extract_model_patch(source, bounds, top=rect.y, left=rect.x, size=PATCH_SIZE)
                for rect in rects
            ]).cuda(non_blocking=True)
            targets = torch.stack([target for _rects, target in groups]).cuda(non_blocking=True)
            supervision = _build_batch_supervision(
                case,
                rects,
                tuple(not bool(target.any().item()) for _rects, target in groups for _ in _rects),
            )
            set_spatial_transfer_training_mode(model)
            raw_logits = model(inputs)
            group_rects = tuple(group for group, _target in groups)
            loss = compute_ticket37_objective(
                raw_logits,
                targets,
                tuple(case.case_id for _group, _target in groups),
                supervision.instance_masks.cuda(non_blocking=True),
                supervision.instance_batch_indices.cuda(non_blocking=True),
                supervision.instance_class_indices.cuda(non_blocking=True),
                supervision.far_negative_masks.cuda(non_blocking=True),
                supervision.far_batch_indices.cuda(non_blocking=True),
                supervision.far_class_indices.cuda(non_blocking=True),
                group_rects,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        history.append(epoch_loss)
        print(f"Ticket 37 {case.case_id}: epoch {epoch + 1}/{EPOCHS} loss={epoch_loss:.6f}", flush=True)
    if _snapshot_batch_norm(model) != initial_bn:
        raise RuntimeError("Ticket 37 BatchNorm running statistics drifted")
    return model, {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}, history


def _snapshot_batch_norm(model: torch.nn.Module) -> tuple[tuple[str, bytes, bytes, bytes], ...]:
    snapshot = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            snapshot.append(
                (
                    name,
                    module.running_mean.detach().cpu().numpy().tobytes(),
                    module.running_var.detach().cpu().numpy().tobytes(),
                    module.num_batches_tracked.detach().cpu().numpy().tobytes(),
                )
            )
    return tuple(snapshot)


def _build_configuration(
    contract: Mapping[str, object],
    git_head: str,
    histories: Mapping[str, Sequence[float]],
    *,
    imagenet_weights: Mapping[str, str],
    deterministic_policy: Mapping[str, object],
) -> dict[str, object]:
    runner_path = Path(__file__).resolve()
    return {
        "schema": CONFIG_SCHEMA,
        "contract_sha256": _contract_sha256(contract),
        "source_dependencies": copy.deepcopy(contract["source_dependencies"]),
        "runner_sha256": _sha256_normalized_file(runner_path),
        "git_head": git_head,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda or "unknown",
            "cudnn": str(torch.backends.cudnn.version() or "unknown"),
            "platform": platform.platform(),
            "device": torch.cuda.get_device_name(0),
            "cublas_workspace_config": str(deterministic_policy["cublas_workspace_config"]),
            "torch_deterministic_algorithms": bool(deterministic_policy["torch_deterministic_algorithms"]),
            "cudnn_deterministic": bool(deterministic_policy["cudnn_deterministic"]),
            "cudnn_benchmark": bool(deterministic_policy["cudnn_benchmark"]),
            "imagenet_weights": copy.deepcopy(dict(imagenet_weights)),
        },
        "case_order": list(CASE_IDS),
        "class_order": list(CLASS_ORDER),
        "recipe": copy.deepcopy(contract["recipe"]),
        "objective": copy.deepcopy(contract["objective"]),
        "deterministic_policy": copy.deepcopy(dict(deterministic_policy)),
        "batch_norm_policy": BATCH_NORM_POLICY,
        "metrics_source": "sealed_float32_re_read",
        "loss_history": {case_id: list(histories[case_id]) for case_id in CASE_IDS},
        "formal_run_count": 1,
        "rerun": False,
        "final_epoch_only": True,
    }


def _build_report(
    contract: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    matching: Mapping[str, object],
    seal: Mapping[str, object],
    *,
    deterministic_policy: Mapping[str, object],
) -> dict[str, object]:
    del matching
    overall = "PASS" if all(row.get("overall") == "PASS" for row in rows) else "FAIL"
    return {
        "schema": SCHEMA,
        "contract_sha256": _contract_sha256(contract),
        "case_order": list(CASE_IDS),
        "per_case": [dict(row) for row in rows],
        "overall": overall,
        "recommendation": "continue_development" if overall == "PASS" else "cam_v2",
        "artifact_seal": dict(seal),
        "execution": {"formal_run_count": 1, "rerun": False, "final_epoch_only": True},
        "metrics_source": "sealed_float32_re_read",
        "recipe": copy.deepcopy(contract["recipe"]),
        "objective": copy.deepcopy(contract["objective"]),
        "deterministic_policy": copy.deepcopy(dict(deterministic_policy)),
        "evaluation": copy.deepcopy(contract["evaluation"]),
        "boundary": copy.deepcopy(contract["boundary"]),
        "claims": copy.deepcopy(contract["claims"]),
    }


def _row_passes(row: Mapping[str, object]) -> bool:
    return (
        float(row["proposal_precision"]) >= 1.0
        and float(row["proposal_recall"]) >= 1.0
        and float(row["unique_truth_touch_recall"]) >= 1.0
        and not row["unmatched_truth_component_ids"]
        and not row["unmatched_proposal_ids"]
        and not row["cross_component_merge_proposal_ids"]
        and int(row["normal_grid_leaks"]) == 0
        and float(row["asserted_grid_occupancy_p95"]) <= 0.25
    )


def _validate_row(row: Mapping[str, object]) -> None:
    required = {
        "case_id", "class_code", "proposal_precision", "proposal_recall", "unique_truth_touch_recall",
        "unmatched_truth_component_ids", "unmatched_proposal_ids", "cross_component_merge_proposal_ids",
        "normal_grid_leaks", "normal_grid_leak_rate", "asserted_grid_occupancy_p95", "remote_response", "overall",
    }
    if set(row) != required or row.get("case_id") not in CASE_IDS or row.get("class_code") not in CLASS_ORDER:
        raise ValueError("Ticket 37 candidate report row fields drift")
    for field in ("proposal_precision", "proposal_recall", "unique_truth_touch_recall", "normal_grid_leak_rate", "asserted_grid_occupancy_p95"):
        value = row[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"Ticket 37 candidate report row {field} invalid")
    if isinstance(row["normal_grid_leaks"], bool) or not isinstance(row["normal_grid_leaks"], int) or row["normal_grid_leaks"] < 0:
        raise ValueError("Ticket 37 normal_grid_leaks invalid")
    for field in ("unmatched_truth_component_ids", "unmatched_proposal_ids", "cross_component_merge_proposal_ids"):
        if not isinstance(row[field], list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in row[field]):
            raise ValueError("Ticket 37 candidate report IDs invalid")
    if row["overall"] not in {"PASS", "FAIL"} or row["overall"] != ("PASS" if _row_passes(row) else "FAIL"):
        raise ValueError("Ticket 37 candidate report row gate drift")
    _validate_remote(row["remote_response"])


def _validate_remote(remote: object) -> None:
    if not isinstance(remote, Mapping) or set(remote) != {"numerator", "denominator", "ratio"}:
        raise ValueError("Ticket 37 remote response diagnostic drift")
    numerator, denominator, ratio = remote["numerator"], remote["denominator"], remote["ratio"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or numerator < 0
        or denominator < 0
        or numerator > denominator
        or isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(float(ratio))
        or not 0 <= float(ratio) <= 1
        or float(ratio) != (numerator / denominator if denominator else 0.0)
    ):
        raise ValueError("Ticket 37 remote response diagnostic invalid")


def _validate_seal_shape(seal: Mapping[str, object]) -> tuple[tuple[str, str, int, str], ...]:
    if not isinstance(seal, Mapping) or set(seal) != {"root", "hash_policy", "seal_status", "roles"}:
        raise ValueError("Ticket 37 artifact seal fields drift")
    if seal.get("root") != CANDIDATE_ARTIFACT_ROOT or seal.get("hash_policy") != HASH_POLICY or seal.get("seal_status") != "sealed_sha256":
        raise ValueError("Ticket 37 artifact seal contract drift")
    roles = seal.get("roles")
    if not isinstance(roles, list) or len(roles) != len(ARTIFACT_ROLES):
        raise ValueError("Ticket 37 artifact seal roles drift")
    result = []
    seen_paths: set[str] = set()
    for index, item in enumerate(roles):
        if not isinstance(item, Mapping) or set(item) != {"role", "path", "bytes", "sha256"}:
            raise ValueError("Ticket 37 artifact seal role fields drift")
        role = item["role"]
        path = item["path"]
        if role != ARTIFACT_ROLES[index] or not isinstance(path, str) or path in seen_paths or path != f"{CANDIDATE_ARTIFACT_ROOT}/{ARTIFACT_FILENAMES[role]}":
            raise ValueError("Ticket 37 artifact seal role/path drift")
        seen_paths.add(path)
        if isinstance(item["bytes"], bool) or not isinstance(item["bytes"], int) or item["bytes"] <= 0:
            raise ValueError("Ticket 37 artifact seal bytes drift")
        if not isinstance(item["sha256"], str) or len(item["sha256"]) != 64 or item["sha256"] != item["sha256"].lower() or any(char not in "0123456789abcdef" for char in item["sha256"]):
            raise ValueError("Ticket 37 artifact seal sha256 drift")
        result.append((str(role), path, int(item["bytes"]), item["sha256"]))
    return tuple(result)


def _validate_checkpoint_artifact(path: Path) -> None:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("Ticket 37 checkpoint cannot be read safely") from error
    if not isinstance(checkpoint, Mapping) or checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("case_order") != list(CASE_IDS) or checkpoint.get("class_codes") != list(CLASS_ORDER) or checkpoint.get("batch_norm_policy") != BATCH_NORM_POLICY:
        raise ValueError("Ticket 37 checkpoint contract drift")
    contract = build_ticket37_sparse_objective_contract()
    if checkpoint.get("recipe") != contract["recipe"] or checkpoint.get("objective") != contract["objective"]:
        raise ValueError("Ticket 37 checkpoint recipe/objective drift")
    state_dicts = checkpoint.get("state_dicts")
    if not isinstance(state_dicts, Mapping) or tuple(state_dicts) != CASE_IDS:
        raise ValueError("Ticket 37 checkpoint state order drift")
    for state in state_dicts.values():
        if not isinstance(state, Mapping):
            raise ValueError("Ticket 37 checkpoint state must be a mapping")
        model = create_resnet18_spatial_logits_v5(len(CLASS_ORDER), weights=WeightsPolicy.NONE, device="cpu")
        expected_state = model.state_dict()
        if set(state) != set(expected_state):
            raise ValueError("Ticket 37 checkpoint state keys drift")
        for name, expected in expected_state.items():
            value = state[name]
            if not isinstance(value, Tensor) or value.device.type != "cpu" or value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"Ticket 37 checkpoint state tensor drift: {name}")
            if torch.is_floating_point(value) and not torch.isfinite(value).all():
                raise ValueError("Ticket 37 checkpoint state contains non-finite tensor")
        try:
            model.load_state_dict(state, strict=True)
        except (RuntimeError, TypeError) as error:
            raise ValueError("Ticket 37 checkpoint strict v5 load failed") from error


def _validate_maps_artifact(path: Path) -> None:
    _read_float32_maps_artifact(path)


def _read_float32_maps_artifact(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if tuple(loaded.files) != CASE_IDS:
                raise ValueError("Ticket 37 sealed map keys drift")
            maps = {}
            for case_id in CASE_IDS:
                value = loaded[case_id]
                if (
                    value.dtype != np.float32
                    or value.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, len(CLASS_ORDER))
                    or not np.isfinite(value).all()
                    or np.any(value < 0.0)
                    or np.any(value > 1.0)
                ):
                    raise ValueError("Ticket 37 sealed maps dtype/shape/finite drift")
                maps[case_id] = np.asarray(value)
            return maps
    except (OSError, ValueError) as error:
        raise ValueError("Ticket 37 sealed confidence maps invalid") from error


def _validate_matching_artifact(path: Path) -> None:
    value = _read_json_artifact(path)
    if value.get("schema") != MATCHING_SCHEMA or value.get("case_order") != list(CASE_IDS) or not isinstance(value.get("per_case"), Mapping) or tuple(value["per_case"]) != CASE_IDS:
        raise ValueError("Ticket 37 matching artifact contract drift")
    for case_id in CASE_IDS:
        item = value["per_case"].get(case_id)
        if not isinstance(item, Mapping) or len(item) != 1:
            raise ValueError("Ticket 37 matching artifact row drift")
        class_code, matching = next(iter(item.items()))
        if class_code not in CLASS_ORDER or not isinstance(matching, Mapping):
            raise ValueError("Ticket 37 matching artifact class drift")
        for field in ("proposal_precision", "proposal_recall", "unique_truth_touch_recall"):
            metric = matching.get(field)
            if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
                raise ValueError("Ticket 37 matching artifact metric drift")


def _validate_configuration_artifact(path: Path) -> None:
    value = _read_json_artifact(path)
    contract = build_ticket37_sparse_objective_contract()
    required = {"schema", "contract_sha256", "source_dependencies", "runner_sha256", "git_head", "environment", "case_order", "class_order", "recipe", "objective", "deterministic_policy", "batch_norm_policy", "metrics_source", "loss_history", "formal_run_count", "rerun", "final_epoch_only"}
    if set(value) != required or value.get("schema") != CONFIG_SCHEMA or value.get("contract_sha256") != _contract_sha256(contract) or value.get("source_dependencies") != contract["source_dependencies"] or value.get("case_order") != list(CASE_IDS) or value.get("class_order") != list(CLASS_ORDER) or value.get("recipe") != contract["recipe"] or value.get("objective") != contract["objective"] or value.get("deterministic_policy") != contract["deterministic_policy"] or value.get("batch_norm_policy") != BATCH_NORM_POLICY or value.get("metrics_source") != "sealed_float32_re_read" or value.get("formal_run_count") != 1 or value.get("rerun") is not False or value.get("final_epoch_only") is not True:
        raise ValueError("Ticket 37 configuration contract drift")
    environment = value.get("environment")
    if (
        not isinstance(environment, Mapping)
        or set(environment) != {
            "python", "torch", "torchvision", "cuda", "cudnn", "platform", "device",
            "cublas_workspace_config", "torch_deterministic_algorithms",
            "cudnn_deterministic", "cudnn_benchmark", "imagenet_weights",
        }
        or any(
            not isinstance(environment[key], str) or not environment[key]
            for key in ("python", "torch", "torchvision", "cuda", "cudnn", "platform", "device")
        )
        or not isinstance(environment["cublas_workspace_config"], str)
        or not isinstance(environment["torch_deterministic_algorithms"], bool)
        or not isinstance(environment["cudnn_deterministic"], bool)
        or not isinstance(environment["cudnn_benchmark"], bool)
        or environment["cublas_workspace_config"] != DETERMINISTIC_POLICY["cublas_workspace_config"]
        or environment["torch_deterministic_algorithms"] is not DETERMINISTIC_POLICY["torch_deterministic_algorithms"]
        or environment["cudnn_deterministic"] is not DETERMINISTIC_POLICY["cudnn_deterministic"]
        or environment["cudnn_benchmark"] is not DETERMINISTIC_POLICY["cudnn_benchmark"]
    ):
        raise ValueError("Ticket 37 configuration environment drift")
    if value.get("deterministic_policy") != DETERMINISTIC_POLICY:
        raise ValueError("Ticket 37 configuration deterministic policy drift")
    weights = environment.get("imagenet_weights")
    if (
        not isinstance(weights, Mapping)
        or set(weights) != {"enum", "url", "path", "sha256"}
        or weights.get("enum") != WEIGHTS_ID
        or weights.get("url") != WEIGHTS_URL
        or not isinstance(weights.get("path"), str)
        or not weights["path"]
        or not isinstance(weights.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", weights["sha256"]) is None
        or weights["sha256"] != WEIGHTS_SHA256
        or weights["sha256"] != _sha256_file(Path(weights["path"]))
    ):
        raise ValueError("Ticket 37 ImageNet weight metadata drift")
    git_head = value.get("git_head")
    if not isinstance(git_head, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", git_head) is None:
        raise ValueError("Ticket 37 configuration git HEAD drift")
    runner_expected = next(
        sha256
        for path, sha256 in SOURCE_DEPENDENCIES
        if path == "docs/demo/ticket37_sparse_objective_candidate.py"
    )
    if not isinstance(value.get("runner_sha256"), str) or value["runner_sha256"] != runner_expected or value["runner_sha256"] != _sha256_normalized_file(Path(__file__)):
        raise ValueError("Ticket 37 runner SHA drift")
    history = value.get("loss_history")
    if not isinstance(history, Mapping) or tuple(history) != CASE_IDS:
        raise ValueError("Ticket 37 loss history case order drift")
    for case_id in CASE_IDS:
        values = history[case_id]
        if not isinstance(values, list) or len(values) != EPOCHS or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item))
            for item in values
        ):
            raise ValueError(f"Ticket 37 loss history drift: {case_id}")


def _read_json_artifact(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Ticket 37 JSON artifact invalid: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Ticket 37 JSON artifact must be an object: {path.name}")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    """Write one canonical JSON artifact without permitting replacement."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
    except FileExistsError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"Ticket 37 JSON artifact write failed: {path}") from error


def _coerce_rect(value: Rect | tuple[int, int, int, int]) -> Rect:
    if isinstance(value, Rect):
        return value
    if isinstance(value, tuple) and len(value) == 4 and all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return Rect(*value)
    raise ValueError("patch_rect_groups must contain Rect values or integer tuples")


def _git_state(root: Path) -> tuple[str, str]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Ticket 37 git preflight failed: {error}") from error
    return head, status


def _resolve_fixed_report_path(root: Path, report_path: Path | None) -> Path:
    expected = (root / REPORT_PATH).resolve()
    candidate = expected if report_path is None else Path(report_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.expanduser().resolve()
    if candidate != expected:
        raise ValueError("Ticket 37 report path is fixed")
    return candidate


def _validate_output_targets(root: Path, report: Path) -> None:
    if (root / CANDIDATE_ARTIFACT_ROOT).exists():
        raise FileExistsError("Ticket 37 candidate artifact root already exists")
    if report.exists():
        raise FileExistsError("Ticket 37 candidate report already exists")


def _contract_sha256(contract: Mapping[str, object]) -> str:
    return hashlib.sha256((canonical_ticket37_sparse_objective_contract_json(contract) + "\n").encode("utf-8")).hexdigest()


def _sha256_normalized_file(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"Ticket 37 ImageNet weight file is unreadable: {path}") from error


def _validate_cached_imagenet_weights(recipe: Mapping[str, object]) -> dict[str, str]:
    """Validate the frozen ImageNet cache before creating candidate artifacts."""

    if ResNet18_Weights.DEFAULT is not ResNet18_Weights.IMAGENET1K_V1:
        raise RuntimeError("Ticket 37 ImageNet default weight enum drift")
    if (
        not isinstance(recipe, Mapping)
        or recipe.get("weights_id") != WEIGHTS_ID
        or recipe.get("weights_url") != WEIGHTS_URL
        or recipe.get("weights_sha256") != WEIGHTS_SHA256
    ):
        raise ValueError("Ticket 37 ImageNet weight contract drift")
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(WEIGHTS_URL).name
    if not path.is_file():
        raise RuntimeError(f"Ticket 37 cached ImageNet weight file is missing: {path}")
    digest = _sha256_file(path)
    if digest != WEIGHTS_SHA256:
        raise RuntimeError("Ticket 37 cached ImageNet weight SHA-256 drift")
    return {
        "enum": WEIGHTS_ID,
        "url": WEIGHTS_URL,
        "path": str(path.resolve()),
        "sha256": digest,
    }


def _establish_deterministic_policy(policy: Mapping[str, object]) -> dict[str, object]:
    """Set and verify deterministic execution before CUDA or output creation."""

    if dict(policy) != DETERMINISTIC_POLICY:
        raise ValueError("Ticket 37 deterministic policy contract drift")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_POLICY["cublas_workspace_config"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") != DETERMINISTIC_POLICY["cublas_workspace_config"]
        or not torch.are_deterministic_algorithms_enabled()
        or torch.backends.cudnn.deterministic is not True
        or torch.backends.cudnn.benchmark is not False
    ):
        raise RuntimeError("Ticket 37 deterministic policy could not be established")
    return dict(DETERMINISTIC_POLICY)


__all__ = [
    "BATCH_NORM_POLICY",
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "DETERMINISTIC_POLICY",
    "MATCHING_SCHEMA",
    "REPORT_PATH",
    "SCHEMA",
    "WEIGHTS_SHA256",
    "WEIGHTS_URL",
    "canonical_ticket37_sparse_objective_candidate_json",
    "compute_ticket37_objective",
    "evaluate_ticket37_response",
    "evaluate_ticket37_stored_maps",
    "main",
    "preflight_ticket37_sparse_objective_candidate",
    "run_ticket37_sparse_objective_candidate",
    "seal_ticket37_artifacts",
    "validate_ticket37_sparse_objective_candidate_report",
    "validate_ticket37_checkpoint_artifact",
    "validate_ticket37_confidence_map",
    "validate_ticket37_maps_artifact",
    "verify_ticket37_artifact_seal",
]


if __name__ == "__main__":
    raise SystemExit(main())
