"""Ticket 36.06 fixed single-image CNN micro-overfit.

The runner is deliberately a candidate-only experiment.  Each frozen
seed-101/train density case gets a fresh ImageNet ResNet18-v5 instance and
its own 30-epoch run.  Sparse point/scribble truth is rasterized in source
coordinates, pooled from 128 to 64 with 2x2 any pooling, and passed to the
existing loss seams.  The old Ticket 35 ring loss is intentionally absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from math import ceil, hypot, isfinite
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PySide6.QtGui import QImage
from torch import Tensor

from docs.demo.defect_oracle import Particle, Scratch
from docs.demo.ticket35_development_corpus import (
    RESERVED_FINAL_MEMBER_IDS,
    DevelopmentCase,
    build_ticket35_development_corpus,
    render_ticket35_development_pixels,
)
from docs.demo.ticket36_instance_matching import match_ticket36_instance_peaks
from docs.demo.ticket36_localization_contract import (
    ARTIFACT_FILENAMES,
    ARTIFACT_ROLES,
    CANDIDATE_ARTIFACT_ROOT,
    CLASS_ORDER,
    FEATURE_STRIDE,
    HASH_POLICY,
    PARTICLE_TOLERANCE_RADIUS,
    SCHEMA as CONTRACT_SCHEMA,
    SCRATCH_TOLERANCE_HALF_WIDTH,
    SPARSE_BUNDLE_SHA256,
    SPARSE_MANIFEST_PATH,
    build_ticket36_localization_contract,
    canonical_ticket36_localization_contract_json,
    validate_ticket36_candidate_artifact_seal,
    validate_ticket36_localization_contract,
)
from docs.demo.wafer_quality_evidence import WaferEvidenceCase, compute_wafer_quality_evidence
from wafer_defect_studio.cam_detection import generate_all_convolutional_artifact
from wafer_defect_studio.detection_windows import Rect, enumerate_inference_windows
from wafer_defect_studio.grid_geometry import annotation_grids
from wafer_defect_studio.model_registry import (
    create_resnet18_spatial_logits_v5,
    spatial_logits_to_probabilities,
)
from wafer_defect_studio.normalization import NormalizationBounds
from wafer_defect_studio.spatial_mil import (
    dense_absent_class_loss,
    far_negative_suppression_loss,
    overlap_consistency_loss,
    per_instance_coverage_loss,
    positive_spatial_topk_loss,
    present_sparse_budget_loss,
    same_image_grid_ranking_loss,
)
from wafer_defect_studio.training_dataset import enumerate_model_patch_rects, extract_model_patch
from wafer_defect_studio.training_input_bundle import (
    TrainingBundleSource,
    TrainingInputBundle,
    TrainingPatchBag,
)
from wafer_defect_studio.training_protocol import TrainingConfig


SCHEMA = "ticket36-single-image-cnn-micro-overfit.v1"
CONFIG_SCHEMA = "ticket36-single-image-cnn-micro-overfit.configuration.v1"
MATCHING_SCHEMA = "ticket36-single-image-cnn-micro-overfit.matching.v1"
CHECKPOINT_SCHEMA = "ticket36-single-image-cnn-micro-overfit.checkpoint.v1"
CLASS_CODES = CLASS_ORDER
IMAGE_WIDTH = 1536
IMAGE_HEIGHT = 1536
FEATURE_STRIDE = 2
THRESHOLD = 0.5
MICRO_EPOCHS = 30
PATCH_SIZE = 128
PATCH_STRIDE = 64
WEIGHTS_POLICY = "imagenet"
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP_NORM = 5.0
FAR_NEGATIVE_HARDEST_FRACTION = 0.01
TOLERANCE_SOURCE_PIXELS = 8
CASE_SPECS = (
    ("scratch", 1, 0),
    ("scratch", 150, 0),
    ("particle", 0, 1),
    ("particle", 0, 150),
)
CASE_IDS = (
    "ticket35-density-101-train-scratch-001-000-01",
    "ticket35-density-101-train-scratch-150-000-05",
    "ticket35-density-101-train-particle-000-001-06",
    "ticket35-density-101-train-particle-000-150-10",
)
ARTIFACT_ROOT = CANDIDATE_ARTIFACT_ROOT
ARTIFACT_DIRNAME = Path(ARTIFACT_ROOT).name
ARTIFACT_PATHS = {
    role: Path(ARTIFACT_ROOT) / ARTIFACT_FILENAMES[role] for role in ARTIFACT_ROLES
}
LOSS_WEIGHTS = {
    "coverage": 1.0,
    "far_negative": 1.0,
    "positive": 0.25,
    "absent": 0.5,
    "ranking": 0.25,
    "sparse": 0.10,
    "overlap": 0.10,
}
POST_RUN_CODE_REPAIR = "normal_grid_full_far"
SEALED_METRICS_SOURCE = "sealed_float16_recompute"
PREFLIGHT_ERRORS = (
    "TrainingConfig rejected unsupported training_policy='spatial_mil_v7'; corrected to 'spatial_mil_v5' before the successful formal run",
)


@dataclass(frozen=True)
class PatchSupervision:
    """CPU source/feature masks for one source-coordinate Model Patch."""

    source_core_masks: Tensor
    feature_core_masks: Tensor
    source_tolerance_union: Tensor
    feature_tolerance_union: Tensor
    far_negative_masks: Tensor
    instance_class_indices: Tensor
    instance_ids: tuple[str, ...]


@dataclass(frozen=True)
class BatchSupervision:
    """Flattened supervision for a batch of same-shape Model Patches."""

    instance_masks: Tensor
    instance_batch_indices: Tensor
    instance_class_indices: Tensor
    instance_ids: tuple[str, ...]
    far_negative_masks: Tensor
    far_batch_indices: Tensor
    far_class_indices: Tensor


def build_ticket36_cases(
    corpus: Sequence[DevelopmentCase] | None = None,
) -> tuple[DevelopmentCase, ...]:
    """Return exactly the four frozen Ticket 35 seed-101/train cases."""

    available = tuple(build_ticket35_development_corpus() if corpus is None else corpus)
    selected = tuple(
        case
        for case in available
        if case.seed == 101
        and case.split == "train"
        and (case.composition, case.scratch_count, case.particle_count) in CASE_SPECS
    )
    selected = tuple(sorted(selected, key=lambda case: CASE_IDS.index(case.case_id)))
    if tuple(case.case_id for case in selected) != CASE_IDS:
        raise ValueError("Ticket 36.06 frozen case membership drift")
    for case in selected:
        names = (case.case_id, case.filename, *(item.instance_id for item in case.instances))
        if any(
            any(token in value for token in ("ticket30-evidence-", *RESERVED_FINAL_MEMBER_IDS))
            for value in names
        ):
            raise ValueError("Ticket 36.06 contains forbidden final member")
    return selected


def build_patch_supervision(
    case: DevelopmentCase,
    patch_rects: Sequence[Rect | tuple[int, int, int, int]],
    *,
    is_normal_grid: bool,
    feature_stride: int = FEATURE_STRIDE,
) -> PatchSupervision:
    """Build source masks, pooled core masks, and per-class far masks.

    This public seam intentionally accepts one patch.  Batch assembly is an
    internal concern so geometry can be inspected without model execution.
    """

    rects = tuple(_coerce_rect(value) for value in patch_rects)
    if len(rects) != 1:
        raise ValueError("build_patch_supervision expects exactly one Model Patch")
    if not isinstance(is_normal_grid, bool):
        raise ValueError("is_normal_grid must be boolean and caller-declared")
    if feature_stride != FEATURE_STRIDE:
        raise ValueError("Ticket 36.06 feature_stride is frozen at 2")
    rect = rects[0]
    if (rect.width, rect.height) != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError("Ticket 36.06 Model Patch geometry is frozen at 128x128")
    source_core: list[np.ndarray] = []
    class_indices: list[int] = []
    instance_ids: list[str] = []
    for item in case.instances:
        mask = _rasterize_core(item.defect, rect)
        if not mask.any():
            continue
        source_core.append(mask)
        class_indices.append(CLASS_CODES.index(item.defect.class_code))
        instance_ids.append(item.instance_id)
    source_tolerance = np.zeros((len(CLASS_CODES), rect.height, rect.width), dtype=np.bool_)
    if not is_normal_grid:
        for class_index, class_code in enumerate(CLASS_CODES):
            for item in case.instances:
                if item.defect.class_code == class_code:
                    source_tolerance[class_index] |= _rasterize_tolerance(item.defect, rect)
    source_core_tensor = (
        torch.from_numpy(np.stack(source_core)).to(dtype=torch.bool)
        if source_core
        else torch.zeros((0, rect.height, rect.width), dtype=torch.bool)
    )
    source_tolerance_tensor = torch.from_numpy(source_tolerance)
    feature_core_tensor = _pool_source_mask(source_core_tensor, feature_stride)
    feature_tolerance_tensor = _pool_source_mask(source_tolerance_tensor, feature_stride)
    return PatchSupervision(
        source_core_masks=source_core_tensor,
        feature_core_masks=feature_core_tensor,
        source_tolerance_union=source_tolerance_tensor,
        feature_tolerance_union=feature_tolerance_tensor,
        far_negative_masks=~feature_tolerance_tensor,
        instance_class_indices=torch.tensor(class_indices, dtype=torch.long),
        instance_ids=tuple(instance_ids),
    )


def _build_batch_supervision(
    case: DevelopmentCase,
    patch_rects: Sequence[Rect],
    normal_flags: Sequence[bool],
) -> BatchSupervision:
    if len(patch_rects) != len(normal_flags):
        raise ValueError("normal_flags length must match patch_rects")
    instance_masks: list[Tensor] = []
    instance_batch: list[int] = []
    instance_class: list[int] = []
    instance_ids: list[str] = []
    far_masks: list[Tensor] = []
    far_batch: list[int] = []
    far_class: list[int] = []
    for patch_index, rect in enumerate(patch_rects):
        patch = build_patch_supervision(case, (rect,), is_normal_grid=normal_flags[patch_index])
        if patch.feature_core_masks.shape[0]:
            instance_masks.extend(patch.feature_core_masks)
            instance_batch.extend([patch_index] * patch.feature_core_masks.shape[0])
            instance_class.extend(patch.instance_class_indices.tolist())
            instance_ids.extend(patch.instance_ids)
        for class_index, mask in enumerate(patch.far_negative_masks):
            far_masks.append(mask)
            far_batch.append(patch_index)
            far_class.append(class_index)
    feature_shape = (PATCH_SIZE // FEATURE_STRIDE, PATCH_SIZE // FEATURE_STRIDE)
    return BatchSupervision(
        instance_masks=(
            torch.stack(instance_masks)
            if instance_masks
            else torch.zeros((0, *feature_shape), dtype=torch.bool)
        ),
        instance_batch_indices=torch.tensor(instance_batch, dtype=torch.long),
        instance_class_indices=torch.tensor(instance_class, dtype=torch.long),
        instance_ids=tuple(instance_ids),
        far_negative_masks=torch.stack(far_masks),
        far_batch_indices=torch.tensor(far_batch, dtype=torch.long),
        far_class_indices=torch.tensor(far_class, dtype=torch.long),
    )


def _pool_source_mask(mask: Tensor, feature_stride: int) -> Tensor:
    if mask.ndim != 3:
        raise ValueError("source mask must have shape (N,H,W)")
    if mask.shape[-2] % feature_stride or mask.shape[-1] % feature_stride:
        raise ValueError("source mask shape must be divisible by feature_stride")
    if mask.shape[0] == 0:
        return torch.zeros(
            (0, mask.shape[-2] // feature_stride, mask.shape[-1] // feature_stride),
            dtype=torch.bool,
        )
    return F.max_pool2d(
        mask.to(dtype=torch.float32).unsqueeze(1),
        kernel_size=feature_stride,
        stride=feature_stride,
    ).squeeze(1).to(dtype=torch.bool)


def _rasterize_core(defect: Scratch | Particle, rect: Rect) -> np.ndarray:
    mask = np.zeros((rect.height, rect.width), dtype=np.bool_)
    for x, y in _defect_points(defect):
        local_x, local_y = x - rect.x, y - rect.y
        if 0 <= local_x < rect.width and 0 <= local_y < rect.height:
            mask[local_y, local_x] = True
    return mask


def _rasterize_tolerance(defect: Scratch | Particle, rect: Rect) -> np.ndarray:
    radius = TOLERANCE_SOURCE_PIXELS
    mask = np.zeros((rect.height, rect.width), dtype=np.bool_)
    if isinstance(defect, Particle):
        points = (defect.center,)
    else:
        points = _defect_points(defect)
    for x, y in points:
        if not (rect.x - radius <= x < rect.right + radius and rect.y - radius <= y < rect.bottom + radius):
            continue
        local_x, local_y = x - rect.x, y - rect.y
        for delta_y in range(-radius, radius + 1):
            row = local_y + delta_y
            if not 0 <= row < rect.height:
                continue
            for delta_x in range(-radius, radius + 1):
                column = local_x + delta_x
                if 0 <= column < rect.width:
                    if isinstance(defect, Particle) and delta_x * delta_x + delta_y * delta_y > radius * radius:
                        continue
                    mask[row, column] = True
    return mask


def _defect_points(defect: Scratch | Particle) -> tuple[tuple[int, int], ...]:
    if isinstance(defect, Particle):
        return (defect.center,)
    points: list[tuple[int, int]] = []
    for start, end in zip(defect.points, defect.points[1:]):
        steps = max(abs(end[0] - start[0]), abs(end[1] - start[1])) + 1
        points.extend(
            (
                round(start[0] + index / max(1, steps - 1) * (end[0] - start[0])),
                round(start[1] + index / max(1, steps - 1) * (end[1] - start[1])),
            )
            for index in range(steps)
        )
    return tuple(dict.fromkeys(points))


def train_single_case(
    case: DevelopmentCase,
    source: np.ndarray,
    *,
    epochs: int = MICRO_EPOCHS,
) -> tuple[torch.nn.Module, list[float]]:
    """Train one case using the frozen CUDA recipe."""

    if epochs != MICRO_EPOCHS:
        raise ValueError("Ticket 36.06 epochs are frozen at 30")
    model = create_resnet18_spatial_logits_v5(
        len(CLASS_CODES), weights=WEIGHTS_POLICY, device="cuda"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    bounds = NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 0.0, 100.0)
    grids = annotation_grids(IMAGE_WIDTH, IMAGE_HEIGHT, 512, 512)
    truth = case.oracle.grid_truth(grids)
    normal_bags: list[tuple[tuple[Rect, ...], Tensor]] = []
    asserted_bags: list[tuple[tuple[Rect, ...], Tensor]] = []
    config = TrainingConfig(
        "ticket36-single-image-cnn-micro-overfit",
        case.case_id,
        len(CLASS_CODES),
        MICRO_EPOCHS,
        FEATURE_STRIDE,
        device="cuda",
        seed=101,
        learning_rate=LEARNING_RATE,
        weights_policy=WEIGHTS_POLICY,
        patch_size=PATCH_SIZE,
        patch_stride=PATCH_STRIDE,
        training_policy="spatial_mil_v5",
    )
    for grid in grids:
        grid_rect = Rect(grid.x, grid.y, grid.width, grid.height)
        rects = enumerate_model_patch_rects(grid_rect, config)
        target = torch.tensor(
            [int(code in truth[(grid.row, grid.column)]) for code in CLASS_CODES],
            dtype=torch.float32,
        )
        (asserted_bags if bool(target.any().item()) else normal_bags).append((rects, target))
    if not asserted_bags:
        raise ValueError(f"Ticket 36.06 case has no asserted Annotation Grid: {case.case_id}")
    pairs = tuple(
        (bag, normal_bags[index % len(normal_bags)] if normal_bags else None)
        for index, bag in enumerate(asserted_bags)
    )
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for (asserted_rects, asserted_target), normal in pairs:
            rect_groups = [(asserted_rects, asserted_target)]
            if normal is not None:
                rect_groups.append(normal)
            inputs = torch.stack(
                [
                    extract_model_patch(source, bounds, top=rect.y, left=rect.x, size=PATCH_SIZE)
                    for rects, _target in rect_groups
                    for rect in rects
                ]
            ).cuda(non_blocking=True)
            targets = torch.stack([target for _rects, target in rect_groups]).cuda(non_blocking=True)
            raw_logits = model(inputs)
            all_rects = tuple(rect for rects, _target in rect_groups for rect in rects)
            normal_flags = tuple(
                not bool(target.any().item())
                for _rects, target in rect_groups
                for _ in _rects
            )
            supervision = _build_batch_supervision(case, all_rects, normal_flags)
            coverage = per_instance_coverage_loss(
                raw_logits,
                supervision.instance_masks.cuda(non_blocking=True),
                supervision.instance_batch_indices.cuda(non_blocking=True),
                supervision.instance_class_indices.cuda(non_blocking=True),
            )
            far_negative = far_negative_suppression_loss(
                raw_logits,
                supervision.far_negative_masks.cuda(non_blocking=True),
                supervision.far_batch_indices.cuda(non_blocking=True),
                supervision.far_class_indices.cuda(non_blocking=True),
                hardest_fraction=FAR_NEGATIVE_HARDEST_FRACTION,
            )
            positive = positive_spatial_topk_loss(raw_logits.reshape(len(rect_groups), -1, len(CLASS_CODES), 64, 64), targets, top_fraction=0.01)
            absent = dense_absent_class_loss(raw_logits.reshape(len(rect_groups), -1, len(CLASS_CODES), 64, 64), targets, hardest_fraction=0.01)
            ranking = (
                same_image_grid_ranking_loss(
                    raw_logits.reshape(len(rect_groups), -1, len(CLASS_CODES), 64, 64),
                    targets,
                    tuple(case.case_id for _ in rect_groups),
                    margin=0.5,
                    top_fraction=0.01,
                )
                if normal is not None
                else raw_logits.sum() * 0.0
            )
            sparse = present_sparse_budget_loss(raw_logits.reshape(len(rect_groups), -1, len(CLASS_CODES), 64, 64), targets, probability_budget=0.01)
            overlap_terms: list[Tensor] = []
            cursor = 0
            for rects, _target in rect_groups:
                count = len(rects)
                overlap_terms.append(overlap_consistency_loss(raw_logits[cursor : cursor + count], rects, feature_stride=FEATURE_STRIDE))
                cursor += count
            overlap = torch.stack(overlap_terms).mean()
            loss = (
                LOSS_WEIGHTS["coverage"] * coverage
                + LOSS_WEIGHTS["far_negative"] * far_negative
                + LOSS_WEIGHTS["positive"] * positive
                + LOSS_WEIGHTS["absent"] * absent
                + LOSS_WEIGHTS["ranking"] * ranking
                + LOSS_WEIGHTS["sparse"] * sparse
                + LOSS_WEIGHTS["overlap"] * overlap
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(sum(epoch_losses) / len(epoch_losses))
        print(f"Ticket 36.06 {case.case_id}: epoch {epoch + 1}/{epochs} loss={losses[-1]:.6f}", flush=True)
    return model, losses


def _score_source_model(model: torch.nn.Module, source: np.ndarray) -> np.ndarray:
    bounds = NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 0.0, 100.0)
    windows = enumerate_inference_windows(IMAGE_WIDTH, IMAGE_HEIGHT, PATCH_SIZE, PATCH_STRIDE)
    local: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), 128):
            selected = windows[start : start + 128]
            inputs = torch.stack(
                [extract_model_patch(source, bounds, top=window.y, left=window.x, size=PATCH_SIZE) for window in selected]
            ).cuda(non_blocking=True)
            local.append(spatial_logits_to_probabilities(model(inputs), (PATCH_SIZE, PATCH_SIZE)).cpu().numpy())
    artifact = generate_all_convolutional_artifact(
        windows,
        np.concatenate(local),
        class_names=CLASS_CODES,
        window_settings={
            "window_size": [PATCH_SIZE, PATCH_SIZE],
            "stride": [PATCH_STRIDE, PATCH_STRIDE],
            "reflect_padding": True,
            "center_weighting": "linear",
        },
        provenance={"checkpoint_format": "wafer_defect_studio.resnet18.v5", "feature_stride": FEATURE_STRIDE},
        model_id="spatial_mil_v7-ticket36.06",
        profile_id="ticket36-single-image-cnn-micro-overfit",
        evaluation_id="ticket35-seed-101-train",
        map_method="spatial_mil_sigmoid",
    )
    return artifact.maps.astype(np.float32, copy=False)


def _quality_row(case: DevelopmentCase, confidence_map: np.ndarray) -> tuple[dict[str, object], dict[str, object]]:
    class_code = case.composition
    class_index = CLASS_CODES.index(class_code)
    grids = annotation_grids(IMAGE_WIDTH, IMAGE_HEIGHT, 512, 512)
    evidence = compute_wafer_quality_evidence(
        (WaferEvidenceCase(case.filename, case.split, case.oracle, grids, confidence_map),),
        CLASS_CODES,
        {value: THRESHOLD for value in CLASS_CODES},
    )["per_class"][class_code]
    response = confidence_map[..., class_index]
    allowed = _global_tolerance_union(case, class_code)
    active = np.isfinite(response) & (response >= THRESHOLD)
    remote_num = int((active & ~allowed).sum())
    active_num = int(active.sum())
    matching = match_ticket36_instance_peaks(
        response,
        tuple(item for item in case.instances if item.defect.class_code == class_code),
        threshold=THRESHOLD,
        tolerance_pixels=TOLERANCE_SOURCE_PIXELS,
    )
    row = {
        "case_id": case.case_id,
        "filename": case.filename,
        "composition": case.composition,
        "class_code": class_code,
        "instance_count": len(case.instances),
        "instance_recall": float(matching["recall"]),
        "unmatched_instance_count": len(matching["unmatched_instance_ids"]),
        "unmatched_component_count": len(matching["unmatched_component_ids"]),
        "merged_component_count": len(matching["merged_component_ids"]),
        "normal_grid_leak_rate": float(evidence["normal_grid_leak_rate"] or 0.0),
        "asserted_grid_occupancy_p95": float(evidence["asserted_grid_occupancy_p95"] or 0.0),
        "remote_response": {
            "numerator": remote_num,
            "denominator": active_num,
            "ratio": remote_num / active_num if active_num else 0.0,
        },
        "threshold": THRESHOLD,
    }
    row["overall"] = "PASS" if _row_passes(row) else "FAIL"
    return row, matching


def _row_passes(row: Mapping[str, object]) -> bool:
    remote = row["remote_response"]
    assert isinstance(remote, Mapping)
    return (
        row["instance_recall"] >= 1.0
        and row["unmatched_instance_count"] == 0
        and row["unmatched_component_count"] == 0
        and row["merged_component_count"] == 0
        and row["normal_grid_leak_rate"] == 0.0
        and row["asserted_grid_occupancy_p95"] <= 0.25
        and remote["ratio"] <= 0.05
    )


def run_single_image_cnn_micro_overfit(output_root: Path) -> dict[str, object]:
    """Execute the frozen four-case CUDA recipe exactly once."""

    if not torch.cuda.is_available():
        raise RuntimeError("Ticket 36.06 requires CUDA")
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if any((output_root / name).exists() for name in ARTIFACT_FILENAMES.values()):
        raise FileExistsError("Ticket 36.06 candidate artifact output already exists")
    repo_root = Path(__file__).resolve().parents[2]
    contract = build_ticket36_localization_contract()
    validate_ticket36_localization_contract(contract, repo_root=repo_root)
    cases = build_ticket36_cases()
    run_started = time.monotonic()
    model_states: dict[str, dict[str, Tensor]] = {}
    maps: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    matching: dict[str, object] = {}
    histories: dict[str, list[float]] = {}
    with tempfile.TemporaryDirectory(prefix="ticket36-single-image-") as temporary:
        source_root = Path(temporary)
        for case in cases:
            _reset_case_seed()
            source = render_ticket35_development_pixels(case)
            model, losses = train_single_case(case, source)
            histories[case.case_id] = losses
            model_states[case.case_id] = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            confidence_map = _score_source_model(model, source)
            maps[case.case_id] = confidence_map.astype(np.float16)
            row, case_matching = _quality_row(case, maps[case.case_id])
            rows.append(row)
            matching[case.case_id] = {case.composition: case_matching}
            del model
            torch.cuda.empty_cache()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / ARTIFACT_FILENAMES["checkpoint"]
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "case_order": list(CASE_IDS),
            "class_codes": list(CLASS_CODES),
            "models": model_states,
        },
        checkpoint_path,
    )
    maps_path = output_root / ARTIFACT_FILENAMES["confidence_maps"]
    np.savez_compressed(maps_path, **{case_id: maps[case_id] for case_id in CASE_IDS})
    matching_path = output_root / ARTIFACT_FILENAMES["per_instance_matching"]
    _write_json(
        matching_path,
        {"schema": MATCHING_SCHEMA, "case_order": list(CASE_IDS), "per_case": matching},
    )
    audit = _build_contract_execution_audit(cases)
    configuration = _build_configuration(cases, histories, repo_root, time.monotonic() - run_started)
    configuration["contract_execution"] = audit["contract_execution"]
    configuration["geometry_audit"] = audit["geometry_audit"]
    configuration["post_run_code_repair"] = POST_RUN_CODE_REPAIR
    configuration["metrics_source"] = SEALED_METRICS_SOURCE
    configuration_path = output_root / ARTIFACT_FILENAMES["configuration"]
    _write_json(configuration_path, configuration)
    seal = _build_artifact_seal(repo_root)
    verify_ticket36_artifact_seal(seal, repo_root=repo_root)
    report = {
        "schema": SCHEMA,
        "overall": "PASS" if all(row["overall"] == "PASS" for row in rows) else "FAIL",
        "recommendation": "spatial_mil_v7" if all(row["overall"] == "PASS" for row in rows) else "cam_v2",
        "spatial_mil_v7_status": "experimental",
        "seed": 101,
        "source_split": "train",
        "class_codes": list(CLASS_CODES),
        "case_order": list(CASE_IDS),
        "per_case": rows,
        "artifact_seal": seal,
        "execution": {
            "device": "cuda",
            "gpu": True,
            "cnn": True,
            "images": True,
            "formal_run_count": 1,
            "rerun": False,
            "artifact_re_read_verified": True,
            "metrics_source": SEALED_METRICS_SOURCE,
            "post_run_code_repair": POST_RUN_CODE_REPAIR,
        },
        "recipe": _recipe(),
        "boundary": {
            "cam_default": "cam_v2",
            "ticket34_final_use": "forbidden",
            "ticket35_reserved_use": "forbidden",
            "ticket35_06": "blocked",
            "ticket36_07": "blocked" if report_overall_is_fail(rows) else "planned",
            "final_pixels_materialized": False,
            "calibration": "forbidden",
        },
        "claims": [
            "single-image CNN micro-overfit only",
            "approximate localization",
            "not segmentation",
            "not Neurocle equivalence",
            "not production accuracy",
        ],
        "contract_execution": audit["contract_execution"],
        "geometry_audit": audit["geometry_audit"],
        "post_run_code_repair": POST_RUN_CODE_REPAIR,
        "metrics_source": SEALED_METRICS_SOURCE,
    }
    validate_ticket36_single_image_report(report, repo_root=repo_root)
    return report


def recompute_report_rows_from_sealed_maps(
    *,
    repo_root: Path,
    report_path: Path,
) -> dict[str, object]:
    """Recompute evidence from the sealed float16 maps without model access."""

    root = Path(repo_root).expanduser().resolve()
    report_file = Path(report_path).expanduser().resolve()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    seal = report.get("artifact_seal")
    if not isinstance(seal, Mapping):
        raise ValueError("sealed evidence report is missing artifact seal")
    verify_ticket36_artifact_seal(seal, repo_root=root)
    maps_path = root / ARTIFACT_PATHS["confidence_maps"]
    with np.load(maps_path, allow_pickle=False) as loaded:
        if tuple(loaded.files) != CASE_IDS:
            raise ValueError("sealed confidence map keys drift")
        maps = {case_id: loaded[case_id] for case_id in CASE_IDS}
    cases = build_ticket36_cases()
    rows: list[dict[str, object]] = []
    matching: dict[str, object] = {}
    for case in cases:
        row, case_matching = _quality_row(case, maps[case.case_id])
        rows.append(row)
        matching[case.case_id] = {case.composition: case_matching}
    matching_path = root / ARTIFACT_PATHS["per_instance_matching"]
    _write_json(
        matching_path,
        {"schema": MATCHING_SCHEMA, "case_order": list(CASE_IDS), "per_case": matching},
    )
    configuration_path = root / ARTIFACT_PATHS["configuration"]
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    audit = _build_contract_execution_audit(cases)
    configuration["contract_execution"] = audit["contract_execution"]
    configuration["geometry_audit"] = audit["geometry_audit"]
    configuration["post_run_code_repair"] = POST_RUN_CODE_REPAIR
    configuration["metrics_source"] = SEALED_METRICS_SOURCE
    _write_json(configuration_path, configuration)
    repaired_seal = _build_artifact_seal(root)
    verify_ticket36_artifact_seal(repaired_seal, repo_root=root)
    updated = dict(report)
    updated["per_case"] = rows
    updated["artifact_seal"] = repaired_seal
    execution = dict(updated.get("execution", {}))
    execution.update(
        {
            "artifact_re_read_verified": True,
            "metrics_source": SEALED_METRICS_SOURCE,
            "post_run_code_repair": POST_RUN_CODE_REPAIR,
            "formal_run_count": 1,
            "rerun": False,
        }
    )
    updated["execution"] = execution
    updated["contract_execution"] = audit["contract_execution"]
    updated["geometry_audit"] = audit["geometry_audit"]
    updated["post_run_code_repair"] = POST_RUN_CODE_REPAIR
    updated["metrics_source"] = SEALED_METRICS_SOURCE
    validate_ticket36_single_image_report(updated, repo_root=root)
    report_file.write_text(
        canonical_ticket36_single_image_report_json(updated) + "\n", encoding="utf-8"
    )
    verify_ticket36_artifact_seal(repaired_seal, repo_root=root)
    return updated


def _build_contract_execution_audit(
    cases: Sequence[DevelopmentCase],
) -> dict[str, object]:
    """Record the known pre-repair Normal Grid geometry violation."""

    case = next(
        value
        for value in cases
        if value.case_id == "ticket35-density-101-train-particle-000-150-10"
    )
    grids = annotation_grids(IMAGE_WIDTH, IMAGE_HEIGHT, 512, 512)
    truth = case.oracle.grid_truth(grids)
    total_rows = 0
    normal_rows = 0
    violating_rows = 0
    examples: list[dict[str, object]] = []
    config = TrainingConfig(
        "ticket36-single-image-cnn-micro-overfit",
        case.case_id,
        len(CLASS_CODES),
        MICRO_EPOCHS,
        FEATURE_STRIDE,
        device="cpu",
        seed=101,
        learning_rate=LEARNING_RATE,
        weights_policy=WEIGHTS_POLICY,
        patch_size=PATCH_SIZE,
        patch_stride=PATCH_STRIDE,
        training_policy="spatial_mil_v5",
    )
    for grid in grids:
        rects = enumerate_model_patch_rects(
            Rect(grid.x, grid.y, grid.width, grid.height), config
        )
        normal = not truth[(grid.row, grid.column)]
        for rect in rects:
            # This explicitly reproduces the formal pre-repair assumption:
            # every patch was treated as asserted, including Normal Grids.
            old = build_patch_supervision(case, (rect,), is_normal_grid=False)
            total_rows += len(CLASS_CODES)
            if normal:
                normal_rows += len(CLASS_CODES)
                for class_index, far_mask in enumerate(old.far_negative_masks):
                    non_far = int((~far_mask).sum().item())
                    if non_far:
                        violating_rows += 1
                        if len(examples) < 3:
                            examples.append(
                                {
                                    "case_id": case.case_id,
                                    "grid": [grid.row, grid.column],
                                    "rect": [rect.x, rect.y, rect.width, rect.height],
                                    "class_code": CLASS_CODES[class_index],
                                    "non_far_cells": non_far,
                                }
                            )
    if (violating_rows, total_rows) != (21, 882):
        raise ValueError(
            "formal Normal Grid geometry audit drift: "
            f"{violating_rows}/{total_rows}"
        )
    geometry_audit = {
        "normal_grid_full_far": {
            "valid": False,
            "violating_rows": violating_rows,
            "total_rows": total_rows,
            "normal_grid_rows": normal_rows,
            "examples": examples,
            "formal_run_implementation": "global_same_class_tolerance_on_all_patches",
            "post_run_repair": POST_RUN_CODE_REPAIR,
        },
        "asserted_grid_same_class_tolerance_union": {
            "valid": True,
            "same_class_core_never_far": True,
            "cross_patch_tolerance_intersections": "retained",
        },
    }
    return {
        "contract_execution": {
            "valid": False,
            "evidence_source": SEALED_METRICS_SOURCE,
            "formal_run_count": 1,
            "rerun": False,
            "runner_source_at_formal_run": "pre_repair_normal_far_global_tolerance",
            "post_run_code_repair": POST_RUN_CODE_REPAIR,
            "overall": "FAIL",
            "recommendation": "cam_v2",
            "preflight_errors": list(PREFLIGHT_ERRORS),
        },
        "geometry_audit": geometry_audit,
    }


def report_overall_is_fail(rows: Sequence[Mapping[str, object]]) -> bool:
    return not all(row.get("overall") == "PASS" for row in rows)


def _reset_case_seed() -> None:
    torch.manual_seed(101)
    np.random.seed(101)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(101)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _recipe() -> dict[str, object]:
    return {
        "seed": 101,
        "epochs": MICRO_EPOCHS,
        "optimizer": "adamw",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "weights_policy": WEIGHTS_POLICY,
        "patch_size": PATCH_SIZE,
        "patch_stride": PATCH_STRIDE,
        "feature_stride": FEATURE_STRIDE,
        "threshold": THRESHOLD,
        "calibration": "forbidden",
        "loss_weights": dict(LOSS_WEIGHTS),
        "far_negative_hardest_fraction": FAR_NEGATIVE_HARDEST_FRACTION,
        "tolerance_source_pixels": TOLERANCE_SOURCE_PIXELS,
        "old_ticket35_ring_loss": "removed",
    }


def _build_configuration(
    cases: Sequence[DevelopmentCase],
    histories: Mapping[str, Sequence[float]],
    repo_root: Path,
    elapsed_seconds: float,
) -> dict[str, object]:
    contract_text = canonical_ticket36_localization_contract_json(build_ticket36_localization_contract()) + "\n"
    contract_path = repo_root / "docs/demo/ticket36-localization-contract.json"
    corpus_path = repo_root / "docs/demo/ticket35-development-corpus.json"
    manifest_path = repo_root / SPARSE_MANIFEST_PATH
    return {
        "schema": CONFIG_SCHEMA,
        "members": [
            {
                "case_id": case.case_id,
                "filename": case.filename,
                "composition": case.composition,
                "class_code": case.composition,
                "instance_count": len(case.instances),
            }
            for case in cases
        ],
        "independent_init": {
            "per_case": True,
            "seed_reset": {case.case_id: 101 for case in cases},
            "fresh_resnet18_v5_imagenet": True,
            "cudnn_deterministic": True,
        },
        "recipe": _recipe(),
        "loss_history": {case_id: [float(value) for value in histories[case_id]] for case_id in CASE_IDS},
        "git_pre_run_head": _git_head(repo_root),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "cuda_available": torch.cuda.is_available(),
            "platform": platform.platform(),
        },
        "contract_truth_hashes": {
            "contract_sha256": _sha256_text(contract_text),
            "contract_file_sha256": _sha256_file(contract_path),
            "development_corpus_file_sha256": _sha256_file(corpus_path),
            "sparse_manifest_file_sha256": _sha256_file(manifest_path),
            "sparse_bundle_sha256": SPARSE_BUNDLE_SHA256,
            "contract_schema": CONTRACT_SCHEMA,
        },
        "elapsed_seconds": float(elapsed_seconds),
        "candidate_artifacts": {
            "root": ARTIFACT_ROOT,
            "hash_policy": HASH_POLICY,
            "seal_status": "future_artifact_seal_required",
        },
    }


def _build_artifact_seal(repo_root: Path) -> dict[str, object]:
    roles = []
    for role in ARTIFACT_ROLES:
        relative = ARTIFACT_PATHS[role]
        path = repo_root / relative
        payload = path.read_bytes()
        roles.append(
            {
                "role": role,
                "path": relative.as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    seal = {
        "root": ARTIFACT_ROOT,
        "hash_policy": HASH_POLICY,
        "seal_status": "future_artifact_seal_required",
        "roles": roles,
    }
    validate_ticket36_candidate_artifact_seal(seal)
    return seal


def verify_ticket36_artifact_seal(
    seal: Mapping[str, object],
    *,
    repo_root: Path | None = None,
) -> None:
    """Re-read every candidate file and fail closed on bytes/hash drift."""

    validate_ticket36_candidate_artifact_seal(seal)
    root = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root)
    roles = seal["roles"]
    assert isinstance(roles, list)
    for item in roles:
        assert isinstance(item, Mapping)
        path = root / str(item["path"])
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ValueError(f"candidate artifact missing: {path}") from error
        expected_bytes = item["bytes"]
        expected_sha = item["sha256"]
        if len(payload) != expected_bytes:
            raise ValueError(f"candidate artifact byte count mismatch: {path}")
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ValueError(f"candidate artifact SHA-256 mismatch: {path}")


def validate_ticket36_single_image_report(
    report: Mapping[str, object],
    *,
    repo_root: Path | None = None,
) -> None:
    """Validate the tracked four-row gate and candidate artifact seal."""

    if not isinstance(report, Mapping) or report.get("schema") != SCHEMA:
        raise ValueError("Ticket 36.06 report schema drift")
    if report.get("seed") != 101 or report.get("source_split") != "train":
        raise ValueError("Ticket 36.06 source boundary drift")
    if tuple(report.get("class_codes", ())) != CLASS_CODES:
        raise ValueError("Ticket 36.06 class order drift")
    if tuple(report.get("case_order", ())) != CASE_IDS:
        raise ValueError("Ticket 36.06 case order drift")
    if report.get("recipe") != _recipe():
        raise ValueError("Ticket 36.06 recipe drift")
    repo = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root).expanduser().resolve()
    boundary = report.get("boundary")
    expected_boundary = {
        "cam_default": "cam_v2",
        "ticket34_final_use": "forbidden",
        "ticket35_reserved_use": "forbidden",
        "ticket35_06": "blocked",
        "ticket36_07": boundary.get("ticket36_07") if isinstance(boundary, Mapping) else None,
        "final_pixels_materialized": False,
        "calibration": "forbidden",
    }
    if not isinstance(boundary, Mapping) or boundary != expected_boundary or boundary.get("ticket36_07") not in {"blocked", "planned"}:
        raise ValueError("Ticket 36.06 decision boundary drift")
    claims = report.get("claims")
    if claims != [
        "single-image CNN micro-overfit only",
        "approximate localization",
        "not segmentation",
        "not Neurocle equivalence",
        "not production accuracy",
    ]:
        raise ValueError("Ticket 36.06 claim boundary drift")
    rows = report.get("per_case")
    if not isinstance(rows, list) or tuple(row.get("case_id") for row in rows if isinstance(row, Mapping)) != CASE_IDS:
        raise ValueError("Ticket 36.06 per-case rows drift")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Ticket 36.06 row malformed")
        for field in ("instance_recall", "normal_grid_leak_rate", "asserted_grid_occupancy_p95"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
                raise ValueError(f"Ticket 36.06 row metric invalid: {field}")
        remote = row.get("remote_response")
        if not isinstance(remote, Mapping) or set(remote) != {"numerator", "denominator", "ratio"}:
            raise ValueError("Ticket 36.06 remote response metric drift")
        if remote["denominator"] < 0 or remote["numerator"] < 0 or remote["numerator"] > remote["denominator"]:
            raise ValueError("Ticket 36.06 remote response counts invalid")
        if row.get("overall") != ("PASS" if _row_passes(row) else "FAIL"):
            raise ValueError(f"Ticket 36.06 row gate drift: {row.get('case_id')}")
    expected_overall = "PASS" if all(row["overall"] == "PASS" for row in rows) else "FAIL"
    if report.get("overall") != expected_overall:
        raise ValueError("Ticket 36.06 overall gate drift")
    if report.get("recommendation") != ("spatial_mil_v7" if expected_overall == "PASS" else "cam_v2"):
        raise ValueError("Ticket 36.06 recommendation drift")
    seal = report.get("artifact_seal")
    if not isinstance(seal, Mapping):
        raise ValueError("Ticket 36.06 artifact seal missing")
    verify_ticket36_artifact_seal(seal, repo_root=repo)
    matching = _read_sealed_json_artifact(seal, "per_instance_matching", repo)
    configuration = _read_sealed_json_artifact(seal, "configuration", repo)
    if matching.get("schema") != MATCHING_SCHEMA or tuple(matching.get("case_order", ())) != CASE_IDS:
        raise ValueError("Ticket 36.06 sealed matching schema drift")
    if configuration.get("schema") != CONFIG_SCHEMA:
        raise ValueError("Ticket 36.06 sealed configuration schema drift")
    if configuration.get("recipe") != _recipe():
        raise ValueError("Ticket 36.06 sealed configuration recipe drift")
    if report.get("metrics_source") != SEALED_METRICS_SOURCE:
        raise ValueError("Ticket 36.06 metrics source must be sealed float16 recompute")
    execution = report.get("execution")
    if not isinstance(execution, Mapping) or execution.get("formal_run_count") != 1 or execution.get("rerun") is not False:
        raise ValueError("Ticket 36.06 formal execution count/rerun drift")
    if configuration.get("metrics_source") != SEALED_METRICS_SOURCE:
        raise ValueError("Ticket 36.06 configuration metrics source drift")
    for audit_field in ("contract_execution", "geometry_audit"):
        if report.get(audit_field) != configuration.get(audit_field):
            raise ValueError(f"Ticket 36.06 {audit_field} drift")
    contract_execution = report.get("contract_execution")
    if not isinstance(contract_execution, Mapping) or contract_execution.get("valid") is not False or contract_execution.get("evidence_source") != SEALED_METRICS_SOURCE or contract_execution.get("formal_run_count") != 1 or contract_execution.get("rerun") is not False or contract_execution.get("overall") != "FAIL" or contract_execution.get("recommendation") != "cam_v2" or contract_execution.get("post_run_code_repair") != POST_RUN_CODE_REPAIR or tuple(contract_execution.get("preflight_errors", ())) != PREFLIGHT_ERRORS:
        raise ValueError("Ticket 36.06 contract execution audit drift")
    geometry_audit = report.get("geometry_audit")
    normal_audit = geometry_audit.get("normal_grid_full_far") if isinstance(geometry_audit, Mapping) else None
    if not isinstance(normal_audit, Mapping) or normal_audit.get("valid") is not False or normal_audit.get("violating_rows") != 21 or normal_audit.get("total_rows") != 882 or normal_audit.get("post_run_repair") != POST_RUN_CODE_REPAIR:
        raise ValueError("Ticket 36.06 geometry audit drift")
    if report.get("post_run_code_repair") != POST_RUN_CODE_REPAIR:
        raise ValueError("Ticket 36.06 post-run repair disclosure drift")
    members = configuration.get("members")
    if not isinstance(members, list) or tuple(item.get("case_id") for item in members) != CASE_IDS:
        raise ValueError("Ticket 36.06 sealed configuration members drift")
    per_case_matching = matching.get("per_case")
    if not isinstance(per_case_matching, Mapping):
        raise ValueError("Ticket 36.06 sealed matching rows missing")
    for row in rows:
        case_id = row["case_id"]
        class_code = row["class_code"]
        case_match = per_case_matching.get(case_id)
        if not isinstance(case_match, Mapping) or not isinstance(case_match.get(class_code), Mapping):
            raise ValueError(f"Ticket 36.06 sealed matching row missing: {case_id}")
        match = case_match[class_code]
        expected_metrics = {
            "instance_count": match.get("instance_count"),
            "instance_recall": match.get("recall"),
            "unmatched_instance_count": len(match.get("unmatched_instance_ids", ())),
            "unmatched_component_count": len(match.get("unmatched_component_ids", ())),
            "merged_component_count": len(match.get("merged_component_ids", ())),
        }
        for field, expected in expected_metrics.items():
            if row.get(field) != expected:
                raise ValueError(f"Ticket 36.06 report/sealed matching drift: {case_id} {field}")


def _read_sealed_json_artifact(
    seal: Mapping[str, object],
    role: str,
    repo_root: Path,
) -> dict[str, object]:
    roles = seal.get("roles")
    if not isinstance(roles, list):
        raise ValueError("Ticket 36.06 sealed artifact roles missing")
    entry = next((item for item in roles if isinstance(item, Mapping) and item.get("role") == role), None)
    if entry is None:
        raise ValueError(f"Ticket 36.06 sealed artifact role missing: {role}")
    path = repo_root / str(entry["path"])
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Ticket 36.06 sealed {role} JSON invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"Ticket 36.06 sealed {role} must be an object")
    return value


def canonical_ticket36_single_image_report_json(report: Mapping[str, object]) -> str:
    validate_ticket36_single_image_report(report)
    return json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _global_tolerance_union(case: DevelopmentCase, class_code: str) -> np.ndarray:
    rect = Rect(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT)
    result = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.bool_)
    for item in case.instances:
        if item.defect.class_code == class_code:
            result |= _rasterize_tolerance(item.defect, rect)
    return result


def _coerce_rect(value: Rect | tuple[int, int, int, int]) -> Rect:
    if isinstance(value, Rect):
        return value
    if isinstance(value, tuple) and len(value) == 4:
        return Rect(*value)
    raise ValueError("patch_rects must contain Rect values or four-integer tuples")


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--recompute-sealed",
        action="store_true",
        help="recompute report/matching/configuration from sealed float16 maps only",
    )
    args = parser.parse_args(argv)
    if args.recompute_sealed:
        report = recompute_report_rows_from_sealed_maps(
            repo_root=Path(__file__).resolve().parents[2],
            report_path=args.report,
        )
    else:
        if args.output_root is None:
            parser.error("--output-root is required unless --recompute-sealed is used")
        report = run_single_image_cnn_micro_overfit(args.output_root)
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(canonical_ticket36_single_image_report_json(report) + "\n", encoding="utf-8")
    print(f"Ticket 36.06 single-image CNN micro-overfit: {report['overall']}", flush=True)
    return 0


__all__ = [
    "ARTIFACT_PATHS",
    "CASE_IDS",
    "CASE_SPECS",
    "CLASS_CODES",
    "FEATURE_STRIDE",
    "PATCH_SIZE",
    "PATCH_STRIDE",
    "PatchSupervision",
    "SCHEMA",
    "build_patch_supervision",
    "build_ticket36_cases",
    "canonical_ticket36_single_image_report_json",
    "recompute_report_rows_from_sealed_maps",
    "run_single_image_cnn_micro_overfit",
    "train_single_case",
    "validate_ticket36_single_image_report",
    "verify_ticket36_artifact_seal",
]


if __name__ == "__main__":
    raise SystemExit(main())
