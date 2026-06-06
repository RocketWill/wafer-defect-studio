"""CUDA density micro-overfit for Ticket 35 instance-aware spatial MIL.

This demo owns only the four seed-101/train density extremes.  It renders the
frozen Ticket 35 oracle on demand, converts sparse point/scribble truth to
stride-2 feature masks, and trains the existing v5 ResNet spatial head with
the Ticket 35.04 instance-aware losses.  It deliberately does not touch the
production worker or model registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from math import ceil, isfinite
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
    negative_ring_suppression_loss,
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
from wafer_defect_studio.training_patch_dataset import TrainingPatchDataset
from wafer_defect_studio.training_protocol import TrainingConfig


SCHEMA = "ticket35-density-micro-overfit.v1"
CLASS_CODES = ("scratch", "particle")
FEATURE_STRIDE = 2
FIXED_THRESHOLD = 0.5
MICRO_EPOCHS = 30
PATCH_SIZE = 128
PATCH_STRIDE = 64
WEIGHTS_POLICY = "imagenet"
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP_NORM = 5.0
DECLARED_RING_RADIUS = 2
MICRO_CASE_COUNTS = (
    ("scratch", 1, 0),
    ("scratch", 150, 0),
    ("particle", 0, 1),
    ("particle", 0, 150),
)


@dataclass(frozen=True)
class FeatureInstanceSupervision:
    """One sparse row per instance/patch overlap at feature-map resolution."""

    masks: Tensor
    batch_indices: Tensor
    class_indices: Tensor
    patch_indices: Tensor
    instance_ids: tuple[str, ...]


def build_ticket35_density_micro_cases(
    corpus: Sequence[DevelopmentCase] | None = None,
) -> tuple[DevelopmentCase, ...]:
    """Return exactly the frozen seed-101/train density extremes."""

    all_cases = tuple(build_ticket35_development_corpus() if corpus is None else corpus)
    selected = tuple(
        case
        for case in all_cases
        if case.seed == 101
        and case.split == "train"
        and (case.composition, case.scratch_count, case.particle_count) in MICRO_CASE_COUNTS
    )
    if tuple(
        (case.composition, case.scratch_count, case.particle_count) for case in selected
    ) != MICRO_CASE_COUNTS:
        raise ValueError("Ticket 35 density micro-overfit case membership drift")
    names = {
        value
        for case in selected
        for value in (
            case.case_id,
            case.filename,
            *(instance.instance_id for instance in case.instances),
        )
    }
    forbidden = ("ticket30-evidence-", *RESERVED_FINAL_MEMBER_IDS)
    if any(token in value for value in names for token in forbidden):
        raise ValueError("Ticket 35 density micro-overfit contains forbidden final member")
    return selected


def build_feature_instance_supervision(
    case: DevelopmentCase,
    patch_rects: Sequence[Rect | tuple[int, int, int, int]],
    *,
    feature_stride: int = FEATURE_STRIDE,
) -> FeatureInstanceSupervision:
    """Rasterize sparse source truth separately for every overlapping patch."""

    if feature_stride < 1 or isinstance(feature_stride, bool):
        raise ValueError("feature_stride must be a positive integer")
    rects = tuple(_coerce_rect(rect) for rect in patch_rects)
    if not rects:
        raise ValueError("patch_rects must not be empty")
    masks: list[Tensor] = []
    batch_indices: list[int] = []
    class_indices: list[int] = []
    patch_indices: list[int] = []
    instance_ids: list[str] = []
    for patch_index, rect in enumerate(rects):
        if rect.width % feature_stride or rect.height % feature_stride:
            raise ValueError("patch dimensions must be divisible by feature_stride")
        height, width = rect.height // feature_stride, rect.width // feature_stride
        for instance in case.instances:
            mask = _rasterize_instance(instance.defect, rect, feature_stride, height, width)
            if not mask.any():
                continue
            masks.append(torch.from_numpy(mask))
            batch_indices.append(patch_index)
            class_indices.append(CLASS_CODES.index(instance.defect.class_code))
            patch_indices.append(patch_index)
            instance_ids.append(instance.instance_id)
    if masks:
        mask_tensor = torch.stack(masks).to(dtype=torch.bool)
        device = mask_tensor.device
    else:
        height = rects[0].height // feature_stride
        width = rects[0].width // feature_stride
        mask_tensor = torch.zeros((0, height, width), dtype=torch.bool)
        device = mask_tensor.device
    return FeatureInstanceSupervision(
        mask_tensor,
        torch.tensor(batch_indices, dtype=torch.long, device=device),
        torch.tensor(class_indices, dtype=torch.long, device=device),
        torch.tensor(patch_indices, dtype=torch.long, device=device),
        tuple(instance_ids),
    )


def build_negative_ring_masks(
    supervision: FeatureInstanceSupervision,
) -> Tensor:
    """Build non-empty caller-side rings after subtracting same-class positives."""

    masks = supervision.masks
    if masks.ndim != 3:
        raise ValueError("supervision masks must have shape (N, H, W)")
    if masks.shape[0] == 0:
        return masks.clone()
    rings: list[Tensor] = []
    for index in range(masks.shape[0]):
        positive = masks[index]
        same_patch_class = (
            (supervision.patch_indices == supervision.patch_indices[index])
            & (supervision.class_indices == supervision.class_indices[index])
        )
        same_class_positive = masks[same_patch_class].any(dim=0)
        dilated = F.max_pool2d(
            positive.to(dtype=torch.float32)[None, None], kernel_size=3, stride=1, padding=1
        )[0, 0].to(dtype=torch.bool)
        ring = dilated & ~same_class_positive
        if not bool(ring.any().item()):
            available = ~same_class_positive
            if not bool(available.any().item()):
                raise ValueError("sparse instance positive extent leaves no ring cell")
            ring = torch.zeros_like(available)
            ring.flatten()[int(torch.nonzero(available.flatten(), as_tuple=False)[0])] = True
        rings.append(ring)
    return torch.stack(rings)


def run_micro_overfit(
    output_root: Path,
    *,
    epochs: int = MICRO_EPOCHS,
) -> dict[str, object]:
    """Train and score the four frozen density extremes on real CUDA."""

    if epochs != MICRO_EPOCHS:
        raise ValueError(f"Ticket 35 micro-overfit epochs are frozen at {MICRO_EPOCHS}")
    if not torch.cuda.is_available():
        raise RuntimeError("Ticket 35 density micro-overfit requires CUDA")
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = build_ticket35_density_micro_cases()
    torch.manual_seed(101)
    np.random.seed(101)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = TrainingConfig(
        "ticket35-density-micro-overfit",
        "ticket35-seed-101",
        len(CLASS_CODES),
        MICRO_EPOCHS,
        2,
        device="cuda",
        seed=101,
        learning_rate=LEARNING_RATE,
        weights_policy=WEIGHTS_POLICY,
        patch_size=PATCH_SIZE,
        patch_stride=PATCH_STRIDE,
        training_policy="spatial_mil_v5",
    )
    materialized = _materialize_bundle(cases, config, output_root / "sources")
    dataset = TrainingPatchDataset(materialized, "train", include_patch_rects=True)
    pairs = _training_pairs(materialized, cases)
    model, losses = train_instance_aware_model(cases, materialized, dataset, pairs, epochs=epochs)
    maps = tuple(
        _score_source_model(model, render_ticket35_development_pixels(case))
        for case in cases
    )
    per_case = {
        case.case_id: _quality_row(case, maps[index], case.composition)
        for index, case in enumerate(cases)
    }
    overall = "PASS" if all(row["overall"] == "PASS" for row in per_case.values()) else "FAIL"
    report = {
        "schema": SCHEMA,
        "overall": overall,
        "recommendation": "spatial_mil_v7" if overall == "PASS" else "cam_v2",
        "spatial_mil_v7_status": "experimental",
        "seed": 101,
        "source_split": "train",
        "class_codes": list(CLASS_CODES),
        "threshold": FIXED_THRESHOLD,
        "calibration": "forbidden",
        "case_count": len(cases),
        "case_compositions": [case.composition for case in cases],
        "recipe": {
            "epochs": MICRO_EPOCHS,
            "weights_policy": WEIGHTS_POLICY,
            "patch_size": PATCH_SIZE,
            "patch_stride": PATCH_STRIDE,
            "feature_stride": FEATURE_STRIDE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "optimizer": "adamw",
            "instance_coverage_loss": "per_instance_coverage_loss",
            "negative_ring_loss": "negative_ring_suppression_loss",
        },
        "loss": {"first": losses[0], "last": losses[-1]},
        "device": {
            "name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "per_case": per_case,
        "boundary": {
            "ticket34_final_use": "forbidden",
            "ticket34_final_members": [
                "ticket30-evidence-17.png",
                "ticket30-evidence-42.png",
                "ticket30-evidence-91.png",
            ],
            "ticket35_reserved_final_use": "forbidden",
            "ticket35_reserved_final_members": list(RESERVED_FINAL_MEMBER_IDS),
            "final_pixels_materialized": False,
            "development_pixels_materialized_on_demand": True,
        },
        "claims": [
            "density micro-overfit only",
            "approximate localization",
            "not segmentation",
            "not Neurocle equivalence",
            "not production accuracy",
        ],
    }
    validate_ticket35_density_micro_overfit_report(report)
    return report


def train_instance_aware_model(
    cases: Sequence[DevelopmentCase],
    bundle: TrainingInputBundle,
    dataset: TrainingPatchDataset,
    pairs: Sequence[tuple[int, int | None, DevelopmentCase]],
    *,
    epochs: int,
):
    """Train the v5 model locally without routing through the production worker."""

    model = create_resnet18_spatial_logits_v5(
        len(CLASS_CODES), weights=WEIGHTS_POLICY, device="cuda"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for asserted_index, normal_index, case in pairs:
            first = dataset[asserted_index]
            items = [first]
            if normal_index is not None:
                items.append(dataset[normal_index])
            inputs = torch.stack([item[0] for item in items]).cuda(non_blocking=True)
            targets = torch.stack([item[1] for item in items]).cuda(non_blocking=True)
            rects = tuple(item[2] for item in items)
            batch_size, patch_count, channels, height, width = inputs.shape
            raw_logits = model(inputs.reshape(batch_size * patch_count, channels, height, width))
            logits = raw_logits.reshape(
                batch_size,
                patch_count,
                len(CLASS_CODES),
                raw_logits.shape[-2],
                raw_logits.shape[-1],
            )
            rect_values = tuple(
                Rect(*map(int, value.tolist())) for value in first[2]
            )
            supervision = build_feature_instance_supervision(
                case, rect_values, feature_stride=FEATURE_STRIDE
            )
            ring_masks = build_negative_ring_masks(supervision)
            device_supervision = FeatureInstanceSupervision(
                supervision.masks.cuda(non_blocking=True),
                supervision.batch_indices.cuda(non_blocking=True),
                supervision.class_indices.cuda(non_blocking=True),
                supervision.patch_indices.cuda(non_blocking=True),
                supervision.instance_ids,
            )
            coverage = per_instance_coverage_loss(
                raw_logits,
                device_supervision.masks,
                device_supervision.batch_indices,
                device_supervision.class_indices,
            )
            ring = negative_ring_suppression_loss(
                raw_logits,
                ring_masks.cuda(non_blocking=True),
                device_supervision.batch_indices,
                device_supervision.class_indices,
            )
            positive = positive_spatial_topk_loss(logits, targets, top_fraction=0.01)
            absent = dense_absent_class_loss(logits, targets, hardest_fraction=0.01)
            ranking = (
                same_image_grid_ranking_loss(
                    logits,
                    targets,
                    (case.case_id, case.case_id),
                    margin=0.5,
                    top_fraction=0.01,
                )
                if normal_index is not None
                else raw_logits.sum() * 0.0
            )
            sparse = present_sparse_budget_loss(logits, targets, probability_budget=0.01)
            overlap = torch.stack(
                [
                    overlap_consistency_loss(
                        bag_logits,
                        [Rect(*map(int, value.tolist())) for value in bag_rects],
                        feature_stride=FEATURE_STRIDE,
                    )
                    for bag_logits, bag_rects in zip(logits, rects, strict=True)
                ]
            ).mean()
            loss = coverage + ring + 0.25 * positive + 0.50 * absent + 0.25 * ranking + 0.10 * sparse + 0.10 * overlap
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(sum(epoch_losses) / len(epoch_losses))
        print(
            f"Ticket 35 seed 101: epoch {epoch + 1}/{epochs} loss={losses[-1]:.6f}",
            flush=True,
        )
    return model, losses


def canonical_ticket35_density_micro_overfit_json(
    report: Mapping[str, object],
) -> str:
    return json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_ticket35_density_micro_overfit_report(
    report: Mapping[str, object],
) -> None:
    """Fail closed on report schema, fixed recipe, and four-row gate semantics."""

    if not isinstance(report, Mapping):
        raise ValueError("Ticket 35 density micro-overfit report must be an object")
    if report.get("schema") != SCHEMA:
        raise ValueError("Ticket 35 density micro-overfit schema drift")
    if report.get("seed") != 101 or report.get("source_split") != "train":
        raise ValueError("Ticket 35 density micro-overfit source boundary drift")
    if report.get("threshold") != FIXED_THRESHOLD or report.get("calibration") != "forbidden":
        raise ValueError("Ticket 35 density micro-overfit threshold/calibration drift")
    if tuple(report.get("class_codes", ())) != CLASS_CODES:
        raise ValueError("Ticket 35 density micro-overfit class order drift")
    recipe = report.get("recipe")
    expected_recipe = {
        "epochs": MICRO_EPOCHS,
        "weights_policy": WEIGHTS_POLICY,
        "patch_size": PATCH_SIZE,
        "patch_stride": PATCH_STRIDE,
        "feature_stride": FEATURE_STRIDE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "optimizer": "adamw",
        "instance_coverage_loss": "per_instance_coverage_loss",
        "negative_ring_loss": "negative_ring_suppression_loss",
    }
    if recipe != expected_recipe:
        raise ValueError("Ticket 35 density micro-overfit recipe drift")
    boundary = report.get("boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("Ticket 35 density micro-overfit boundary missing")
    if (
        boundary.get("ticket34_final_use") != "forbidden"
        or boundary.get("ticket35_reserved_final_use") != "forbidden"
        or boundary.get("final_pixels_materialized") is not False
        or tuple(boundary.get("ticket34_final_members", ()))
        != ("ticket30-evidence-17.png", "ticket30-evidence-42.png", "ticket30-evidence-91.png")
        or tuple(boundary.get("ticket35_reserved_final_members", ()))
        != RESERVED_FINAL_MEMBER_IDS
    ):
        raise ValueError("Ticket 35 density micro-overfit final boundary drift")
    rows = report.get("per_case")
    if not isinstance(rows, Mapping) or len(rows) != len(MICRO_CASE_COUNTS):
        raise ValueError("Ticket 35 density micro-overfit case rows drift")
    expected_compositions = tuple(item[0] for item in MICRO_CASE_COUNTS)
    actual_compositions = tuple(report.get("case_compositions", ()))
    if actual_compositions != expected_compositions:
        raise ValueError("Ticket 35 density micro-overfit composition order drift")
    for case_id, row in rows.items():
        if not isinstance(case_id, str) or not isinstance(row, Mapping):
            raise ValueError("Ticket 35 density micro-overfit row malformed")
        if row.get("threshold") != FIXED_THRESHOLD:
            raise ValueError(f"Ticket 35 density micro-overfit row threshold drift: {case_id}")
        for field in (
            "instance_recall",
            "normal_grid_leak_rate",
            "asserted_grid_occupancy_p95",
            "outside_declared_extent_response",
        ):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
                raise ValueError(f"Ticket 35 density micro-overfit row metric invalid: {case_id} {field}")
        expected_pass = (
            row["instance_recall"] >= 1.0
            and row["normal_grid_leak_rate"] == 0.0
            and row["asserted_grid_occupancy_p95"] <= 0.25
            and row["outside_declared_extent_response"] == 0.0
        )
        if row.get("overall") != ("PASS" if expected_pass else "FAIL"):
            raise ValueError(f"Ticket 35 density micro-overfit row gate drift: {case_id}")
    expected_overall = "PASS" if all(row["overall"] == "PASS" for row in rows.values()) else "FAIL"
    if report.get("overall") != expected_overall:
        raise ValueError("Ticket 35 density micro-overfit overall gate drift")


def _materialize_bundle(
    cases: Sequence[DevelopmentCase], config: TrainingConfig, source_dir: Path
) -> TrainingInputBundle:
    source_dir.mkdir(parents=True, exist_ok=True)
    sources: list[TrainingBundleSource] = []
    bags: list[TrainingPatchBag] = []
    grids = annotation_grids(1536, 1536, 512, 512)
    for case in cases:
        pixels = render_ticket35_development_pixels(case)
        path = source_dir / case.filename
        image = QImage(
            pixels.data,
            1536,
            1536,
            1536,
            QImage.Format.Format_Grayscale8,
        ).copy()
        if not image.save(str(path), "PNG"):
            raise RuntimeError(f"unable to write Ticket 35 source: {path}")
        image_id = case.filename.removesuffix(".png")
        sources.append(
            TrainingBundleSource(
                image_id,
                case.split,
                str(path),
                hashlib.sha256(path.read_bytes()).hexdigest(),
                "uint8",
            )
        )
        truth = case.oracle.grid_truth(grids)
        for grid in grids:
            grid_rect = Rect(grid.x, grid.y, grid.width, grid.height)
            bags.append(
                TrainingPatchBag(
                    f"{image_id}:{grid.row}:{grid.column}",
                    image_id,
                    grid.row,
                    grid.column,
                    enumerate_model_patch_rects(grid_rect, config),
                    tuple(code for code in CLASS_CODES if code in truth[(grid.row, grid.column)]),
                )
            )
    return TrainingInputBundle(
        "ticket35-density-micro-overfit",
        "ticket35-seed-101-train",
        CLASS_CODES,
        (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 0.0, 100.0),),
        tuple(sources),
        (),
        2,
        tuple(bags),
    )


def _training_pairs(
    bundle: TrainingInputBundle,
    cases: Sequence[DevelopmentCase],
) -> tuple[tuple[int, int | None, DevelopmentCase], ...]:
    case_by_image_id = {case.filename.removesuffix(".png"): case for case in cases}
    bags = tuple(bundle.patch_bags)
    pairs: list[tuple[int, int | None, DevelopmentCase]] = []
    for image_id in dict.fromkeys(bag.image_asset_id for bag in bags):
        indices = [index for index, bag in enumerate(bags) if bag.image_asset_id == image_id]
        normal = [index for index in indices if not bags[index].class_codes]
        asserted = [index for index in indices if bags[index].class_codes]
        if not asserted:
            raise ValueError(f"Ticket 35 micro-overfit image lacks asserted bags: {image_id}")
        case = case_by_image_id[image_id]
        if normal:
            pairs.extend(
                (index, normal[offset % len(normal)], case)
                for offset, index in enumerate(asserted)
            )
        else:
            pairs.extend((index, None, case) for index in asserted)
    return tuple(pairs)


def _score_source_model(model, source: np.ndarray) -> np.ndarray:
    bounds = NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 0.0, 100.0)
    windows = enumerate_inference_windows(1536, 1536, PATCH_SIZE, PATCH_STRIDE)
    local: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), 128):
            selected = windows[start : start + 128]
            inputs = torch.stack(
                [
                    extract_model_patch(
                        source,
                        bounds,
                        top=window.y,
                        left=window.x,
                        size=PATCH_SIZE,
                    )
                    for window in selected
                ]
            ).cuda(non_blocking=True)
            local.append(
                spatial_logits_to_probabilities(model(inputs), (PATCH_SIZE, PATCH_SIZE))
                .cpu()
                .numpy()
            )
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
        model_id="spatial_mil_v7-micro-overfit",
        profile_id="ticket35-density-micro-overfit",
        evaluation_id="ticket35-seed-101-train",
        map_method="spatial_mil_sigmoid",
    )
    return artifact.maps.astype(np.float32, copy=False)


def _quality_row(case: DevelopmentCase, confidence_map: np.ndarray, code: str) -> dict[str, object]:
    class_index = CLASS_CODES.index(code)
    grids = annotation_grids(1536, 1536, 512, 512)
    evidence = compute_wafer_quality_evidence(
        (
            WaferEvidenceCase(
                case.filename,
                case.split,
                case.oracle,
                grids,
                confidence_map,
            ),
        ),
        CLASS_CODES,
        {value: FIXED_THRESHOLD for value in CLASS_CODES},
    )["per_class"][code]
    retained = np.isfinite(confidence_map[..., class_index]) & (
        confidence_map[..., class_index] >= FIXED_THRESHOLD
    )
    allowed = _declared_extent_plus_ring(case, code)
    retained_count = int(retained.sum())
    outside_count = int((retained & ~allowed).sum())
    outside = outside_count / retained_count if retained_count else 0.0
    instance_recall = float(evidence["defect_coverage_recall"] or 0.0)
    occupancy = float(evidence["asserted_grid_occupancy_p95"] or 0.0)
    leak = float(evidence["normal_grid_leak_rate"] or 0.0)
    passed = instance_recall >= 1.0 and leak == 0.0 and occupancy <= 0.25 and outside == 0.0
    return {
        "case_id": case.case_id,
        "filename": case.filename,
        "composition": case.composition,
        "class_code": code,
        "instance_count": sum(instance.defect.class_code == code for instance in case.instances),
        "instance_recall": instance_recall,
        "normal_grid_leak_rate": leak,
        "asserted_grid_occupancy_p95": occupancy,
        "outside_declared_extent_response": outside,
        "outside_declared_extent_cells": outside_count,
        "threshold": FIXED_THRESHOLD,
        "overall": "PASS" if passed else "FAIL",
    }


def _declared_extent_plus_ring(case: DevelopmentCase, code: str) -> np.ndarray:
    base = np.zeros((case.oracle.image_height, case.oracle.image_width), dtype=np.float32)
    for instance in case.instances:
        defect = instance.defect
        if defect.class_code != code:
            continue
        if isinstance(defect, Particle):
            x, y = defect.center
            if 0 <= x < base.shape[1] and 0 <= y < base.shape[0]:
                base[y, x] = 1.0
        elif isinstance(defect, Scratch):
            for start, end in zip(defect.points, defect.points[1:]):
                steps = max(abs(end[0] - start[0]), abs(end[1] - start[1])) + 1
                for index in range(steps):
                    fraction = index / max(1, steps - 1)
                    x = round(start[0] + fraction * (end[0] - start[0]))
                    y = round(start[1] + fraction * (end[1] - start[1]))
                    if 0 <= x < base.shape[1] and 0 <= y < base.shape[0]:
                        base[y, x] = 1.0
        else:
            raise ValueError(f"unsupported Ticket 35 sparse defect: {type(defect).__name__}")
    positive_radius = max(
        (instance.defect.radius for instance in case.instances if instance.defect.class_code == code),
        default=0,
    )
    if positive_radius:
        base_tensor = torch.from_numpy(base)[None, None]
        base = F.max_pool2d(
            base_tensor,
            kernel_size=2 * positive_radius + 1,
            stride=1,
            padding=positive_radius,
        )[0, 0].numpy()
    allowed = F.max_pool2d(
        torch.from_numpy(base)[None, None],
        kernel_size=2 * DECLARED_RING_RADIUS + 1,
        stride=1,
        padding=DECLARED_RING_RADIUS,
    )[0, 0]
    return allowed.to(dtype=torch.bool).numpy()


def _rasterize_instance(
    defect: Scratch | Particle,
    rect: Rect,
    feature_stride: int,
    height: int,
    width: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.bool_)
    if isinstance(defect, Particle):
        points = (defect.center,)
    elif isinstance(defect, Scratch):
        points_list: list[tuple[int, int]] = []
        for start, end in zip(defect.points, defect.points[1:]):
            steps = max(abs(end[0] - start[0]), abs(end[1] - start[1])) + 1
            points_list.extend(
                (
                    round(start[0] + index / max(1, steps - 1) * (end[0] - start[0])),
                    round(start[1] + index / max(1, steps - 1) * (end[1] - start[1])),
                )
                for index in range(steps)
            )
        points = tuple(points_list)
    else:
        raise ValueError(f"unsupported Ticket 35 sparse defect: {type(defect).__name__}")
    for x, y in points:
        if rect.x <= x < rect.x + rect.width and rect.y <= y < rect.y + rect.height:
            feature_x = min(width - 1, max(0, (x - rect.x) // feature_stride))
            feature_y = min(height - 1, max(0, (y - rect.y) // feature_stride))
            mask[feature_y, feature_x] = True
    return mask


def _coerce_rect(value: Rect | tuple[int, int, int, int]) -> Rect:
    if isinstance(value, Rect):
        return value
    if isinstance(value, tuple) and len(value) == 4:
        return Rect(*value)
    raise ValueError("patch_rects must contain Rect values or four-integer tuples")


def _canonical(value: Mapping[str, object]) -> str:
    return canonical_ticket35_density_micro_overfit_json(value) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run_micro_overfit(args.output_root.resolve())
    args.report.resolve().write_text(_canonical(report), encoding="utf-8")
    print(f"Ticket 35 density micro-overfit: {report['overall']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLASS_CODES",
    "DECLARED_RING_RADIUS",
    "FEATURE_STRIDE",
    "FIXED_THRESHOLD",
    "FeatureInstanceSupervision",
    "MICRO_CASE_COUNTS",
    "MICRO_EPOCHS",
    "SCHEMA",
    "build_feature_instance_supervision",
    "build_negative_ring_masks",
    "build_ticket35_density_micro_cases",
    "canonical_ticket35_density_micro_overfit_json",
    "run_micro_overfit",
    "train_instance_aware_model",
    "validate_ticket35_density_micro_overfit_report",
]
