"""CPU-only loss-level micro-overfit for Ticket 36.

This exercise optimizes synthetic source-coordinate logits directly.  It is a
loss and metric check, not a CNN run and not a segmentation experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from docs.demo.defect_oracle import Particle, Scratch
from docs.demo.ticket35_development_corpus import DevelopmentInstance
from docs.demo.ticket36_instance_matching import match_ticket36_instance_peaks
from wafer_defect_studio.spatial_mil import (
    far_negative_suppression_loss,
    per_instance_coverage_loss,
)


SCHEMA = "ticket36-loss-micro-overfit.v1"
REPORT_PATH = Path(__file__).with_name("ticket36-loss-micro-overfit.json")
CLASS_ORDER = ("scratch", "particle")
SOURCE_SIZE = 256
THRESHOLD = 0.5
TOLERANCE_PIXELS = 8.0
CASE_IDS = (
    "single_particle",
    "dense150_particle",
    "single_scratch",
    "dense150_scratch",
    "boundary_particle",
    "boundary_scratch",
    "close_two_particles",
)
SEED = 3605
STEPS = 24
LEARNING_RATE = 0.5
INITIAL_LOGIT = -4.0
HARDEST_FRACTION = 1.0


@dataclass(frozen=True)
class _SyntheticCase:
    case_id: str
    instances: tuple[DevelopmentInstance, ...]


def _particle(case_id: str, index: int, center: tuple[int, int], radius: int = 3) -> DevelopmentInstance:
    return DevelopmentInstance(
        f"ticket36-{case_id}-particle-{index:03d}",
        Particle("particle", center, radius),
    )


def _scratch(
    case_id: str,
    index: int,
    points: tuple[tuple[int, int], ...],
    radius: int = 2,
) -> DevelopmentInstance:
    return DevelopmentInstance(
        f"ticket36-{case_id}-scratch-{index:03d}",
        Scratch("scratch", points, radius),
    )


def _build_cases() -> tuple[_SyntheticCase, ...]:
    dense_particles = tuple(
        _particle(
            "dense150_particle",
            index,
            (16 + 16 * (index % 15), 16 + 16 * (index // 15)),
        )
        for index in range(150)
    )
    dense_scratches = tuple(
        _scratch(
            "dense150_scratch",
            index,
            (
                (8 + 16 * (index % 15), 8 + 16 * (index // 15)),
                (14 + 16 * (index % 15), 8 + 16 * (index // 15)),
            ),
        )
        for index in range(150)
    )
    return (
        _SyntheticCase("single_particle", (_particle("single_particle", 0, (96, 96)),)),
        _SyntheticCase("dense150_particle", dense_particles),
        _SyntheticCase(
            "single_scratch",
            (_scratch("single_scratch", 0, ((64, 64), (220, 140))),),
        ),
        _SyntheticCase("dense150_scratch", dense_scratches),
        _SyntheticCase("boundary_particle", (_particle("boundary_particle", 0, (0, 0)),)),
        _SyntheticCase(
            "boundary_scratch",
            (_scratch("boundary_scratch", 0, ((0, 24), (0, 220))),),
        ),
        _SyntheticCase(
            "close_two_particles",
            (
                _particle("close_two_particles", 0, (160, 160)),
                _particle("close_two_particles", 1, (172, 160)),
            ),
        ),
    )


def _rasterized_segment(
    start: tuple[int, int], end: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
    if steps == 0:
        return (start,)
    cells = {
        (
            int(round(start[0] + (end[0] - start[0]) * fraction / steps)),
            int(round(start[1] + (end[1] - start[1]) * fraction / steps)),
        )
        for fraction in range(steps + 1)
    }
    return tuple(sorted(cells, key=lambda point: (point[1], point[0])))


def _distance_to_segment_grid(
    xx: np.ndarray,
    yy: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> np.ndarray:
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return np.hypot(xx - start[0], yy - start[1])
    projection = ((xx - start[0]) * dx + (yy - start[1]) * dy) / denominator
    projection = np.clip(projection, 0.0, 1.0)
    closest_x = start[0] + projection * dx
    closest_y = start[1] + projection * dy
    return np.hypot(xx - closest_x, yy - closest_y)


def _instance_masks(
    instances: Sequence[DevelopmentInstance],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, object],
]:
    height = width = SOURCE_SIZE
    yy, xx = np.mgrid[0:height, 0:width]
    tolerance_union = {
        class_code: np.zeros((height, width), dtype=bool) for class_code in CLASS_ORDER
    }
    core_masks: list[np.ndarray] = []
    core_cell_counts: list[int] = []
    min_xy: list[tuple[int, int]] = []
    class_indices: list[int] = []
    for instance in instances:
        defect = instance.defect
        class_index = CLASS_ORDER.index(defect.class_code)
        class_indices.append(class_index)
        core = np.zeros((height, width), dtype=bool)
        if isinstance(defect, Particle):
            x, y = defect.center
            core[y, x] = True
            tolerance = np.hypot(xx - x, yy - y) <= TOLERANCE_PIXELS
            min_xy.append((x, y))
        else:
            cells = set()
            tolerance = np.zeros((height, width), dtype=bool)
            for start, end in zip(defect.points, defect.points[1:]):
                cells.update(_rasterized_segment(start, end))
                tolerance |= _distance_to_segment_grid(xx, yy, start, end) <= TOLERANCE_PIXELS
            for x, y in cells:
                core[y, x] = True
            min_xy.extend(defect.points)
        core_masks.append(core)
        core_cell_counts.append(int(core.sum()))
        tolerance_union[defect.class_code] |= tolerance

    if core_masks:
        instance_masks = torch.from_numpy(np.stack(core_masks))
        batch_indices = torch.zeros(len(core_masks), dtype=torch.int64)
        instance_class_indices = torch.tensor(class_indices, dtype=torch.int64)
    else:
        instance_masks = torch.zeros((0, height, width), dtype=torch.bool)
        batch_indices = torch.zeros(0, dtype=torch.int64)
        instance_class_indices = torch.zeros(0, dtype=torch.int64)
    far_masks = torch.from_numpy(
        np.stack([~tolerance_union[class_code] for class_code in CLASS_ORDER])
    )
    geometry = {
        "core_cell_count": int(sum(core_cell_counts)),
        "core_cell_count_min": int(min(core_cell_counts)) if core_cell_counts else 0,
        "tolerance_cell_count": [int(tolerance_union[class_code].sum()) for class_code in CLASS_ORDER],
        "min_xy": list(min(min_xy, key=lambda point: (point[1], point[0]))) if min_xy else [],
    }
    return instance_masks, batch_indices, instance_class_indices, far_masks, geometry


def _run_case(case: _SyntheticCase) -> dict[str, object]:
    instance_masks, batch_indices, class_indices, far_masks, geometry = _instance_masks(case.instances)
    torch.manual_seed(SEED)
    logits = torch.nn.Parameter(torch.full((1, 2, SOURCE_SIZE, SOURCE_SIZE), INITIAL_LOGIT))
    normal_logits = torch.nn.Parameter(torch.full_like(logits, INITIAL_LOGIT))
    optimizer = torch.optim.Adam((logits, normal_logits), lr=LEARNING_RATE)
    normal_masks = torch.ones((2, SOURCE_SIZE, SOURCE_SIZE), dtype=torch.bool)
    normal_batch = torch.zeros(2, dtype=torch.int64)
    normal_classes = torch.tensor([0, 1], dtype=torch.int64)
    losses: list[float] = []
    for _step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        coverage = per_instance_coverage_loss(logits, instance_masks, batch_indices, class_indices)
        far_negative = far_negative_suppression_loss(
            logits,
            far_masks,
            normal_batch,
            normal_classes,
            hardest_fraction=HARDEST_FRACTION,
        )
        normal_negative = far_negative_suppression_loss(
            normal_logits,
            normal_masks,
            normal_batch,
            normal_classes,
            hardest_fraction=HARDEST_FRACTION,
        )
        loss = coverage + far_negative + normal_negative
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    probabilities = torch.sigmoid(logits.detach())[0].cpu().numpy()
    normal_probabilities = torch.sigmoid(normal_logits.detach())[0].cpu().numpy()
    matched = {}
    core_responses: list[float] = []
    for instance_mask, class_index in zip(instance_masks, class_indices, strict=True):
        core_responses.append(float(probabilities[int(class_index)][instance_mask.numpy()].max()))
    for class_index, class_code in enumerate(CLASS_ORDER):
        class_instances = tuple(item for item in case.instances if item.defect.class_code == class_code)
        if not class_instances:
            continue
        matched[class_code] = match_ticket36_instance_peaks(
            probabilities[class_index], class_instances, threshold=THRESHOLD, tolerance_pixels=TOLERANCE_PIXELS
        )
    far_active = int(sum(int((probabilities[index][far_masks[index].numpy()] >= THRESHOLD).sum()) for index in range(2)))
    normal_active = int((normal_probabilities >= THRESHOLD).sum())
    unmatched = sum(len(value["unmatched_instance_ids"]) for value in matched.values())
    merged = sum(len(value["merged_component_ids"]) for value in matched.values())
    instance_recall = min((float(value["recall"]) for value in matched.values()), default=1.0)
    gates = {
        "instance_recall": instance_recall,
        "unmatched": unmatched,
        "merged": merged,
        "far_negative_active_response": far_active,
        "normal_active_response": normal_active,
        "core_response_min": min(core_responses, default=1.0),
    }
    gates["overall"] = "PASS" if (
        instance_recall == 1.0
        and unmatched == 0
        and merged == 0
        and far_active == 0
        and normal_active == 0
        and gates["core_response_min"] >= THRESHOLD
    ) else "FAIL"
    return {
        "case_id": case.case_id,
        "class_codes": sorted({instance.defect.class_code for instance in case.instances}),
        "instance_count": len(case.instances),
        "geometry": geometry,
        "loss": {"first": losses[0], "last": losses[-1]},
        "matching": matched,
        "gates": gates,
    }


def build_ticket36_loss_micro_overfit_report() -> dict[str, object]:
    """Run the frozen CPU recipe once and return the canonical evidence object."""

    random.seed(SEED)
    np.random.seed(SEED)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rows = tuple(_run_case(case) for case in _build_cases())
    finally:
        torch.set_num_threads(previous_threads)
    return {
        "schema": SCHEMA,
        "source_geometry": {
            "coordinate_system": "source_pixels",
            "width": SOURCE_SIZE,
            "height": SOURCE_SIZE,
            "synthetic": True,
        },
        "logits_shape": [1, 2, SOURCE_SIZE, SOURCE_SIZE],
        "recipe": {
            "seed": SEED,
            "device": "cpu",
            "optimizer": "Adam",
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "initial_logit": INITIAL_LOGIT,
            "threshold": THRESHOLD,
            "hardest_fraction": HARDEST_FRACTION,
            "epoch_selection": "forbidden",
            "rerun_or_tuning": "forbidden",
        },
        "losses": ["per_instance_coverage_loss", "far_negative_suppression_loss"],
        "execution": {"device": "cpu", "cnn": False, "gpu": False, "images": False},
        "artifact_contract": {
            "roles": ["future CNN candidate only"],
            "seal_status": "future_artifact_seal_required",
            "checkpoint": False,
            "confidence_maps": False,
            "per_instance_matching": "metric_only",
        },
        "claims": [
            "loss-level micro-overfit only",
            "not segmentation",
            "not Neurocle equivalence",
            "not production accuracy",
        ],
        "boundary": {
            "cam_default": "cam_v2",
            "ticket34_final_use": "forbidden",
            "ticket35_reserved_use": "forbidden",
        },
        "per_case": list(rows),
        "overall": "PASS" if all(row["gates"]["overall"] == "PASS" for row in rows) else "FAIL",
    }


def canonical_ticket36_loss_micro_overfit_json(report: Mapping[str, object]) -> str:
    return json.dumps(report, sort_keys=True, separators=(",", ":"))


def validate_ticket36_loss_micro_overfit_report(report: Mapping[str, object]) -> None:
    if not isinstance(report, Mapping) or report.get("schema") != SCHEMA:
        raise ValueError("Ticket 36 loss micro-overfit schema mismatch")
    if report.get("overall") != "PASS":
        raise ValueError("Ticket 36 loss micro-overfit gate is not PASS")
    if report.get("execution") != {"device": "cpu", "cnn": False, "gpu": False, "images": False}:
        raise ValueError("Ticket 36 loss micro-overfit execution boundary drift")
    rows = report.get("per_case")
    if not isinstance(rows, list) or tuple(row.get("case_id") for row in rows) != CASE_IDS:
        raise ValueError("Ticket 36 loss micro-overfit case coverage drift")
    for row in rows:
        gates = row.get("gates")
        if not isinstance(gates, Mapping) or gates.get("overall") != "PASS":
            raise ValueError(f"Ticket 36 loss micro-overfit case failed: {row.get('case_id')!r}")
        if gates.get("instance_recall") != 1.0 or gates.get("unmatched") != 0 or gates.get("merged") != 0:
            raise ValueError(f"Ticket 36 loss micro-overfit matching gate failed: {row.get('case_id')!r}")
        if gates.get("far_negative_active_response") != 0 or gates.get("normal_active_response") != 0:
            raise ValueError(f"Ticket 36 loss micro-overfit negative gate failed: {row.get('case_id')!r}")
        if float(gates.get("core_response_min", -1)) < THRESHOLD:
            raise ValueError(f"Ticket 36 loss micro-overfit core gate failed: {row.get('case_id')!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)
    report = build_ticket36_loss_micro_overfit_report()
    args.output.write_text(canonical_ticket36_loss_micro_overfit_json(report) + "\n", encoding="utf-8")
    print(report["overall"])
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CASE_IDS",
    "REPORT_PATH",
    "build_ticket36_loss_micro_overfit_report",
    "canonical_ticket36_loss_micro_overfit_json",
    "validate_ticket36_loss_micro_overfit_report",
]
