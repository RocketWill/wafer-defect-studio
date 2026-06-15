"""Read-only dense truth-component response census for Ticket 38."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from docs.demo.ticket36_single_image_cnn_micro_overfit import build_ticket36_cases
from docs.demo.ticket37_existing_artifact_report import (
    FROZEN_ARTIFACT_SEAL,
    validate_ticket37_frozen_artifact_seal,
)
from docs.demo.ticket37_proposal_evaluation import (
    _neighbor_offsets,
    build_rendered_truth_components,
)
from docs.demo.ticket37_sparse_objective_candidate import (
    CANDIDATE_ARTIFACT_ROOT,
    _read_float32_maps_artifact,
    _read_json_artifact,
    seal_ticket37_artifacts,
)


SCHEMA = "ticket38-dense-response-census.v1"
OUTPUT_PATH = Path("docs/demo/ticket38-dense-response-census.json")
TICKET37_REPORT_PATH = Path("docs/demo/ticket37-sparse-objective-candidate.json")
TICKET37_REPORT_SHA256 = "9e40f6527bf39697bf5d1172f2d278ef6d5993424782c09282a7628bd719759e"
THRESHOLD = 0.5
TOLERANCE_PIXELS = 8
PATCH_STRIDE = 64
DENSE_CASE_IDS = (
    "ticket35-density-101-train-scratch-150-000-05",
    "ticket35-density-101-train-particle-000-150-10",
)


def build_dense_case_response_census(
    response: np.ndarray,
    rendered_truth: Mapping[str, object],
    matching: Mapping[str, object],
    *,
    case_id: str,
    class_code: str,
    threshold: float = THRESHOLD,
    tolerance_pixels: int = TOLERANCE_PIXELS,
    patch_stride: int = PATCH_STRIDE,
) -> dict[str, object]:
    """Describe component response and Proposal fragmentation for one case."""

    values = np.asarray(response)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("response must be a finite floating-point 2-D array")
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite in [0,1]")
    if isinstance(tolerance_pixels, bool) or not isinstance(tolerance_pixels, Integral) or tolerance_pixels < 0:
        raise ValueError("tolerance_pixels must be a non-negative integer")
    if isinstance(patch_stride, bool) or not isinstance(patch_stride, Integral) or patch_stride <= 0:
        raise ValueError("patch_stride must be a positive integer")
    if not isinstance(case_id, str) or not case_id or not isinstance(class_code, str) or not class_code:
        raise ValueError("case_id and class_code must be non-empty strings")

    components = rendered_truth.get("truth_components")
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
        raise ValueError("rendered_truth.truth_components must be a sequence")
    if matching.get("truth_component_count") != len(components):
        raise ValueError("matching truth component count drift")
    matched_pairs = matching.get("matched_pairs")
    proposals = matching.get("proposal_components")
    if not isinstance(matched_pairs, Sequence) or not isinstance(proposals, Sequence):
        raise ValueError("matching pairs and proposal components must be sequences")
    matched_ids = {int(pair["truth_component_id"]) for pair in matched_pairs}
    touching: dict[int, list[int]] = {index: [] for index in range(len(components))}
    for proposal in proposals:
        proposal_id = int(proposal["proposal_id"])
        for component_id in proposal["truth_component_ids"]:
            touching[int(component_id)].append(proposal_id)

    rows: list[dict[str, object]] = []
    for expected_id, component in enumerate(components):
        if not isinstance(component, Mapping) or component.get("component_id") != expected_id:
            raise ValueError("truth component IDs must be contiguous")
        pixels = _component_pixels(component, values.shape)
        core_peak, core_xy = _peak(values, pixels)
        tolerance_cells = _tolerance_cells(pixels, values.shape, int(tolerance_pixels))
        tolerance_peak, tolerance_xy = _peak(values, tolerance_cells)
        proposal_ids = sorted(touching[expected_id])
        xs = [point[0] for point in pixels]
        ys = [point[1] for point in pixels]
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2
        state = (
            "core_active"
            if core_peak >= threshold
            else "tolerance_only"
            if tolerance_peak >= threshold
            else "below_threshold"
        )
        rows.append(
            {
                "component_id": expected_id,
                "core_peak": core_peak,
                "core_peak_source_xy": list(core_xy),
                "tolerance_peak": tolerance_peak,
                "tolerance_peak_source_xy": list(tolerance_xy),
                "threshold_state": state,
                "matched": expected_id in matched_ids,
                "touching_proposal_ids": proposal_ids,
                "fragmentation_count": len(proposal_ids),
                "truth_bbox_center_source_xy": [center_x, center_y],
                "truth_bbox_center_patch_phase_xy": [center_x % patch_stride, center_y % patch_stride],
            }
        )

    summary = {
        "truth_component_count": len(rows),
        "proposal_count": int(matching["proposal_count"]),
        "matched_count": int(matching["matched_count"]),
        "core_active_count": sum(row["threshold_state"] == "core_active" for row in rows),
        "tolerance_only_count": sum(row["threshold_state"] == "tolerance_only" for row in rows),
        "below_threshold_count": sum(row["threshold_state"] == "below_threshold" for row in rows),
        "unmatched_component_count": sum(not row["matched"] for row in rows),
        "fragmented_component_count": sum(row["fragmentation_count"] > 1 for row in rows),
        "fragment_excess_proposal_count": sum(max(0, row["fragmentation_count"] - 1) for row in rows),
    }
    return {
        "case_id": case_id,
        "class_code": class_code,
        "threshold": float(threshold),
        "tolerance_pixels": int(tolerance_pixels),
        "patch_stride": int(patch_stride),
        "summary": summary,
        "components": rows,
    }


def build_ticket38_dense_response_census(*, repo_root: Path | None = None) -> dict[str, object]:
    """Build the fixed two-case census from Ticket 37 sealed artifacts."""

    root = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root).expanduser().resolve()
    report_path = root / TICKET37_REPORT_PATH
    if hashlib.sha256(report_path.read_bytes()).hexdigest() != TICKET37_REPORT_SHA256:
        raise ValueError("Ticket 37 tracked report SHA-256 drift")
    ticket37_report = _read_json_artifact(report_path)
    if ticket37_report.get("overall") != "FAIL" or ticket37_report.get("recommendation") != "cam_v2":
        raise ValueError("Ticket 37 decision drift")

    candidate_root = root / CANDIDATE_ARTIFACT_ROOT
    observed_seal = seal_ticket37_artifacts(candidate_root, repo_root=root)
    validate_ticket37_frozen_artifact_seal(observed_seal)
    maps = _read_float32_maps_artifact(candidate_root / "confidence-maps.npz")
    stored_matching = _read_json_artifact(candidate_root / "proposal-matching.json")
    cases = {case.case_id: case for case in build_ticket36_cases()}
    per_case = []
    for case_id in DENSE_CASE_IDS:
        case = cases[case_id]
        class_code = case.composition
        class_index = ("scratch", "particle").index(class_code)
        truth = build_rendered_truth_components(
            tuple(item for item in case.instances if item.defect.class_code == class_code),
            maps[case_id].shape[:2],
        )
        per_case.append(
            build_dense_case_response_census(
                maps[case_id][..., class_index],
                truth,
                stored_matching["per_case"][case_id][class_code],
                case_id=case_id,
                class_code=class_code,
            )
        )
    return {
        "schema": SCHEMA,
        "source": {
            "ticket37_report": TICKET37_REPORT_PATH.as_posix(),
            "ticket37_report_sha256": TICKET37_REPORT_SHA256,
            "ticket37_artifact_seal": FROZEN_ARTIFACT_SEAL,
        },
        "case_order": list(DENSE_CASE_IDS),
        "settings": {
            "threshold": THRESHOLD,
            "tolerance_pixels": TOLERANCE_PIXELS,
            "patch_stride": PATCH_STRIDE,
        },
        "per_case": per_case,
    }


def canonical_ticket38_dense_response_census_json(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    expected = (root / OUTPUT_PATH).resolve()
    try:
        if output != expected:
            raise ValueError("Ticket 38 census output path must remain fixed")
        if output.exists():
            raise FileExistsError("Ticket 38 census output already exists")
        census = build_ticket38_dense_response_census(repo_root=root)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_ticket38_dense_response_census_json(census) + "\n")
    except (OSError, TypeError, ValueError) as error:
        print(f"Ticket 38 dense response census failed: {error}", file=sys.stderr)
        return 1
    print(f"Ticket 38 dense response census: {output}", flush=True)
    return 0


def _component_pixels(component: Mapping[str, object], shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    pixels = component.get("pixels")
    if not isinstance(pixels, Sequence) or not pixels:
        raise ValueError("truth component pixels must be non-empty")
    height, width = shape
    result = []
    for value in pixels:
        if not isinstance(value, Sequence) or len(value) != 2:
            raise ValueError("truth component pixels must be x/y pairs")
        x, y = value
        if any(isinstance(item, bool) or not isinstance(item, Integral) for item in (x, y)):
            raise ValueError("truth component pixels must be integer x/y pairs")
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError("truth component pixel outside response")
        result.append((int(x), int(y)))
    return tuple(result)


def _tolerance_cells(
    pixels: Sequence[tuple[int, int]], shape: tuple[int, int], tolerance: int
) -> tuple[tuple[int, int], ...]:
    height, width = shape
    cells = {
        (x + offset_x, y + offset_y)
        for x, y in pixels
        for offset_x, offset_y, _distance in _neighbor_offsets(tolerance)
        if 0 <= x + offset_x < width and 0 <= y + offset_y < height
    }
    return tuple(sorted(cells, key=lambda point: (point[1], point[0])))


def _peak(values: np.ndarray, pixels: Sequence[tuple[int, int]]) -> tuple[float, tuple[int, int]]:
    ranked = tuple(sorted(pixels, key=lambda point: (point[1], point[0])))
    scores = np.asarray([values[y, x] for x, y in ranked])
    index = int(np.argmax(scores))
    return float(scores[index]), ranked[index]


if __name__ == "__main__":
    raise SystemExit(main())
