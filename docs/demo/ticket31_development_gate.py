"""Real CUDA development gate for Ticket 31 Spatial MIL v5."""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import threading
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PySide6.QtGui import QImage

from docs.demo.ticket30_map_thresholds import calibrate_map_thresholds
from docs.demo.ticket31_contract import DEVELOPMENT_SEEDS, DEVELOPMENT_TARGETS, V5_POLICY
from docs.demo.ticket31_development_corpus import (
    DevelopmentCase,
    build_ticket31_development_corpus,
    render_ticket31_development_pixels,
)
from docs.demo.wafer_quality_evidence import WaferEvidenceCase, compute_wafer_quality_evidence
from wafer_defect_studio.detection_windows import Rect, enumerate_inference_windows
from wafer_defect_studio.grid_geometry import annotation_grids
from wafer_defect_studio.model_registry import load_project_checkpoint, spatial_logits_to_probabilities
from wafer_defect_studio.normalization import NormalizationBounds
from wafer_defect_studio.training_dataset import enumerate_model_patch_rects, extract_model_patch
from wafer_defect_studio.training_input_bundle import (
    TrainingBundleSource,
    TrainingInputBundle,
    TrainingPatchBag,
)
from wafer_defect_studio.training_protocol import TerminalMessage, TrainingConfig, TrainingRequest, decode_message
from wafer_defect_studio.training_worker import run_worker


CLASS_CODES = ("scratch", "particle")
SCHEMA = "ticket31-development-gate.v1"


def select_validation_spatial_epoch(
    candidates: Sequence[Mapping[str, object]],
    class_codes: Sequence[str],
) -> dict[str, object]:
    """Select the strongest validation spatial evidence, then earliest epoch."""

    codes = tuple(class_codes)
    if not candidates or not codes:
        raise ValueError("validation spatial candidates and class_codes must not be empty")
    ranked = []
    seen_epochs = set()
    for candidate in candidates:
        epoch = candidate.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1 or epoch in seen_epochs:
            raise ValueError("validation spatial candidate epochs must be unique positive integers")
        seen_epochs.add(epoch)
        if candidate.get("source_split") != "validation":
            raise ValueError("checkpoint selection requires validation spatial metrics")
        per_class = candidate.get("per_class")
        if not isinstance(per_class, Mapping) or tuple(per_class) != codes:
            raise ValueError("validation spatial candidate class order drift")
        rows = tuple(per_class[code] for code in codes)
        values = tuple(_selection_values(row) for row in rows)
        feasible = all(_row_result(row) == "PASS" for row in rows)
        key = (
            feasible,
            min(value[0] for value in values),
            min(min(value[1], value[2]) for value in values),
            min(value[1] for value in values),
            min(value[2] for value in values),
            -max(value[3] for value in values),
            -max(value[4] for value in values),
            -epoch,
        )
        ranked.append((key, candidate))
    selected = dict(max(ranked, key=lambda item: item[0])[1])
    selected["selection_source"] = "validation_spatial_metrics"
    return selected


def _selection_values(row: object) -> tuple[float, float, float, float, float]:
    if not isinstance(row, Mapping):
        raise ValueError("validation spatial class metrics must be mappings")
    names = (
        "defect_coverage_recall", "grid_precision", "grid_recall",
        "normal_grid_leak_rate", "asserted_grid_occupancy_p95",
    )
    values = tuple(row.get(name) for name in names)
    if any(
        isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value))
        for value in values
    ):
        raise ValueError("validation spatial class metrics must be finite numbers")
    return tuple(float(value) for value in values)


def run_development_gate(output_root: Path) -> dict[str, object]:
    """Train and evaluate all frozen development seeds on real CUDA."""

    if not torch.cuda.is_available():
        raise RuntimeError("Ticket 31 development gate requires CUDA")
    device_name = torch.cuda.get_device_name(0)
    corpus = build_ticket31_development_corpus()
    seed_rows = {}
    for seed in DEVELOPMENT_SEEDS:
        run_dir = output_root / str(seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        cases = tuple(case for case in corpus if case.seed == seed)
        bundle_path = run_dir / "training-input-bundle.json"
        config_path = run_dir / "configuration.json"
        config = _training_config(seed)
        config_path.write_text(_canonical(config.to_dict()), encoding="utf-8")
        _materialize_bundle(cases, config, run_dir / "sources").write_text(
            bundle_path
        )
        checkpoint = _train(config, bundle_path, run_dir / "train")
        validation = tuple(case for case in cases if case.split == "validation")
        evidence_cases = tuple(
            WaferEvidenceCase(
                case.filename,
                "validation",
                case.oracle,
                annotation_grids(1536, 1536, 512, 512),
                _score_source(checkpoint, render_ticket31_development_pixels(case)),
            )
            for case in validation
        )
        threshold_payload = calibrate_map_thresholds(
            evidence_cases, CLASS_CODES, "absolute_spatial_probability"
        )
        thresholds = {
            code: float(threshold_payload["selected"][code]["threshold"])
            for code in CLASS_CODES
        }
        metrics = compute_wafer_quality_evidence(evidence_cases, CLASS_CODES, thresholds)
        threshold_path = run_dir / "thresholds.json"
        metrics_path = run_dir / "metrics.json"
        maps_path = run_dir / "validation-maps.npz"
        threshold_path.write_text(_canonical(threshold_payload), encoding="utf-8")
        metrics_path.write_text(_canonical(metrics), encoding="utf-8")
        np.savez_compressed(
            maps_path,
            maps=np.stack([case.absolute_maps.astype(np.float16) for case in evidence_cases]),
            filenames=np.asarray([case.filename for case in validation]),
            pixel_sha256=np.asarray([case.pixel_sha256 for case in validation]),
            class_codes=np.asarray(CLASS_CODES),
        )
        seed_rows[str(seed)] = {
            "classes": {
                code: {**metrics["per_class"][code], "overall": _row_result(metrics["per_class"][code])}
                for code in CLASS_CODES
            },
            "artifacts": {
                name: _artifact(path, output_root)
                for name, path in (
                    ("configuration", config_path),
                    ("corpus", bundle_path),
                    ("checkpoint", checkpoint),
                    ("thresholds", threshold_path),
                    ("maps", maps_path),
                    ("metrics", metrics_path),
                )
            },
        }
    overall = "PASS" if all(
        row["overall"] == "PASS"
        for seed in seed_rows.values()
        for row in seed["classes"].values()
    ) else "FAIL"
    return {
        "schema": SCHEMA,
        "overall": overall,
        "recommendation": "spatial_mil_v5" if overall == "PASS" else "cam_v2",
        "spatial_mil_v5_status": "development_pass" if overall == "PASS" else "experimental",
        "device": {"name": device_name, "torch": torch.__version__, "cuda": torch.version.cuda},
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "class_codes": list(CLASS_CODES),
        "targets": dict(DEVELOPMENT_TARGETS),
        "policy": dict(V5_POLICY),
        "per_seed": seed_rows,
        "claims": ["approximate localization", "not segmentation", "not production accuracy", "not Neurocle equivalence"],
    }


class _BundleText:
    def __init__(self, bundle: TrainingInputBundle):
        self.bundle = bundle

    def write_text(self, path: Path) -> None:
        path.write_text(self.bundle.to_json() + "\n", encoding="utf-8")


def _materialize_bundle(
    cases: Sequence[DevelopmentCase], config: TrainingConfig, source_dir: Path
) -> _BundleText:
    source_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    bags = []
    grids = annotation_grids(1536, 1536, 512, 512)
    for case in cases:
        pixels = render_ticket31_development_pixels(case)
        path = source_dir / case.filename
        image = QImage(pixels.data, 1536, 1536, 1536, QImage.Format.Format_Grayscale8).copy()
        if not image.save(str(path), "PNG"):
            raise RuntimeError(f"unable to write development source: {path}")
        image_id = case.filename.removesuffix(".png")
        source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        sources.append(TrainingBundleSource(image_id, case.split, str(path), source_sha256, "uint8"))
        truth = case.oracle.grid_truth(grids)
        for grid in grids:
            rect = Rect(grid.x, grid.y, grid.width, grid.height)
            bags.append(TrainingPatchBag(
                f"{image_id}:{grid.row}:{grid.column}", image_id, grid.row, grid.column,
                enumerate_model_patch_rects(rect, config), tuple(
                    code for code in CLASS_CODES if code in truth[(grid.row, grid.column)]
                ),
            ))
    bundle = TrainingInputBundle(
        "ticket31-development", f"ticket31-seed-{config.seed}", CLASS_CODES,
        (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 0.0, 100.0),),
        tuple(sources), (), 2, tuple(bags),
    )
    return _BundleText(bundle)


def _training_config(seed: int) -> TrainingConfig:
    return TrainingConfig(
        "ticket31-development", f"ticket31-seed-{seed}", 2, 30, 4,
        device="cuda", seed=seed, learning_rate=0.0003, weights_policy="imagenet",
        patch_size=128, patch_stride=64, training_policy="spatial_mil_v5",
    )


def _train(config: TrainingConfig, bundle_path: Path, staging: Path) -> Path:
    output = queue.SimpleQueue()
    run_worker(
        TrainingRequest(f"ticket31-{config.seed}", config, staging, bundle_path),
        output, threading.Event(),
    )
    terminal = None
    while not output.empty():
        message = decode_message(output.get())
        if isinstance(message, TerminalMessage):
            terminal = message
    if terminal is None or terminal.status != "completed":
        detail = "missing terminal" if terminal is None else terminal.message
        raise RuntimeError(f"Ticket 31 seed {config.seed} training failed: {detail}")
    return staging / "model.pt"


def _score_source(checkpoint_path: Path, source: np.ndarray) -> np.ndarray:
    model, checkpoint = load_project_checkpoint(checkpoint_path, device="cuda", expected_class_codes=CLASS_CODES)
    bounds = NormalizationBounds(**checkpoint["normalization_bounds"][0])
    windows = enumerate_inference_windows(1536, 1536, 128, 64)
    local = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), 64):
            selected = windows[start:start + 64]
            inputs = torch.stack([
                extract_model_patch(source, bounds, top=window.y, left=window.x, size=128)
                for window in selected
            ]).to("cuda")
            local.append(spatial_logits_to_probabilities(model(inputs), (128, 128)).cpu().numpy())
    from wafer_defect_studio.cam_detection import generate_all_convolutional_artifact
    artifact = generate_all_convolutional_artifact(
        windows, np.concatenate(local), class_names=CLASS_CODES,
        window_settings={"window_size": [128, 128], "stride": [64, 64],
                         "reflect_padding": True, "center_weighting": "linear"},
        provenance={"checkpoint_format": "wafer_defect_studio.resnet18.v5", "feature_stride": 2},
        model_id="spatial_mil_v5", profile_id="ticket31-development",
        evaluation_id="ticket31-validation", map_method="spatial_mil_sigmoid",
    )
    return artifact.maps


def _row_result(row: Mapping[str, object]) -> str:
    return "PASS" if (
        row["defect_coverage_recall"] is not None and row["defect_coverage_recall"] >= DEVELOPMENT_TARGETS["defect_coverage_recall"]
        and row["grid_precision"] is not None and row["grid_precision"] >= DEVELOPMENT_TARGETS["grid_precision"]
        and row["grid_recall"] is not None and row["grid_recall"] >= DEVELOPMENT_TARGETS["grid_recall"]
        and row["normal_grid_leak_rate"] is not None and row["normal_grid_leak_rate"] <= DEVELOPMENT_TARGETS["normal_grid_leak_rate"]
        and row["asserted_grid_occupancy_p95"] is not None and row["asserted_grid_occupancy_p95"] <= DEVELOPMENT_TARGETS["asserted_grid_occupancy_p95"]
    ) else "FAIL"


def _artifact(path: Path, root: Path) -> dict[str, object]:
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _canonical(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run_development_gate(args.output_root.resolve())
    args.report.resolve().write_text(_canonical(report), encoding="utf-8")
    print(f"Ticket 31 development gate: {report['overall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLASS_CODES", "SCHEMA", "run_development_gate",
    "select_validation_spatial_epoch",
]
