"""Single-candidate v6 development training and validation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    _REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import torch

from docs.demo.ticket30_map_thresholds import calibrate_map_thresholds
from docs.demo.ticket31_contract import DEVELOPMENT_SEEDS, DEVELOPMENT_TARGETS
from docs.demo.ticket31_development_corpus import (
    DevelopmentCase,
    build_ticket31_development_corpus,
    render_ticket31_development_pixels,
)
from docs.demo.ticket31_development_gate import CLASS_CODES, _row_result, _score_source_model
from docs.demo.ticket33_micro_overfit import _separation_rows, train_grid_contrastive_model
from docs.demo.ticket34_contract import FINAL_MEMBERS, V6_POLICY
from docs.demo.wafer_quality_evidence import WaferEvidenceCase, compute_wafer_quality_evidence
from wafer_defect_studio.grid_geometry import annotation_grids
from wafer_defect_studio.normalization import NormalizationBounds


SCHEMA = "ticket34-development-candidate.v1"
CHECKPOINT_FORMAT = "wafer_defect_studio.resnet18.spatial_mil_v6"
CANONICAL_TRAINING_SEED = DEVELOPMENT_SEEDS[0]
EPOCHS = 30
_ENVIRONMENT_FIELDS = ("python", "torch", "cuda", "gpu")


def run_development_candidate(
    output_root: Path,
    *,
    corpus: Sequence[DevelopmentCase] | None = None,
    trainer: Callable[..., tuple[object, Sequence[float]]] | None = None,
    scorer: Callable[..., np.ndarray] | None = None,
    renderer: Callable[[DevelopmentCase], np.ndarray] | None = None,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Train one frozen v6 candidate and evaluate every development validation member."""

    if trainer is None and not torch.cuda.is_available():
        raise RuntimeError("Ticket 34 development candidate requires CUDA")
    cases = tuple(build_ticket31_development_corpus() if corpus is None else corpus)
    _validate_development_membership(cases)
    train_cases = tuple(case for case in cases if case.split == "train")
    validation_cases = tuple(case for case in cases if case.split == "validation")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    trainer = _default_trainer if trainer is None else trainer
    scorer = _score_source_model if scorer is None else scorer
    renderer = render_ticket31_development_pixels if renderer is None else renderer
    model, losses = trainer(
        train_cases,
        root / "training",
        epochs=EPOCHS,
        seed=CANONICAL_TRAINING_SEED,
    )
    losses = _normalize_losses(losses)
    if len(losses) != EPOCHS:
        raise ValueError(f"candidate loss trace must contain exactly {EPOCHS} epochs")

    bounds = NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 0.0, 100.0)
    maps = []
    evidence = []
    grids = annotation_grids(1536, 1536, 512, 512)
    for case in validation_cases:
        source = renderer(case)
        absolute_map = np.asarray(scorer(model, bounds, source), dtype=np.float32)
        expected = (1536, 1536, len(CLASS_CODES))
        if absolute_map.shape != expected:
            raise ValueError(
                f"{case.filename}: validation map shape expected {expected}, actual {absolute_map.shape}"
            )
        if not np.isfinite(absolute_map).all() or absolute_map.min() < 0 or absolute_map.max() > 1:
            raise ValueError(f"{case.filename}: validation map values must be finite in [0, 1]")
        retained_map = absolute_map.astype(np.float16)
        maps.append(retained_map)
        evidence.append(WaferEvidenceCase(case.filename, "validation", case.oracle, grids, retained_map))

    threshold_payload = calibrate_map_thresholds(
        tuple(evidence), CLASS_CODES, "absolute_spatial_probability"
    )
    thresholds = {
        code: float(threshold_payload["selected"][code]["threshold"])
        for code in CLASS_CODES
    }
    metrics = compute_wafer_quality_evidence(tuple(evidence), CLASS_CODES, thresholds)
    separation = _separation_rows(validation_cases, tuple(maps))
    rows = {
        code: {
            **metrics["per_class"][code],
            "score_separation_margin": separation[code]["margin"],
            "overall": (
                "PASS"
                if _row_result(metrics["per_class"][code]) == "PASS"
                and separation[code]["margin"] > 0
                else "FAIL"
            ),
        }
        for code in CLASS_CODES
    }
    overall = "PASS" if all(row["overall"] == "PASS" for row in rows.values()) else "FAIL"

    config = _candidate_config(bounds, cases)
    _write_json(root / "candidate-config.json", config)
    _write_json(root / "validation-thresholds.json", threshold_payload)
    _write_json(
        root / "validation-metrics.json",
        {
            "schema": "ticket34-validation-metrics.v1",
            "source_split": "validation",
            "selection_source": "validation_spatial_metrics",
            "candidate_epoch": EPOCHS,
            "candidate_count": 1,
            "per_class": rows,
            "case_count": len(evidence),
        },
    )
    _write_json(
        root / "training-loss.json",
        {"schema": "ticket34-training-loss.v1", "epochs": EPOCHS, "values": losses},
    )
    np.savez_compressed(
        root / "validation-maps.npz",
        maps=np.stack(maps),
        filenames=np.asarray([case.filename for case in validation_cases]),
        pixel_sha256=np.asarray([case.pixel_sha256 for case in validation_cases]),
        class_codes=np.asarray(CLASS_CODES),
        split=np.asarray("validation"),
    )

    checkpoint = _build_checkpoint(model, config, rows, losses, thresholds)
    checkpoint_path = root / "checkpoint.pt"
    _write_new_checkpoint(checkpoint_path, checkpoint)

    artifacts = {
        name: _artifact(path, root)
        for name, path in (
            ("configuration", root / "candidate-config.json"),
            ("checkpoint", checkpoint_path),
            ("validation_thresholds", root / "validation-thresholds.json"),
            ("validation_maps", root / "validation-maps.npz"),
            ("validation_metrics", root / "validation-metrics.json"),
            ("training_loss", root / "training-loss.json"),
        )
    }
    report = {
        "schema": SCHEMA,
        "overall": overall,
        "recommendation": "run_final_held_out_gate" if overall == "PASS" else "cam_v2",
        "final_gate_eligible": overall == "PASS",
        "training_seed": CANONICAL_TRAINING_SEED,
        "epochs": EPOCHS,
        "candidate_count": 1,
        "candidate_epoch": EPOCHS,
        "selection_source": "validation_spatial_metrics",
        "membership": _membership(cases),
        "configuration": config,
        "validation_thresholds": threshold_payload,
        "validation_metrics": {
            "case_count": len(evidence),
            "per_class": rows,
            "source_split": "validation",
            "selection_source": "validation_spatial_metrics",
            "candidate_epoch": EPOCHS,
            "candidate_count": 1,
        },
        "validation_separation": separation,
        "loss": {"first": losses[0], "last": losses[-1], "values": losses},
        "environment": _normalize_environment(environment),
        "checkpoint_format": CHECKPOINT_FORMAT,
        "artifacts": artifacts,
        "claims": ["development evidence", "approximate localization", "not segmentation", "not Neurocle equivalence"],
    }
    return report


def _default_trainer(cases, output_root: Path, *, epochs: int, seed: int):
    return train_grid_contrastive_model(cases, output_root, epochs=epochs, seed=seed)


def _validate_development_membership(cases: Sequence[DevelopmentCase]) -> None:
    if not cases:
        raise ValueError("development corpus must not be empty")
    filenames = []
    seeds = {seed: {"train": 0, "validation": 0} for seed in DEVELOPMENT_SEEDS}
    for case in cases:
        filename = getattr(case, "filename", None)
        seed = getattr(case, "seed", None)
        split = getattr(case, "split", None)
        if filename in FINAL_MEMBERS:
            raise ValueError(f"final evidence member is forbidden for development candidate: {filename}")
        if not isinstance(filename, str) or not filename:
            raise ValueError("development member filename must be a non-empty string")
        if seed not in seeds or split not in {"train", "validation"}:
            raise ValueError(f"development member has unsupported seed/split: {seed!r}/{split!r}")
        if filename in filenames:
            raise ValueError(f"development member filename is duplicated: {filename}")
        filenames.append(filename)
        seeds[seed][split] += 1
    missing = [str(seed) for seed, counts in seeds.items() if not counts["train"] or not counts["validation"]]
    if missing:
        raise ValueError("development corpus must contain train and validation members for seeds: " + ",".join(missing))


def _membership(cases: Sequence[DevelopmentCase]) -> dict[str, object]:
    by_seed = {}
    for seed in DEVELOPMENT_SEEDS:
        by_seed[str(seed)] = {
            "train": [case.filename for case in cases if case.seed == seed and case.split == "train"],
            "validation": [case.filename for case in cases if case.seed == seed and case.split == "validation"],
        }
    return {
        "train": [case.filename for case in cases if case.split == "train"],
        "validation": [case.filename for case in cases if case.split == "validation"],
        "by_seed": by_seed,
    }


def _candidate_config(bounds: NormalizationBounds, cases: Sequence[DevelopmentCase]) -> dict[str, object]:
    return {
        "schema": "ticket34-development-candidate-config.v1",
        "class_codes": list(CLASS_CODES),
        "training_seed": CANONICAL_TRAINING_SEED,
        "epochs": EPOCHS,
        "architecture": "resnet18_spatial_logits_v5",
        "checkpoint_format": CHECKPOINT_FORMAT,
        "device": "cuda",
        "normalization_bounds": [asdict(bounds)],
        "input_size": {"width": 128, "height": 128},
        "patch_size": 128,
        "patch_stride": 64,
        "feature_stride": 2,
        "training_policy": dict(V6_POLICY),
        "optimizer_policy": {
            "name": "adamw",
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 5.0,
        },
        "loss_policy": {
            "positive_spatial_evidence": "top_1_percent",
            "absent_class_suppression": "dense",
            "normal_grid_ranking": "same_image",
            "sparse_probability_budget": 0.01,
            "sparse_loss_weight": 0.25,
            "overlap_loss_weight": 0.10,
        },
        "membership": _membership(cases),
        "selection_policy": {
            "source": "validation",
            "metric": "validation_spatial_metrics",
            "candidate_epoch": EPOCHS,
            "candidate_count": 1,
        },
    }


def _build_checkpoint(model, config, rows, losses, thresholds):
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise ValueError("candidate model must expose state_dict()")
    raw_state = state_dict()
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise ValueError("candidate model state_dict must be a non-empty mapping")
    cpu_state = {}
    for name, tensor in raw_state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("candidate model state_dict contains an invalid entry")
        cpu_state[name] = tensor.detach().cpu()
    return {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "architecture": config["architecture"],
        "class_count": len(CLASS_CODES),
        "class_codes": list(CLASS_CODES),
        "normalization_bounds": config["normalization_bounds"],
        "input_size": config["input_size"],
        "feature_stride": config["feature_stride"],
        "patch_size": config["patch_size"],
        "patch_stride": config["patch_stride"],
        "training_policy": config["training_policy"],
        "loss_policy": config["loss_policy"],
        "optimizer_policy": config["optimizer_policy"],
        "epochs": EPOCHS,
        "training_seed": CANONICAL_TRAINING_SEED,
        "checkpoint_selection": {
            "source": "validation",
            "metric": "validation_spatial_metrics",
            "selected_epoch": EPOCHS,
            "candidate_count": 1,
            "per_class": rows,
        },
        "thresholds": thresholds,
        "loss": {"first": losses[0], "last": losses[-1]},
        "state_dict": cpu_state,
    }


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise ValueError(f"candidate artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(value), encoding="utf-8")


def _write_new_checkpoint(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise ValueError(f"candidate artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def _normalize_losses(losses: Sequence[float]) -> list[float]:
    values = [float(value) for value in losses]
    if not values or any(not np.isfinite(value) for value in values):
        raise ValueError("candidate loss trace must contain finite values")
    return values


def _normalize_environment(environment: Mapping[str, object] | None) -> dict[str, str]:
    if environment is not None:
        if not isinstance(environment, Mapping):
            raise ValueError("candidate environment must be a mapping")
        values = {field: environment.get(field) for field in _ENVIRONMENT_FIELDS}
    else:
        values = {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda or "unavailable"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable",
        }
    if any(not isinstance(values[field], str) or not values[field] for field in _ENVIRONMENT_FIELDS):
        raise ValueError("candidate environment identity is incomplete")
    return {field: values[field] for field in _ENVIRONMENT_FIELDS}


def _artifact(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _canonical(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_development_candidate(args.output_root.resolve())
    report_path = args.report.resolve()
    if report_path.exists():
        raise ValueError(f"candidate report already exists: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_canonical(report), encoding="utf-8")
    print(f"Ticket 34 development candidate: {report['overall']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_TRAINING_SEED",
    "CHECKPOINT_FORMAT",
    "EPOCHS",
    "SCHEMA",
    "run_development_candidate",
]
