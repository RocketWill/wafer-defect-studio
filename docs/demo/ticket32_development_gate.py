"""Repaired Spatial MIL v5 development replay for Ticket 32."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from docs.demo.ticket30_map_thresholds import calibrate_map_thresholds
from docs.demo.ticket31_contract import DEVELOPMENT_SEEDS, DEVELOPMENT_TARGETS, V5_POLICY
from docs.demo.ticket31_development_corpus import (
    build_ticket31_development_corpus,
    render_ticket31_development_pixels,
)
from docs.demo.ticket31_development_gate import (
    CLASS_CODES,
    _artifact,
    _canonical,
    _materialize_bundle,
    _row_result,
    _score_sources,
    _train,
    select_validation_spatial_epoch,
)
from docs.demo.wafer_quality_evidence import WaferEvidenceCase, compute_wafer_quality_evidence
from wafer_defect_studio.grid_geometry import annotation_grids
from wafer_defect_studio.training_protocol import TrainingConfig
from wafer_defect_studio.training_run import validate_project_checkpoint


SCHEMA = "ticket32-development-gate.v1"


def run_repaired_development_gate(output_root: Path) -> dict[str, object]:
    """Replay every retained epoch and select only by validation spatial quality."""

    if not torch.cuda.is_available():
        raise RuntimeError("Ticket 32 development gate requires CUDA")
    corpus = build_ticket31_development_corpus()
    seed_rows = {}
    for seed in DEVELOPMENT_SEEDS:
        run_dir = output_root / str(seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        cases = tuple(case for case in corpus if case.seed == seed)
        validation = tuple(case for case in cases if case.split == "validation")
        config = _training_config(seed)
        config_path = run_dir / "configuration.json"
        bundle_path = run_dir / "training-input-bundle.json"
        config_path.write_text(_canonical(config.to_dict()), encoding="utf-8")
        _materialize_bundle(cases, config, run_dir / "sources").write_text(bundle_path)
        checkpoint_path = run_dir / "train" / "model.pt"
        epoch_dir = run_dir / "train" / "epoch-states"
        partial_path = run_dir / "epoch-evidence.partial.json"
        candidates = (
            json.loads(partial_path.read_text(encoding="utf-8"))["epochs"]
            if partial_path.is_file()
            else []
        )
        existing_selection = None
        if checkpoint_path.is_file():
            existing_selection = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            ).get("checkpoint_selection")
        already_selected = (
            isinstance(existing_selection, dict)
            and existing_selection.get("metric") == "validation_spatial_metrics"
            and len(candidates) == 30
        )
        if not already_selected and not (
            checkpoint_path.is_file()
            and epoch_dir.is_dir()
            and len(tuple(epoch_dir.glob("epoch-*.pt"))) == 30
        ):
            checkpoint_path = _train(config, bundle_path, run_dir / "train")

        for epoch in range(len(candidates) + 1, 31) if not already_selected else ():
            evidence = _score_epoch(
                checkpoint_path,
                run_dir / "train" / "epoch-states" / f"epoch-{epoch:03d}.pt",
                validation,
            )
            threshold_payload = calibrate_map_thresholds(
                evidence, CLASS_CODES, "absolute_spatial_probability"
            )
            thresholds = {
                code: float(threshold_payload["selected"][code]["threshold"])
                for code in CLASS_CODES
            }
            metrics = compute_wafer_quality_evidence(evidence, CLASS_CODES, thresholds)
            candidates.append({
                "epoch": epoch,
                "source_split": "validation",
                "per_class": metrics["per_class"],
                "thresholds": thresholds,
            })
            partial_path.write_text(_canonical({
                "schema": "ticket32-epoch-evidence.partial.v1",
                "seed": seed,
                "epochs": candidates,
            }), encoding="utf-8")
            print(f"Ticket 32 seed {seed}: replayed epoch {epoch}/30", flush=True)
        selected = select_validation_spatial_epoch(candidates, CLASS_CODES)
        selected_epoch = int(selected["epoch"])
        if not already_selected:
            _publish_selected_checkpoint(
                checkpoint_path,
                run_dir / "train" / "epoch-states" / f"epoch-{selected_epoch:03d}.pt",
                selected,
            )
            shutil.rmtree(run_dir / "train" / "epoch-states")

        selected_maps = _score_sources(
            checkpoint_path,
            tuple(render_ticket31_development_pixels(case) for case in validation),
        )
        selected_evidence = tuple(
            WaferEvidenceCase(
                case.filename, "validation", case.oracle,
                annotation_grids(1536, 1536, 512, 512),
                absolute_maps,
            )
            for case, absolute_maps in zip(validation, selected_maps, strict=True)
        )
        threshold_payload = calibrate_map_thresholds(
            selected_evidence, CLASS_CODES, "absolute_spatial_probability"
        )
        thresholds = {
            code: float(threshold_payload["selected"][code]["threshold"])
            for code in CLASS_CODES
        }
        metrics = compute_wafer_quality_evidence(
            selected_evidence, CLASS_CODES, thresholds
        )
        epoch_evidence_path = run_dir / "epoch-evidence.json"
        threshold_path = run_dir / "thresholds.json"
        metrics_path = run_dir / "metrics.json"
        maps_path = run_dir / "validation-maps.npz"
        epoch_evidence_path.write_text(_canonical({
            "schema": "ticket32-epoch-evidence.v1",
            "selection_source": "validation_spatial_metrics",
            "selected_epoch": selected_epoch,
            "epochs": candidates,
        }), encoding="utf-8")
        partial_path.unlink(missing_ok=True)
        threshold_path.write_text(_canonical(threshold_payload), encoding="utf-8")
        metrics_path.write_text(_canonical(metrics), encoding="utf-8")
        np.savez_compressed(
            maps_path,
            maps=np.stack([item.absolute_maps.astype(np.float16) for item in selected_evidence]),
            filenames=np.asarray([case.filename for case in validation]),
            pixel_sha256=np.asarray([case.pixel_sha256 for case in validation]),
            class_codes=np.asarray(CLASS_CODES),
        )
        _rewrite_training_manifest(run_dir / "train")
        validate_project_checkpoint(checkpoint_path, expected_class_codes=CLASS_CODES)
        seed_rows[str(seed)] = {
            "selected_epoch": selected_epoch,
            "classes": {
                code: {**metrics["per_class"][code], "overall": _row_result(metrics["per_class"][code])}
                for code in CLASS_CODES
            },
            "artifacts": {
                name: _artifact(path, output_root)
                for name, path in (
                    ("configuration", config_path), ("corpus", bundle_path),
                    ("epoch_evidence", epoch_evidence_path),
                    ("checkpoint", checkpoint_path), ("thresholds", threshold_path),
                    ("maps", maps_path), ("metrics", metrics_path),
                )
            },
        }
    overall = "PASS" if all(
        row["overall"] == "PASS"
        for seed in seed_rows.values() for row in seed["classes"].values()
    ) else "FAIL"
    return {
        "schema": SCHEMA,
        "overall": overall,
        "recommendation": "spatial_mil_v5" if overall == "PASS" else "cam_v2",
        "spatial_mil_v5_status": "development_pass" if overall == "PASS" else "experimental",
        "device": {"name": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "class_codes": list(CLASS_CODES),
        "targets": dict(DEVELOPMENT_TARGETS),
        "policy": dict(V5_POLICY),
        "per_seed": seed_rows,
        "ticket31_report_sha256": hashlib.sha256(
            Path("docs/demo/ticket31-development-gate.json").read_bytes()
        ).hexdigest(),
        "claims": ["approximate localization", "not segmentation", "not production accuracy", "not Neurocle equivalence"],
    }


def _training_config(seed: int) -> TrainingConfig:
    return TrainingConfig(
        "ticket32-development", f"ticket32-seed-{seed}", 2, 30, 4,
        device="cuda", seed=seed, learning_rate=0.0003, weights_policy="imagenet",
        patch_size=128, patch_stride=64, training_policy="spatial_mil_v5",
        retain_epoch_states=True,
    )


def _score_epoch(checkpoint_path: Path, state_path: Path, cases) -> tuple[WaferEvidenceCase, ...]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["state_dict"] = torch.load(state_path, map_location="cpu", weights_only=True)
    temporary = state_path.parent / "replay-checkpoint.pt"
    torch.save(checkpoint, temporary)
    try:
        maps = _score_sources(
            temporary,
            tuple(render_ticket31_development_pixels(case) for case in cases),
        )
        return tuple(
            WaferEvidenceCase(
                case.filename, "validation", case.oracle,
                annotation_grids(1536, 1536, 512, 512),
                absolute_maps,
            )
            for case, absolute_maps in zip(cases, maps, strict=True)
        )
    finally:
        temporary.unlink(missing_ok=True)


def _publish_selected_checkpoint(checkpoint_path: Path, state_path: Path, selected) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["state_dict"] = torch.load(state_path, map_location="cpu", weights_only=True)
    checkpoint["checkpoint_selection"] = {
        "source": "validation",
        "metric": "validation_spatial_metrics",
        "selected_epoch": int(selected["epoch"]),
        "per_class": selected["per_class"],
    }
    torch.save(checkpoint, checkpoint_path)
    metrics_path = checkpoint_path.parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update({
        "checkpoint_epoch": int(selected["epoch"]),
        "checkpoint_selection_metric": "validation_spatial_metrics",
        "checkpoint_selection_value": None,
        "selected_validation_spatial_metrics": selected["per_class"],
    })
    metrics_path.write_text(_canonical(metrics), encoding="utf-8")


def _rewrite_training_manifest(train_dir: Path) -> None:
    files = [
        {"path": name, "sha256": hashlib.sha256((train_dir / name).read_bytes()).hexdigest()}
        for name in ("model.pt", "metrics.json")
    ]
    (train_dir / "manifest.json").write_text(_canonical({
        "required_files": [item["path"] for item in files], "files": files,
    }), encoding="utf-8")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run_repaired_development_gate(args.output_root.resolve())
    args.report.resolve().write_text(_canonical(report), encoding="utf-8")
    print(f"Ticket 32 development gate: {report['overall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "run_repaired_development_gate"]
