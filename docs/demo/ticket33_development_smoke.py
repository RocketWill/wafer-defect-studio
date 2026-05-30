"""Seed-101 validation smoke gate for grid-contrastive Spatial MIL v6."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from docs.demo.ticket30_map_thresholds import calibrate_map_thresholds
from docs.demo.ticket31_contract import DEVELOPMENT_TARGETS
from docs.demo.ticket31_development_corpus import (
    build_ticket31_development_corpus,
    render_ticket31_development_pixels,
)
from docs.demo.ticket31_development_gate import (
    CLASS_CODES,
    _canonical,
    _row_result,
    _score_source_model,
)
from docs.demo.ticket33_micro_overfit import _separation_rows, train_grid_contrastive_model
from docs.demo.wafer_quality_evidence import WaferEvidenceCase, compute_wafer_quality_evidence
from wafer_defect_studio.grid_geometry import annotation_grids
from wafer_defect_studio.training_input_bundle import NormalizationBounds


SCHEMA = "ticket33-development-smoke.v1"


def run_development_smoke(output_root: Path, *, epochs: int = 30) -> dict[str, object]:
    cases = tuple(case for case in build_ticket31_development_corpus() if case.seed == 101)
    validation = tuple(case for case in cases if case.split == "validation")
    model, losses = train_grid_contrastive_model(cases, output_root, epochs=epochs)
    bounds = NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 0.0, 100.0)
    maps = tuple(
        _score_source_model(model, bounds, render_ticket31_development_pixels(case))
        for case in validation
    )
    evidence = tuple(
        WaferEvidenceCase(
            case.filename, "validation", case.oracle,
            annotation_grids(1536, 1536, 512, 512), confidence_map,
        )
        for case, confidence_map in zip(validation, maps, strict=True)
    )
    threshold_payload = calibrate_map_thresholds(
        evidence, CLASS_CODES, "absolute_spatial_probability"
    )
    thresholds = {
        code: float(threshold_payload["selected"][code]["threshold"])
        for code in CLASS_CODES
    }
    metrics = compute_wafer_quality_evidence(evidence, CLASS_CODES, thresholds)
    separation = _separation_rows(validation, maps)
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
    np.savez_compressed(
        output_root / "validation-maps.npz",
        maps=np.stack([confidence_map.astype(np.float16) for confidence_map in maps]),
        filenames=np.asarray([case.filename for case in validation]),
        class_codes=np.asarray(CLASS_CODES),
    )
    return {
        "schema": SCHEMA,
        "seed": 101,
        "selection_source": "validation",
        "overall": "PASS" if all(row["overall"] == "PASS" for row in rows.values()) else "FAIL",
        "targets": dict(DEVELOPMENT_TARGETS),
        "thresholds": thresholds,
        "loss": {"first": losses[0], "last": losses[-1]},
        "per_class": rows,
        "device": {"name": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
        "claims": ["development smoke only", "approximate localization", "not segmentation", "not Neurocle equivalence"],
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report = run_development_smoke(output_root, epochs=args.epochs)
    args.report.resolve().write_text(_canonical(report), encoding="utf-8")
    print(f"Ticket 33 development smoke: {report['overall']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "run_development_smoke"]
