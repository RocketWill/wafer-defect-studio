"""Score-separation diagnostics for Ticket 32 validation maps."""

from __future__ import annotations

import hashlib
import json
from math import ceil
from pathlib import Path

import numpy as np

from docs.demo.ticket31_development_corpus import build_ticket31_development_corpus
from wafer_defect_studio.grid_geometry import annotation_grids


SCHEMA = "ticket33-score-separation.v1"
TOP_FRACTION = 0.01


def analyze_ticket32_score_separation(
    report_path: Path,
    evidence_root: Path,
) -> dict[str, object]:
    """Measure asserted-vs-Normal Grid separation without using final members."""

    ticket32 = json.loads(report_path.read_text(encoding="utf-8"))
    cases = build_ticket31_development_corpus()
    per_seed = {}
    for seed in ticket32["development_seeds"]:
        seed_key = str(seed)
        artifact = ticket32["per_seed"][seed_key]["artifacts"]["maps"]
        maps_path = evidence_root / artifact["path"]
        payload = maps_path.read_bytes()
        if len(payload) != artifact["bytes"] or hashlib.sha256(payload).hexdigest() != artifact["sha256"]:
            raise ValueError(f"Ticket 32 seed {seed} maps artifact integrity mismatch")
        with np.load(maps_path) as maps_payload:
            maps = maps_payload["maps"]
            filenames = tuple(str(value) for value in maps_payload["filenames"])
            class_codes = tuple(str(value) for value in maps_payload["class_codes"])
        validation = {
            case.filename: case
            for case in cases
            if case.seed == seed and case.split == "validation"
        }
        if tuple(validation) != filenames or class_codes != tuple(ticket32["class_codes"]):
            raise ValueError(f"Ticket 32 seed {seed} validation map membership mismatch")
        per_seed[seed_key] = _seed_separation(maps, filenames, validation, class_codes)
    return {
        "schema": SCHEMA,
        "source_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "per_seed": per_seed,
    }


def _seed_separation(maps, filenames, cases, class_codes):
    grids = annotation_grids(1536, 1536, 512, 512)
    rows = {}
    for class_index, code in enumerate(class_codes):
        asserted_scores = []
        normal_scores = []
        for image_index, filename in enumerate(filenames):
            truth = cases[filename].oracle.grid_truth(grids)
            for grid in grids:
                values = maps[
                    image_index,
                    grid.y : grid.y + grid.height,
                    grid.x : grid.x + grid.width,
                    class_index,
                ].reshape(-1)
                score = _top_fraction_mean(values)
                (asserted_scores if code in truth[(grid.row, grid.column)] else normal_scores).append(score)
        weakest_asserted = min(asserted_scores)
        hardest_normal = max(normal_scores)
        margin = weakest_asserted - hardest_normal
        rows[code] = {
            "top_fraction": TOP_FRACTION,
            "asserted_grid_count": len(asserted_scores),
            "normal_grid_count": len(normal_scores),
            "weakest_asserted_grid_top_score": weakest_asserted,
            "hardest_normal_grid_top_score": hardest_normal,
            "margin": margin,
            "collapsed": margin <= 0.0,
        }
    return rows


def _top_fraction_mean(values: np.ndarray) -> float:
    count = max(1, ceil(values.size * TOP_FRACTION))
    return float(np.partition(values, values.size - count)[-count:].mean())


__all__ = ["SCHEMA", "analyze_ticket32_score_separation"]
