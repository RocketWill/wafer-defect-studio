"""Grid-level evidence for the generated Phase 2 Demo."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def compute_grid_quality_evidence(
    class_codes: Sequence[str],
    grid_evaluation: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Join test Grid Evaluation with Detection-only coarse localization."""

    classes = tuple(class_codes)
    values = tuple(rows)
    support = {
        code: {
            split: len(
                {
                    row["image_id"]
                    for row in values
                    if row["split"] == split and code in row["asserted"]
                }
            )
            for split in ("validation", "test")
        }
        for code in classes
    }
    per_class = {}
    for code in classes:
        asserted = tuple(row for row in values if code in row["asserted"])
        normal = tuple(row for row in values if code not in row["asserted"])
        intersections = sum(code in row["retained"] for row in asserted)
        leaks = sum(code in row["retained"] for row in normal)
        per_class[code] = {
            "asserted_grid_intersections": intersections,
            "asserted_grids": len(asserted),
            "intersection_rate": 0.0 if not asserted else intersections / len(asserted),
            "normal_grid_leaks": leaks,
            "normal_grids": len(normal),
            "normal_grid_leak_rate": 0.0 if not normal else leaks / len(normal),
        }
    sufficient = all(
        support[code][split] >= 2
        for code in classes
        for split in ("validation", "test")
    )
    return {
        "evidence_status": "measured" if sufficient else "insufficient_evidence",
        "grid_evaluation": dict(grid_evaluation),
        "coarse_localization": {
            "per_class": per_class,
            "asserted_image_support": support,
        },
    }


def build_comparison_report(
    split_id: str,
    cam_v2: Mapping[str, Any],
    patch_v3: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind both measured model reports to one persisted Dataset Split."""

    return {
        "split_id": split_id,
        "models": {
            "cam_v2": {"split_id": split_id, **cam_v2},
            "patch_v3": {"split_id": split_id, **patch_v3},
        },
        "limitations": [
            "Generated Demo data is not a production accuracy claim.",
            "Localization is approximate Grid evidence, not pixel segmentation.",
        ],
    }


__all__ = ["build_comparison_report", "compute_grid_quality_evidence"]
