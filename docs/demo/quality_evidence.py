"""Grid-level evidence for the generated Phase 2 Demo."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def compute_grid_quality_evidence(
    class_codes: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure classification and coarse Grid localization without pixel metrics."""

    classes = tuple(class_codes)
    values = tuple(rows)
    exact = sum(
        set(row["asserted"]) == set(row["predicted"])
        for row in values
    )
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
        tp = sum(code in row["asserted"] and code in row["predicted"] for row in values)
        fp = sum(code not in row["asserted"] and code in row["predicted"] for row in values)
        fn = sum(code in row["asserted"] and code not in row["predicted"] for row in values)
        f1_denominator = 2 * tp + fp + fn
        asserted = tuple(row for row in values if code in row["asserted"])
        normal = tuple(row for row in values if code not in row["asserted"])
        intersections = sum(code in row["retained"] for row in asserted)
        leaks = sum(code in row["retained"] for row in normal)
        per_class[code] = {
            "classification": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "f1": 0.0 if f1_denominator == 0 else 2 * tp / f1_denominator,
            },
            "localization": {
                "asserted_grid_intersections": intersections,
                "asserted_grids": len(asserted),
                "intersection_rate": 0.0 if not asserted else intersections / len(asserted),
                "normal_grid_leaks": leaks,
                "normal_grids": len(normal),
                "normal_grid_leak_rate": 0.0 if not normal else leaks / len(normal),
            },
        }
    sufficient = all(
        support[code][split] >= 2
        for code in classes
        for split in ("validation", "test")
    )
    return {
        "evidence_status": "measured" if sufficient else "insufficient_evidence",
        "exact_grid_match": {
            "count": exact,
            "total": len(values),
            "rate": 0.0 if not values else exact / len(values),
        },
        "per_class": per_class,
        "asserted_image_support": support,
    }


__all__ = ["compute_grid_quality_evidence"]
