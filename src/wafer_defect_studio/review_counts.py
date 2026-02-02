"""Derive current image annotation and review counts without persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .review import _explicit_annotation_cells, _load_review_context, load_review_state


@dataclass(frozen=True)
class ReviewCounts:
    """Counts shown for one Wafer Image's current review state."""

    labeled: int
    unreviewed: int
    derived_normal: int
    excluded: int


def load_review_counts(project_path: str | Path, image_asset_id: str) -> ReviewCounts:
    """Derive annotation, review, and Effective Wafer Area counts for an image."""

    context = _load_review_context(project_path, image_asset_id)
    explicit = _explicit_annotation_cells(context.project_path, image_asset_id)
    labeled = sum((grid.row, grid.column) in explicit for grid in context.grids)
    unlabeled = len(context.grids) - labeled
    reviewed = load_review_state(context.project_path, image_asset_id).reviewed
    return ReviewCounts(
        labeled=labeled,
        unreviewed=0 if reviewed else unlabeled,
        derived_normal=unlabeled if reviewed else 0,
        excluded=len(context.all_grids) - len(context.grids),
    )
