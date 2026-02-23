"""Canonical reviewed-proposal rows and deterministic CSV export.

The export seam deliberately stays independent of the database and widgets.  A
caller supplies immutable proposals plus the latest review revisions; this
module projects them into source-image coordinates without changing either
input.  JSON and image exporters can reuse :class:`ReviewedProposalRow` in
later slices.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any

from .detection_windows import Rect
from .proposal_generation import DefectProposal
from .proposal_queue import ReviewQueueItem, build_review_queue


SOURCE_COORDINATE_SYSTEM = "source-image-pixels"
APPROXIMATE_LOCALIZATION_WARNING = (
    "Approximate weak localization: proposal geometry is not a pixel-accurate segmentation mask."
)

# Keep this order stable: it is the machine-readable CSV contract.
CSV_COLUMNS = (
    "project_id",
    "image_asset_id",
    "run_id",
    "profile_id",
    "class_name",
    "threshold",
    "proposal_id",
    "source_x",
    "source_y",
    "source_width",
    "source_height",
    "area",
    "peak_confidence",
    "mean_confidence",
    "review_status",
    "revision_number",
    "revision_provenance",
    "source_coordinate_system",
    "localization_warning",
)


@dataclass(frozen=True, slots=True)
class ExportContext:
    """Immutable identifiers and thresholds shared by export rows."""

    project_id: str
    image_asset_id: str
    run_id: str
    profile_id: str
    thresholds: Mapping[str, Real] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReviewedProposalRow:
    """One reviewed proposal in canonical source-image coordinates."""

    project_id: str
    image_asset_id: str
    run_id: str
    profile_id: str
    class_name: str
    threshold: float | None
    proposal_id: str
    source_rect: Rect
    area: int
    peak_confidence: float
    mean_confidence: float
    review_status: str
    revision_number: int
    revision_provenance: dict[str, Any]
    proposal_source_rect: Rect
    proposal_provenance: dict[str, Any]
    source_coordinate_system: str = SOURCE_COORDINATE_SYSTEM
    localization_warning: str = APPROXIMATE_LOCALIZATION_WARNING

    @property
    def geometry(self) -> Rect:
        """The latest reviewed source-image rectangle."""

        return self.source_rect

    @property
    def rect(self) -> Rect:
        """Short geometry alias used by existing proposal consumers."""

        return self.source_rect

    @property
    def source_image_rect(self) -> Rect:
        """Explicit source-image-coordinate geometry alias."""

        return self.source_rect

    @property
    def review_revision(self) -> int:
        """Short alias for the append-only review revision number."""

        return self.revision_number

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation for later exporters."""

        return {
            "project_id": self.project_id,
            "image_asset_id": self.image_asset_id,
            "run_id": self.run_id,
            "profile_id": self.profile_id,
            "class_name": self.class_name,
            "threshold": self.threshold,
            "proposal_id": self.proposal_id,
            "source_rect": _rect_dict(self.source_rect),
            "proposal_source_rect": _rect_dict(self.proposal_source_rect),
            "area": self.area,
            "peak_confidence": self.peak_confidence,
            "mean_confidence": self.mean_confidence,
            "review_status": self.review_status,
            "revision_number": self.revision_number,
            "revision_provenance": _json_safe(self.revision_provenance),
            "proposal_provenance": _json_safe(self.proposal_provenance),
            "source_coordinate_system": self.source_coordinate_system,
            "localization_warning": self.localization_warning,
        }


def build_reviewed_rows(
    proposals: Iterable[DefectProposal] | Iterable[ReviewQueueItem],
    revisions: Mapping[str, Any] | Iterable[Any] | None = None,
    *,
    project_id: str | os.PathLike[str] | None = None,
    image_asset_id: str | os.PathLike[str] | None = None,
    run_id: str | os.PathLike[str] | None = None,
    profile_id: str | os.PathLike[str] | None = None,
    thresholds: Mapping[str, Real] | None = None,
    context: ExportContext | Mapping[str, Any] | None = None,
) -> tuple[ReviewedProposalRow, ...]:
    """Project proposals and latest review revisions into canonical rows.

    ``build_review_queue`` selects the greatest revision number per proposal.
    Consequently a corrected review contributes its corrected source rectangle
    while an accepted/rejected review retains the immutable proposal geometry.
    Passing already-built :class:`ReviewQueueItem` values is supported for UI
    callers and avoids recomputing queue flags.
    """

    resolved = _context(
        context,
        project_id=project_id,
        image_asset_id=image_asset_id,
        run_id=run_id,
        profile_id=profile_id,
        thresholds=thresholds,
    )
    values = tuple(proposals)
    if values and all(isinstance(value, ReviewQueueItem) for value in values):
        if revisions is not None:
            raise TypeError("revisions cannot accompany ReviewQueueItem values")
        queue = tuple(sorted(values, key=lambda value: str(value.proposal_id)))
    else:
        if any(not isinstance(value, DefectProposal) for value in values):
            raise TypeError("proposals must contain DefectProposal or ReviewQueueItem values")
        queue = build_review_queue(values, revisions)

    rows: list[ReviewedProposalRow] = []
    for item in queue:
        proposal = item.proposal
        class_name = _text(proposal.class_name, "class_name")
        threshold = _threshold(class_name, resolved.thresholds, proposal.provenance)
        revision = item.latest_revision
        if isinstance(revision, Mapping):
            raw_revision_provenance = revision.get("provenance", {})
        else:
            raw_revision_provenance = getattr(revision, "provenance", {})
        revision_provenance = _mapping(raw_revision_provenance, "revision provenance")
        rows.append(
            ReviewedProposalRow(
                project_id=resolved.project_id,
                image_asset_id=resolved.image_asset_id,
                run_id=resolved.run_id,
                profile_id=resolved.profile_id,
                class_name=class_name,
                threshold=threshold,
                proposal_id=_text(proposal.proposal_id, "proposal_id"),
                source_rect=_coerce_rect(item.source_rect),
                area=_positive_int(proposal.area, "area"),
                peak_confidence=_confidence(proposal.peak_confidence, "peak_confidence"),
                mean_confidence=_confidence(proposal.mean_confidence, "mean_confidence"),
                review_status=_text(item.status, "review_status"),
                revision_number=item.revision_number,
                revision_provenance=_json_safe(revision_provenance),
                proposal_source_rect=_coerce_rect(proposal.source_rect),
                proposal_provenance=_json_safe(_mapping(proposal.provenance, "proposal provenance")),
            )
        )
    return tuple(sorted(rows, key=lambda row: row.proposal_id))


# The shorter name is convenient for CSV/JSON callers and kept as a public
# alias so later slices do not need to duplicate row construction.
build_export_rows = build_reviewed_rows
canonical_reviewed_rows = build_reviewed_rows
build_rows = build_reviewed_rows


def export_proposals_csv(
    destination: str | os.PathLike[str],
    rows: Iterable[ReviewedProposalRow],
    *,
    overwrite: bool = True,
) -> Path:
    """Write canonical rows using the stable CSV schema and UTF-8 encoding."""

    path = Path(destination)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    values = tuple(rows)
    if any(not isinstance(row, ReviewedProposalRow) for row in values):
        raise TypeError("rows must contain ReviewedProposalRow values")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in values:
            writer.writerow(_csv_row(row))
    return path


write_proposals_csv = export_proposals_csv
write_csv = export_proposals_csv


def _context(
    context: ExportContext | Mapping[str, Any] | None,
    *,
    project_id: str | os.PathLike[str] | None,
    image_asset_id: str | os.PathLike[str] | None,
    run_id: str | os.PathLike[str] | None,
    profile_id: str | os.PathLike[str] | None,
    thresholds: Mapping[str, Real] | None,
) -> ExportContext:
    if context is not None:
        if any(value is not None for value in (project_id, image_asset_id, run_id, profile_id, thresholds)):
            raise TypeError("context cannot be combined with explicit export identifiers")
        if isinstance(context, Mapping):
            context = ExportContext(**dict(context))
        if not isinstance(context, ExportContext):
            raise TypeError("context must be ExportContext or a mapping")
        project_id, image_asset_id, run_id, profile_id, thresholds = (
            context.project_id,
            context.image_asset_id,
            context.run_id,
            context.profile_id,
            context.thresholds,
        )
    if any(value is None for value in (project_id, image_asset_id, run_id, profile_id)):
        raise ValueError("project_id, image_asset_id, run_id, and profile_id are required")
    return ExportContext(
        project_id=_text(project_id, "project_id"),
        image_asset_id=_text(image_asset_id, "image_asset_id"),
        run_id=_text(run_id, "run_id"),
        profile_id=_text(profile_id, "profile_id"),
        thresholds=dict(thresholds or {}),
    )


def _threshold(
    class_name: str,
    thresholds: Mapping[str, Real],
    provenance: Mapping[str, Any],
) -> float | None:
    value = thresholds[class_name] if class_name in thresholds else provenance.get("threshold")
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"threshold for class {class_name!r} must be finite and in the range 0 to 1")
    return float(value)


def _csv_row(row: ReviewedProposalRow) -> dict[str, Any]:
    rect = row.source_rect
    return {
        "project_id": row.project_id,
        "image_asset_id": row.image_asset_id,
        "run_id": row.run_id,
        "profile_id": row.profile_id,
        "class_name": row.class_name,
        "threshold": "" if row.threshold is None else _number(row.threshold),
        "proposal_id": row.proposal_id,
        "source_x": rect.x,
        "source_y": rect.y,
        "source_width": rect.width,
        "source_height": rect.height,
        "area": row.area,
        "peak_confidence": _number(row.peak_confidence),
        "mean_confidence": _number(row.mean_confidence),
        "review_status": row.review_status,
        "revision_number": row.revision_number,
        "revision_provenance": _json_text(row.revision_provenance),
        "source_coordinate_system": row.source_coordinate_system,
        "localization_warning": row.localization_warning,
    }


def _rect_dict(rect: Rect) -> dict[str, int]:
    return {"x": rect.x, "y": rect.y, "width": rect.width, "height": rect.height}


def _coerce_rect(value: Any) -> Rect:
    if isinstance(value, Rect):
        return value
    if isinstance(value, Mapping):
        values = (value.get("x"), value.get("y"), value.get("width"), value.get("height"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
        values = tuple(value)
    else:
        raise TypeError("source geometry must be a Rect, mapping, or four-value sequence")
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in values):
        raise TypeError("source geometry coordinates must be numeric")
    normalized = tuple(int(item) for item in values)
    if normalized[2] <= 0 or normalized[3] <= 0:
        raise ValueError("source geometry dimensions must be positive")
    return Rect(*normalized)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return {str(key): _json_safe(item) for key, item in value.items()}


def _json_safe(value: Any) -> Any:
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("export provenance values must be finite")
    return value


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _text(value: Any, name: str) -> str:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _confidence(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite and numeric")
    return float(value)


def _number(value: float) -> str:
    return format(float(value), ".15g")


__all__ = [
    "APPROXIMATE_LOCALIZATION_WARNING",
    "CSV_COLUMNS",
    "ExportContext",
    "ReviewedProposalRow",
    "SOURCE_COORDINATE_SYSTEM",
    "build_export_rows",
    "build_rows",
    "build_reviewed_rows",
    "canonical_reviewed_rows",
    "export_proposals_csv",
    "write_csv",
    "write_proposals_csv",
]
