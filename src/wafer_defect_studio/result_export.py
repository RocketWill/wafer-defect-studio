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
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPen

from .detection_windows import Rect
from .proposal_generation import DefectProposal
from .proposal_queue import ReviewQueueItem, build_review_queue


SOURCE_COORDINATE_SYSTEM = "source-image-pixels"
APPROXIMATE_LOCALIZATION_WARNING = (
    "Approximate weak localization: proposal geometry is not a pixel-accurate segmentation mask."
)
JSON_SCHEMA_VERSION = 1
JSON_EXPORT_TYPE = "reviewed_detection_results"
_HEADLESS_QT_APP: QGuiApplication | None = None

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


@dataclass(frozen=True, slots=True)
class ExportBundleResult:
    """Outcome of one transactional CSV/JSON/PNG export."""

    success: bool
    paths: tuple[Path, ...]
    error: str | None = None


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


def export_proposals_json(
    destination: str | os.PathLike[str],
    rows: Iterable[ReviewedProposalRow],
    *,
    overwrite: bool = True,
) -> Path:
    """Write a versioned, deterministic UTF-8 JSON export.

    All rows in one document must share the immutable project/image/run/profile
    context.  The check prevents a document from claiming one run while
    carrying rows from another run.
    """

    path = Path(destination)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    values = tuple(rows)
    if any(not isinstance(row, ReviewedProposalRow) for row in values):
        raise TypeError("rows must contain ReviewedProposalRow values")
    metadata = _json_metadata(values)
    payload = {
        "schema_version": JSON_SCHEMA_VERSION,
        "export_type": JSON_EXPORT_TYPE,
        "metadata": metadata,
        "proposals": [row.to_dict() for row in sorted(values, key=lambda item: item.proposal_id)],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        separators=(",", ": "),
    )
    path.write_text(serialized + "\n", encoding="utf-8")
    return path


write_proposals_json = export_proposals_json
write_json = export_proposals_json


def export_proposals_png(
    destination: str | os.PathLike[str],
    source_image: QImage,
    rows: Iterable[ReviewedProposalRow],
    *,
    selected_class: str,
    confidence_map: np.ndarray | Sequence[Sequence[Real]] | None = None,
    grid_rects: Iterable[Rect | Sequence[int] | Mapping[str, Any]] = (),
    overwrite: bool = True,
) -> Path:
    """Render a native-size reviewed detection image to PNG.

    The source image is copied before any drawing.  Confidence values are
    interpreted in source-image pixel coordinates; NaN values remain
    transparent and finite values are colourized from blue (low) to red
    (high).  Proposal and optional grid rectangles are drawn over that copy.
    """

    if not isinstance(source_image, QImage) or source_image.isNull():
        raise ValueError("source_image must be a non-empty QImage")
    selected_name = _text(selected_class, "selected_class")
    path = Path(destination)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    values = tuple(rows)
    if any(not isinstance(row, ReviewedProposalRow) for row in values):
        raise TypeError("rows must contain ReviewedProposalRow values")

    # QPainter text rendering requires a live Qt GUI application.  Normal UI
    # callers already have one; headless export callers get a minimal one.
    global _HEADLESS_QT_APP
    if QGuiApplication.instance() is None:
        _HEADLESS_QT_APP = QGuiApplication([])

    width, height = source_image.width(), source_image.height()
    rendered = source_image.convertToFormat(QImage.Format_ARGB32)
    if confidence_map is not None:
        _paint_confidence_map(rendered, confidence_map)

    painter = QPainter(rendered)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        _paint_proposal_rectangles(painter, values, selected_name)
        _paint_grid_rectangles(painter, grid_rects)
        legend = _legend_text(selected_name)
        legend_height = min(height, max(18, min(48, height)))
        painter.fillRect(0, 0, width, legend_height, QColor(0, 0, 0, 190))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(
            2,
            2,
            max(1, width - 4),
            max(1, legend_height - 2),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap),
            legend,
        )
    finally:
        painter.end()

    legend = _legend_text(selected_name)
    rendered.setText("legend", legend)
    rendered.setText("selected_class", selected_name)
    rendered.setText("confidence_meaning", "Confidence is a per-pixel score from 0 (low) to 1 (high).")
    rendered.setText("source_coordinate_system", SOURCE_COORDINATE_SYSTEM)
    rendered.setText("localization_warning", APPROXIMATE_LOCALIZATION_WARNING)
    if not rendered.save(str(path), "PNG"):
        raise OSError(f"failed to write PNG: {path}")
    return path


write_proposals_png = export_proposals_png
write_png = export_proposals_png


def export_result_bundle(
    csv_destination: str | os.PathLike[str],
    json_destination: str | os.PathLike[str],
    png_destination: str | os.PathLike[str],
    source_image: QImage,
    rows: Iterable[ReviewedProposalRow],
    *,
    selected_class: str,
    confidence_map: np.ndarray | Sequence[Sequence[Real]] | None = None,
    grid_rects: Iterable[Rect | Sequence[int] | Mapping[str, Any]] = (),
    overwrite: bool = False,
) -> ExportBundleResult:
    """Stage and atomically publish CSV, JSON, and PNG exports.

    The three destinations are checked before any output is touched.  Writers
    receive temporary paths in the destination directories; publication uses
    ``os.replace`` and existing files are moved to same-directory backups so a
    writer or publication failure can restore the previous set in full.
    """

    destinations = tuple(Path(value) for value in (csv_destination, json_destination, png_destination))
    duplicate = _duplicate_destination(destinations)
    if duplicate is not None:
        return ExportBundleResult(False, (), f"duplicate export destination: {duplicate}")
    existing = tuple(path for path in destinations if path.exists())
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        return ExportBundleResult(False, (), f"refusing to overwrite existing export: {joined}")

    values = tuple(rows)
    temporary_paths: list[Path] = []
    publication_records: list[dict[str, Any]] = []
    try:
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)

        csv_stage = _temporary_path(destinations[0], "csv")
        json_stage = _temporary_path(destinations[1], "json")
        png_stage = _temporary_path(destinations[2], "png")
        temporary_paths.extend((csv_stage, json_stage, png_stage))
        export_proposals_csv(csv_stage, values, overwrite=True)
        export_proposals_json(json_stage, values, overwrite=True)
        export_proposals_png(
            png_stage,
            source_image,
            values,
            selected_class=selected_class,
            confidence_map=confidence_map,
            grid_rects=grid_rects,
            overwrite=True,
        )

        for stage, destination in zip(temporary_paths, destinations):
            record: dict[str, Any] = {
                "destination": destination,
                "original_exists": destination.exists(),
                "backup": None,
            }
            publication_records.append(record)
            if record["original_exists"]:
                backup = _temporary_path(destination, "backup")
                backup.unlink(missing_ok=True)
                record["backup"] = backup
                os.replace(destination, backup)
            os.replace(stage, destination)

        for record in publication_records:
            backup = record["backup"]
            if backup is not None:
                Path(backup).unlink(missing_ok=True)
        temporary_paths.clear()
        return ExportBundleResult(True, destinations, None)
    except Exception as error:
        _rollback_publication(publication_records)
        return ExportBundleResult(False, (), f"export bundle failed: {error}")
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        for record in publication_records:
            backup = record["backup"]
            if backup is not None:
                Path(backup).unlink(missing_ok=True)


def _temporary_path(destination: Path, kind: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=f".{kind}.tmp",
        dir=str(destination.parent),
    )
    os.close(descriptor)
    return Path(name)


def _duplicate_destination(destinations: Sequence[Path]) -> str | None:
    seen: dict[str, Path] = {}
    for destination in destinations:
        key = os.path.normcase(os.path.abspath(os.fspath(destination)))
        if key in seen:
            return str(destination)
        seen[key] = destination
    return None


def _rollback_publication(records: Sequence[Mapping[str, Any]]) -> None:
    for record in reversed(records):
        destination = Path(record["destination"])
        backup = record["backup"]
        original_exists = bool(record["original_exists"])
        try:
            if backup is not None:
                if destination.exists():
                    destination.unlink()
                backup_path = Path(backup)
                if backup_path.exists():
                    os.replace(backup_path, destination)
            elif not original_exists and destination.exists():
                destination.unlink()
        except OSError:
            # The primary failure is reported; best-effort rollback must not
            # hide it or turn a visible failed result into a success.
            continue


def _paint_confidence_map(rendered: QImage, confidence_map: Any) -> None:
    try:
        values = np.asarray(confidence_map, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("confidence_map must be a numeric two-dimensional array") from error
    if values.shape != (rendered.height(), rendered.width()):
        raise ValueError("confidence_map dimensions must match the source image")
    overlay = QImage(rendered.size(), QImage.Format_ARGB32)
    overlay.fill(0)
    for y in range(rendered.height()):
        for x in range(rendered.width()):
            value = values[y, x]
            if np.isnan(value):
                continue
            if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("confidence_map values must be NaN or in the range 0 to 1")
            strength = float(value)
            overlay.setPixelColor(
                x,
                y,
                QColor(
                    int(round(255.0 * strength)),
                    35,
                    int(round(255.0 * (1.0 - strength))),
                    125,
                ),
            )
    painter = QPainter(rendered)
    try:
        painter.drawImage(0, 0, overlay)
    finally:
        painter.end()


def _paint_proposal_rectangles(
    painter: QPainter,
    rows: Sequence[ReviewedProposalRow],
    selected_class: str,
) -> None:
    for row in sorted(rows, key=lambda value: value.proposal_id):
        color = "#ffd400" if row.class_name == selected_class else "#00d9ff"
        painter.setPen(QPen(QColor(color), 1))
        rect = row.source_rect
        painter.drawRect(rect.x, rect.y, max(0, rect.width - 1), max(0, rect.height - 1))


def _paint_grid_rectangles(
    painter: QPainter,
    grid_rects: Iterable[Rect | Sequence[int] | Mapping[str, Any]],
) -> None:
    painter.setPen(QPen(QColor("#66ff66"), 1, Qt.PenStyle.DashLine))
    for value in grid_rects:
        rect = _coerce_rect(value)
        painter.drawRect(rect.x, rect.y, max(0, rect.width - 1), max(0, rect.height - 1))


def _legend_text(selected_class: str) -> str:
    return (
        f"Class: {selected_class} | confidence: 0 (low) to 1 (high) | "
        "Approximate localization (not a segmentation mask) | coordinates: source-image-pixels"
    )


def _json_metadata(rows: Sequence[ReviewedProposalRow]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one ReviewedProposalRow is required")
    first = rows[0]
    identity = (first.project_id, first.image_asset_id, first.run_id, first.profile_id)
    for row in rows[1:]:
        if (row.project_id, row.image_asset_id, row.run_id, row.profile_id) != identity:
            raise ValueError("all rows must share project/image/run/profile identifiers")
        if row.source_coordinate_system != first.source_coordinate_system:
            raise ValueError("all rows must share a source coordinate system")
        if row.localization_warning != first.localization_warning:
            raise ValueError("all rows must share a localization warning")
    project_id, image_asset_id, run_id, profile_id = identity
    return {
        "schema_version": JSON_SCHEMA_VERSION,
        "export_type": JSON_EXPORT_TYPE,
        "project_id": project_id,
        "image_asset_id": image_asset_id,
        "run_id": run_id,
        "profile_id": profile_id,
        "source_coordinate_system": first.source_coordinate_system,
        "localization_warning": first.localization_warning,
        "approximate_localization": True,
    }


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
    "ExportBundleResult",
    "JSON_EXPORT_TYPE",
    "JSON_SCHEMA_VERSION",
    "ReviewedProposalRow",
    "SOURCE_COORDINATE_SYSTEM",
    "build_export_rows",
    "build_rows",
    "build_reviewed_rows",
    "canonical_reviewed_rows",
    "export_proposals_csv",
    "export_proposals_png",
    "export_proposals_json",
    "export_result_bundle",
    "write_csv",
    "write_json",
    "write_png",
    "write_proposals_csv",
    "write_proposals_json",
    "write_proposals_png",
]
