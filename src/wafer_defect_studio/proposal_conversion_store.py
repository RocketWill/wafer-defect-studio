"""GUI-side persistence for explicit proposal-to-Grid Annotation conversions."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .grid_geometry import AnnotationGrid
from .project import (
    _GRID_ANNOTATIONS_TABLE_SQL,
    _PROPOSAL_CONVERSION_SCHEMA_VERSION,
    _PROPOSAL_CONVERSIONS_TABLE_SQL,
    _PROPOSAL_CONVERSION_COLUMNS,
    _PROPOSAL_REVIEW_SCHEMA_VERSION,
    ProjectError,
    open_project,
)
from .proposal_conversion import ConversionCell, ConversionPreview
from .proposal_generation import _json_safe


class ConversionStoreError(ProjectError):
    """Raised when an explicit proposal conversion cannot be committed."""


@dataclass(frozen=True, slots=True)
class ConversionRecord:
    """Immutable audit record for one confirmed proposal conversion."""

    conversion_id: str
    image_asset_id: str
    affected_cells: tuple[ConversionCell, ...]
    class_codes: tuple[str, ...]
    proposal_ids: tuple[str, ...]
    provenance: Mapping[str, Any]
    actor: str | None
    converted_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.affected_cells, tuple) or not all(
            isinstance(cell, ConversionCell) for cell in self.affected_cells
        ):
            raise ValueError("affected_cells must be a tuple of ConversionCell values")
        if not isinstance(self.class_codes, tuple):
            raise ValueError("class_codes must be a tuple")
        if not isinstance(self.proposal_ids, tuple):
            raise ValueError("proposal_ids must be a tuple")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def cells(self) -> tuple[ConversionCell, ...]:
        """Short alias for affected Annotation Grids."""

        return self.affected_cells

    @property
    def affected_classes(self) -> tuple[str, ...]:
        """Defect Class codes contributed by the preview."""

        return self.class_codes

    @property
    def classes(self) -> tuple[str, ...]:
        """Short alias for :attr:`class_codes`."""

        return self.class_codes

    @property
    def source_proposal_ids(self) -> tuple[str, ...]:
        """Proposal identities that were explicitly converted."""

        return self.proposal_ids

    @property
    def preview_provenance(self) -> Mapping[str, Any]:
        """The immutable provenance captured from the conversion preview."""

        return self.provenance

    @property
    def source_coordinate_system(self) -> Any:
        """Coordinate-system marker retained with the preview provenance."""

        return self.provenance.get("source_coordinate_system")

    @property
    def timestamp(self) -> str:
        """Timestamp alias for consumers that call the field ``time``."""

        return self.converted_at

    @property
    def time(self) -> str:
        return self.converted_at

    @property
    def preview(self) -> ConversionPreview:
        """Reconstruct the exact value-only preview represented by this record."""

        return ConversionPreview(self.affected_cells, self.proposal_ids, self.provenance)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-friendly audit representation."""

        return {
            "conversion_id": self.conversion_id,
            "image_asset_id": self.image_asset_id,
            "affected_cells": tuple(
                {
                    "row": cell.row,
                    "column": cell.column,
                    "x": cell.grid.x,
                    "y": cell.grid.y,
                    "width": cell.grid.width,
                    "height": cell.grid.height,
                    "class_codes": cell.class_codes,
                    "proposal_ids": cell.proposal_ids,
                }
                for cell in self.affected_cells
            ),
            "class_codes": self.class_codes,
            "proposal_ids": self.proposal_ids,
            "provenance": dict(self.provenance),
            "actor": self.actor,
            "converted_at": self.converted_at,
        }


def confirm_proposal_conversion(
    project_path: str | Path,
    preview: ConversionPreview,
    image_asset_id: str,
    confirmed: bool = False,
    *,
    actor: str | None = None,
    conversion_id: str | None = None,
    converted_at: str | None = None,
    timestamp: str | None = None,
    time: str | None = None,
) -> ConversionRecord:
    """Commit an explicitly confirmed preview in one GUI-side transaction.

    The confirmation check happens before opening a writable database.  A
    cancelled or unchecked action therefore has no schema migration, table
    creation, annotation update, or provenance write as a side effect.
    """

    if confirmed is not True:
        raise ConversionStoreError("Proposal conversion requires explicit confirmation")
    preview = _validate_preview(preview)
    image_asset_id = _identifier(image_asset_id, "image_asset_id")
    conversion_id = _identifier(conversion_id or str(uuid.uuid4()), "conversion_id")
    actor = _optional_identifier(actor, "actor")
    chosen_time = _coalesce_time(converted_at, timestamp, time)
    if chosen_time is None:
        chosen_time = datetime.now(timezone.utc).isoformat()
    chosen_time = _text(chosen_time, "converted_at")

    info = open_project(project_path)
    if info.schema_version < _PROPOSAL_REVIEW_SCHEMA_VERSION:
        raise ConversionStoreError("Proposal conversion requires project schema 17 or newer")
    if info.schema_version > _PROPOSAL_CONVERSION_SCHEMA_VERSION:
        raise ConversionStoreError(f"Unsupported project schema: {info.path / 'project.sqlite'}")

    database = info.path / "project.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        _ensure_schema(connection, info.project_id, database)
        _validate_image(connection, image_asset_id)
        enabled_classes = _load_class_state(connection)
        _validate_preview_metadata(preview, enabled_classes)

        for cell in preview.cells:
            existing = connection.execute(
                "SELECT class_codes_json FROM grid_annotations "
                "WHERE image_asset_id = ? AND row = ? AND column = ?",
                (image_asset_id, cell.row, cell.column),
            ).fetchone()
            existing_codes = _decode_codes(existing[0]) if existing is not None else ()
            for code in existing_codes:
                if code not in enabled_classes:
                    raise ConversionStoreError(f"Unknown existing Defect Class code: {code}")
            merged_codes = existing_codes + tuple(
                code for code in cell.class_codes if code not in existing_codes
            )
            connection.execute(
                "INSERT INTO grid_annotations "
                "(image_asset_id, row, column, class_codes_json) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(image_asset_id, row, column) DO UPDATE SET "
                "class_codes_json = excluded.class_codes_json",
                (image_asset_id, cell.row, cell.column, _encode_codes(merged_codes)),
            )

        class_codes = tuple(sorted({code for cell in preview.cells for code in cell.class_codes}))
        connection.execute(
            "INSERT INTO proposal_conversions "
            "(conversion_id, image_asset_id, affected_cells_json, class_codes_json, "
            "proposal_ids_json, provenance_json, actor, converted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                conversion_id,
                image_asset_id,
                _encode_cells(preview.cells),
                _encode_codes(class_codes),
                _encode_strings(preview.source_proposal_ids),
                _encode_provenance(preview.provenance),
                actor,
                chosen_time,
            ),
        )
        connection.commit()
    except ConversionStoreError:
        connection.rollback()
        raise
    except sqlite3.IntegrityError as error:
        connection.rollback()
        raise ConversionStoreError("Conversion identity is already used or invalid") from error
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        connection.rollback()
        raise ConversionStoreError(f"Invalid proposal conversion metadata: {database}") from error
    except sqlite3.Error as error:
        connection.rollback()
        raise ConversionStoreError(f"Invalid proposal conversion metadata: {database}") from error
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    return ConversionRecord(
        conversion_id,
        image_asset_id,
        preview.cells,
        class_codes,
        preview.source_proposal_ids,
        preview.provenance,
        actor,
        chosen_time,
    )


def apply_proposal_conversion(
    project_path: str | Path,
    preview: ConversionPreview,
    image_asset_id: str,
    confirmed: bool = False,
    *,
    actor: str | None = None,
    conversion_id: str | None = None,
    converted_at: str | None = None,
    timestamp: str | None = None,
    time: str | None = None,
) -> ConversionRecord:
    """Explicitly named application alias for :func:`confirm_proposal_conversion`."""

    return confirm_proposal_conversion(
        project_path,
        preview,
        image_asset_id,
        confirmed,
        actor=actor,
        conversion_id=conversion_id,
        converted_at=converted_at,
        timestamp=timestamp,
        time=time,
    )


def load_proposal_conversion(
    project_path: str | Path, conversion_id: str
) -> ConversionRecord:
    """Load one immutable proposal conversion audit record."""

    conversion_id = _identifier(conversion_id, "conversion_id")
    info = open_project(project_path)
    if info.schema_version < _PROPOSAL_CONVERSION_SCHEMA_VERSION:
        raise ConversionStoreError(f"Unknown Proposal Conversion: {conversion_id}")
    database = info.path / "project.sqlite"
    try:
        connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise ConversionStoreError(f"Cannot open project database: {database}") from error
    try:
        row = connection.execute(
            "SELECT conversion_id, image_asset_id, affected_cells_json, class_codes_json, "
            "proposal_ids_json, provenance_json, actor, converted_at "
            "FROM proposal_conversions WHERE conversion_id = ?",
            (conversion_id,),
        ).fetchone()
    except sqlite3.Error as error:
        raise ConversionStoreError(f"Invalid Proposal Conversion metadata: {database}") from error
    finally:
        connection.close()
    if row is None:
        raise ConversionStoreError(f"Unknown Proposal Conversion: {conversion_id}")
    try:
        return _decode_record(row)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ConversionStoreError(f"Invalid Proposal Conversion metadata: {database}") from error


def load_proposal_conversions(
    project_path: str | Path, image_asset_id: str | None = None
) -> tuple[ConversionRecord, ...]:
    """Load immutable conversion records in deterministic identity order."""

    info = open_project(project_path)
    if info.schema_version < _PROPOSAL_CONVERSION_SCHEMA_VERSION:
        return ()
    if image_asset_id is not None:
        image_asset_id = _identifier(image_asset_id, "image_asset_id")
    database = info.path / "project.sqlite"
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        if image_asset_id is None:
            rows = connection.execute(
                "SELECT conversion_id, image_asset_id, affected_cells_json, class_codes_json, "
                "proposal_ids_json, provenance_json, actor, converted_at "
                "FROM proposal_conversions ORDER BY conversion_id"
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT conversion_id, image_asset_id, affected_cells_json, class_codes_json, "
                "proposal_ids_json, provenance_json, actor, converted_at "
                "FROM proposal_conversions WHERE image_asset_id = ? ORDER BY conversion_id",
                (image_asset_id,),
            ).fetchall()
    except sqlite3.Error as error:
        raise ConversionStoreError(f"Invalid Proposal Conversion metadata: {database}") from error
    finally:
        connection.close()
    try:
        return tuple(_decode_record(row) for row in rows)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ConversionStoreError(f"Invalid Proposal Conversion metadata: {database}") from error


def _ensure_schema(connection: sqlite3.Connection, project_id: str, database: Path) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == _PROPOSAL_REVIEW_SCHEMA_VERSION:
        connection.execute(_PROPOSAL_CONVERSIONS_TABLE_SQL)
        _create_immutable_triggers(connection)
        updated = connection.execute(
            "UPDATE project_metadata SET schema_version = ? "
            "WHERE project_id = ? AND schema_version = ?",
            (_PROPOSAL_CONVERSION_SCHEMA_VERSION, project_id, _PROPOSAL_REVIEW_SCHEMA_VERSION),
        ).rowcount
        if updated != 1:
            raise ConversionStoreError(f"Invalid project metadata: {database}")
        connection.execute(f"PRAGMA user_version = {_PROPOSAL_CONVERSION_SCHEMA_VERSION}")
    elif version == _PROPOSAL_CONVERSION_SCHEMA_VERSION:
        connection.execute(_PROPOSAL_CONVERSIONS_TABLE_SQL)
        _create_immutable_triggers(connection)
    elif version < _PROPOSAL_REVIEW_SCHEMA_VERSION or version > _PROPOSAL_CONVERSION_SCHEMA_VERSION:
        raise ConversionStoreError(f"Unsupported project schema: {database}")


def _create_immutable_triggers(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TRIGGER IF NOT EXISTS proposal_conversions_no_update "
        "BEFORE UPDATE ON proposal_conversions "
        "BEGIN SELECT RAISE(ABORT, 'Proposal Conversions are immutable'); END"
    )
    connection.execute(
        "CREATE TRIGGER IF NOT EXISTS proposal_conversions_no_delete "
        "BEFORE DELETE ON proposal_conversions "
        "BEGIN SELECT RAISE(ABORT, 'Proposal Conversions are immutable'); END"
    )


def _validate_image(connection: sqlite3.Connection, image_asset_id: str) -> None:
    if (
        connection.execute(
            "SELECT 1 FROM image_assets WHERE image_asset_id = ?", (image_asset_id,)
        ).fetchone()
        is None
    ):
        raise ConversionStoreError(f"Unknown image asset: {image_asset_id}")


def _load_class_state(connection: sqlite3.Connection) -> dict[str, bool]:
    try:
        rows = connection.execute("SELECT code, enabled FROM defect_classes").fetchall()
    except sqlite3.Error as error:
        raise ConversionStoreError("Invalid Defect Class metadata") from error
    state: dict[str, bool] = {}
    for code, enabled in rows:
        code = _identifier(code, "Defect Class code")
        if code in state:
            raise ConversionStoreError(f"Duplicate Defect Class code: {code}")
        state[code] = bool(enabled)
    return state


def _validate_preview(preview: object) -> ConversionPreview:
    if not isinstance(preview, ConversionPreview):
        raise ConversionStoreError("preview must be a ConversionPreview")
    if not isinstance(preview.provenance, Mapping):
        raise ConversionStoreError("preview provenance must be a mapping")
    return preview


def _validate_preview_metadata(preview: ConversionPreview, class_state: Mapping[str, bool]) -> None:
    marker = preview.provenance.get("source_coordinate_system")
    if marker != "source-image-pixels":
        raise ConversionStoreError(
            "Conversion preview provenance must identify source-image-pixels coordinates"
        )
    source_ids = _validate_strings(preview.source_proposal_ids, "source_proposal_ids")
    if len(source_ids) != len(set(source_ids)):
        raise ConversionStoreError("source_proposal_ids must not contain duplicates")
    seen_cells: set[tuple[int, int]] = set()
    for cell in preview.cells:
        if not isinstance(cell, ConversionCell):
            raise ConversionStoreError("preview cells must contain ConversionCell values")
        identity = (cell.row, cell.column)
        if identity in seen_cells:
            raise ConversionStoreError("preview cells must not contain duplicate row/column values")
        seen_cells.add(identity)
        classes = _validate_strings(cell.class_codes, "cell.class_codes")
        if len(classes) != len(set(classes)):
            raise ConversionStoreError("cell.class_codes must not contain duplicates")
        for code in classes:
            if code not in class_state:
                raise ConversionStoreError(f"Unknown Defect Class code: {code}")
            if not class_state[code]:
                raise ConversionStoreError(f"Archived Defect Class code: {code}")
        proposal_ids = _validate_strings(cell.proposal_ids, "cell.proposal_ids")
        if len(proposal_ids) != len(set(proposal_ids)):
            raise ConversionStoreError("cell.proposal_ids must not contain duplicates")
        if not set(proposal_ids).issubset(source_ids):
            raise ConversionStoreError("cell proposal IDs must be present in source_proposal_ids")
    try:
        json.dumps(_json_safe(dict(preview.provenance)), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ConversionStoreError("preview provenance must contain JSON-compatible values") from error


def _decode_record(row: Sequence[Any]) -> ConversionRecord:
    cells_value = json.loads(row[2])
    if not isinstance(cells_value, list):
        raise ValueError("affected_cells_json must be an array")
    cells = tuple(_decode_cell(value) for value in cells_value)
    class_codes = tuple(_decode_strings(row[3]))
    proposal_ids = tuple(_decode_strings(row[4]))
    provenance = json.loads(row[5])
    if not isinstance(provenance, dict):
        raise ValueError("provenance_json must be an object")
    return ConversionRecord(
        _identifier(row[0], "conversion_id"),
        _identifier(row[1], "image_asset_id"),
        cells,
        class_codes,
        proposal_ids,
        provenance,
        _optional_identifier(row[6], "actor"),
        _text(row[7], "converted_at"),
    )


def _decode_cell(value: object) -> ConversionCell:
    if not isinstance(value, Mapping):
        raise ValueError("affected cell must be an object")
    grid = AnnotationGrid(
        _integer(value.get("row"), "cell.row"),
        _integer(value.get("column"), "cell.column"),
        _integer(value.get("x"), "cell.x"),
        _integer(value.get("y"), "cell.y"),
        _positive_integer(value.get("width"), "cell.width"),
        _positive_integer(value.get("height"), "cell.height"),
    )
    class_codes = tuple(_decode_strings(value.get("class_codes")))
    proposal_ids = tuple(_decode_strings(value.get("proposal_ids")))
    return ConversionCell(grid, class_codes, proposal_ids)


def _encode_cells(cells: tuple[ConversionCell, ...]) -> str:
    return json.dumps(
        [
            {
                "row": cell.row,
                "column": cell.column,
                "x": cell.grid.x,
                "y": cell.grid.y,
                "width": cell.grid.width,
                "height": cell.grid.height,
                "class_codes": cell.class_codes,
                "proposal_ids": cell.proposal_ids,
            }
            for cell in cells
        ],
        sort_keys=True,
        separators=(",", ":"),
    )


def _encode_codes(codes: tuple[str, ...]) -> str:
    return json.dumps(codes, separators=(",", ":"))


def _encode_strings(values: tuple[str, ...]) -> str:
    return json.dumps(values, separators=(",", ":"))


def _encode_provenance(provenance: Mapping[str, Any]) -> str:
    return json.dumps(_json_safe(dict(provenance)), sort_keys=True, separators=(",", ":"))


def _decode_codes(serialized: object) -> tuple[str, ...]:
    values = json.loads(serialized)
    if not isinstance(values, list):
        raise ValueError("class_codes_json must be an array")
    return tuple(_validate_strings(tuple(values), "class_codes_json"))


def _decode_strings(serialized: object) -> tuple[str, ...]:
    if not isinstance(serialized, (str, list, tuple)):
        raise ValueError("string array must be an array")
    values = json.loads(serialized) if isinstance(serialized, str) else serialized
    if not isinstance(values, (list, tuple)):
        raise ValueError("string array must be an array")
    return _validate_strings(tuple(values), "string array")


def _validate_strings(values: tuple[object, ...], name: str) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        result.append(_identifier(value, name))
    return tuple(result)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversionStoreError(f"{name} must be a non-empty string")
    if any(token in value for token in ("/", "\\", "..")):
        raise ConversionStoreError(f"{name} must be a safe identifier")
    return value


def _optional_identifier(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, name)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversionStoreError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    value = _integer(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _coalesce_time(*values: str | None) -> str | None:
    chosen: str | None = None
    for value in values:
        if value is None:
            continue
        if chosen is not None and value != chosen:
            raise ConversionStoreError("conversion time aliases must identify the same time")
        chosen = value
    return chosen


__all__ = [
    "ConversionRecord",
    "ConversionStoreError",
    "apply_proposal_conversion",
    "confirm_proposal_conversion",
    "load_proposal_conversion",
    "load_proposal_conversions",
]
