"""Persist source-aligned multi-label Grid Annotations."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .project import (
    _DEFECT_CLASS_SCHEMA_VERSION,
    _GRID_ANNOTATIONS_TABLE_SQL,
    _GRID_ANNOTATION_SCHEMA_VERSION,
    _REVIEW_SCHEMA_VERSION,
    _TRAINING_SCOPE_SCHEMA_VERSION,
    _DATASET_SNAPSHOT_SCHEMA_VERSION,
    _TRAINING_RUN_SCHEMA_VERSION,
    _EVALUATION_SCHEMA_VERSION,
    ProjectError,
    open_project,
)


class GridAnnotationError(ProjectError):
    """Raised when a Grid Annotation cannot be read or persisted."""


@dataclass(frozen=True)
class GridAnnotation:
    """The ordered Defect Class codes asserted for one Annotation Grid."""

    image_asset_id: str
    row: int
    column: int
    class_codes: tuple[str, ...]


def save_grid_annotation(
    project_path: str | Path,
    annotation: GridAnnotation,
) -> None:
    """Atomically replace one cell's ordered Defect Class code set."""

    _validate_annotation(annotation)
    project_info = open_project(project_path)
    if project_info.schema_version < _DEFECT_CLASS_SCHEMA_VERSION:
        raise GridAnnotationError("Grid Annotations require project schema 7 or newer")
    if project_info.schema_version > _EVALUATION_SCHEMA_VERSION:
        raise GridAnnotationError(
            f"Unsupported project schema: {project_info.path / 'project.sqlite'}"
        )

    database_path = project_info.path / "project.sqlite"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if schema_version < _DEFECT_CLASS_SCHEMA_VERSION:
            raise GridAnnotationError("Grid Annotations require project schema 7 or newer")
        if schema_version > _EVALUATION_SCHEMA_VERSION:
            raise GridAnnotationError(f"Unsupported project schema: {database_path}")

        image_exists = connection.execute(
            "SELECT 1 FROM image_assets WHERE image_asset_id = ?",
            (annotation.image_asset_id,),
        ).fetchone()
        if image_exists is None:
            raise GridAnnotationError(f"Unknown image asset: {annotation.image_asset_id}")

        class_rows = connection.execute(
            "SELECT code, enabled FROM defect_classes"
        ).fetchall()
        class_state = {row[0]: bool(row[1]) for row in class_rows}
        for code in annotation.class_codes:
            if code not in class_state:
                raise GridAnnotationError(f"Unknown Defect Class code: {code}")
            if not class_state[code]:
                raise GridAnnotationError(f"Archived Defect Class code: {code}")

        if schema_version == _DEFECT_CLASS_SCHEMA_VERSION:
            connection.execute(_GRID_ANNOTATIONS_TABLE_SQL)
            updated = connection.execute(
                "UPDATE project_metadata SET schema_version = ? "
                "WHERE project_id = ? AND schema_version = ?",
                (
                    _GRID_ANNOTATION_SCHEMA_VERSION,
                    project_info.project_id,
                    _DEFECT_CLASS_SCHEMA_VERSION,
                ),
            ).rowcount
            if updated != 1:
                raise GridAnnotationError(f"Invalid project metadata: {database_path}")
            connection.execute(f"PRAGMA user_version = {_GRID_ANNOTATION_SCHEMA_VERSION}")

        connection.execute(
            "INSERT INTO grid_annotations "
            "(image_asset_id, row, column, class_codes_json) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(image_asset_id, row, column) DO UPDATE SET "
            "class_codes_json = excluded.class_codes_json",
            (
                annotation.image_asset_id,
                annotation.row,
                annotation.column,
                _encode_codes(annotation.class_codes),
            ),
        )
        connection.commit()
    except sqlite3.Error as error:
        connection.rollback()
        raise GridAnnotationError(f"Invalid Grid Annotation metadata: {database_path}") from error
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def load_grid_annotation(
    project_path: str | Path,
    image_asset_id: str,
    row: int,
    column: int,
) -> GridAnnotation | None:
    """Read one cell without creating a table or migrating the project."""

    _validate_cell(image_asset_id, row, column)
    project_info = open_project(project_path)
    if project_info.schema_version < _DEFECT_CLASS_SCHEMA_VERSION:
        raise GridAnnotationError("Grid Annotations require project schema 7 or newer")
    if project_info.schema_version > _EVALUATION_SCHEMA_VERSION:
        raise GridAnnotationError(
            f"Unsupported project schema: {project_info.path / 'project.sqlite'}"
        )
    if project_info.schema_version == _DEFECT_CLASS_SCHEMA_VERSION:
        return None

    database_path = project_info.path / "project.sqlite"
    try:
        connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise GridAnnotationError(f"Cannot open project database: {database_path}") from error
    try:
        image_exists = connection.execute(
            "SELECT 1 FROM image_assets WHERE image_asset_id = ?",
            (image_asset_id,),
        ).fetchone()
        if image_exists is None:
            raise GridAnnotationError(f"Unknown image asset: {image_asset_id}")
        persisted = connection.execute(
            "SELECT image_asset_id, row, column, class_codes_json "
            "FROM grid_annotations WHERE image_asset_id = ? AND row = ? AND column = ?",
            (image_asset_id, row, column),
        ).fetchone()
    except sqlite3.Error as error:
        raise GridAnnotationError(f"Invalid Grid Annotation metadata: {database_path}") from error
    finally:
        connection.close()

    if persisted is None:
        return None
    try:
        class_codes = _decode_codes(persisted[3])
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise GridAnnotationError(f"Invalid Grid Annotation metadata: {database_path}") from error
    return GridAnnotation(persisted[0], persisted[1], persisted[2], class_codes)


def _validate_annotation(annotation: object) -> None:
    if not isinstance(annotation, GridAnnotation):
        raise ValueError("annotation must be a GridAnnotation")
    _validate_cell(annotation.image_asset_id, annotation.row, annotation.column)
    if not isinstance(annotation.class_codes, tuple):
        raise ValueError("class_codes must be a tuple")
    _validate_codes(annotation.class_codes)


def _validate_cell(image_asset_id: object, row: object, column: object) -> None:
    if not isinstance(image_asset_id, str) or not image_asset_id:
        raise ValueError("image_asset_id must be a non-empty string")
    if isinstance(row, bool) or not isinstance(row, int):
        raise ValueError("row must be an integer")
    if isinstance(column, bool) or not isinstance(column, int):
        raise ValueError("column must be an integer")


def _validate_codes(class_codes: object) -> None:
    if not isinstance(class_codes, tuple):
        raise ValueError("class_codes must be a tuple")
    seen: set[str] = set()
    for code in class_codes:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("class codes must be non-empty strings")
        if code in seen:
            raise ValueError(f"duplicate Defect Class code: {code}")
        seen.add(code)


def _encode_codes(class_codes: tuple[str, ...]) -> str:
    return json.dumps(class_codes, separators=(",", ":"))


def _decode_codes(serialized: object) -> tuple[str, ...]:
    values = json.loads(serialized)
    if not isinstance(values, list):
        raise ValueError("class codes must be a JSON array")
    class_codes = tuple(values)
    _validate_codes(class_codes)
    return class_codes
