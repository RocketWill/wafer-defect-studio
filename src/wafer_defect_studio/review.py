"""Persist image review state and derive Normal Grids from reviewed images."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .effective_area import confirmed_participating_grids, load_effective_wafer_area
from .grid_geometry import AnnotationGrid, annotation_grids
from .grid_profile import load_grid_profiles
from .image_grid_placement import load_image_grid_placement
from .project import (
    _GRID_ANNOTATION_SCHEMA_VERSION,
    _IMAGE_REVIEWS_TABLE_SQL,
    _REVIEW_SCHEMA_VERSION,
    _TRAINING_SCOPE_SCHEMA_VERSION,
    _DATASET_SNAPSHOT_SCHEMA_VERSION,
    _TRAINING_RUN_SCHEMA_VERSION,
    _EVALUATION_SCHEMA_VERSION,
    _DETECTION_SCHEMA_VERSION,
    ProjectError,
    open_project,
)


class ReviewError(ProjectError):
    """Raised when a Wafer Image cannot enter or leave Reviewed state."""


@dataclass(frozen=True)
class ReviewState:
    """Whether one Wafer Image is explicitly Reviewed."""

    image_asset_id: str
    reviewed: bool


@dataclass(frozen=True)
class _ReviewContext:
    project_path: Path
    image_asset_id: str
    grids: tuple[AnnotationGrid, ...]
    all_grids: tuple[AnnotationGrid, ...]


def mark_image_reviewed(project_path: str | Path, image_asset_id: str) -> ReviewState:
    """Confirm that an image's current Grid Annotations have been reviewed."""

    _load_review_context(project_path, image_asset_id)
    return _write_review_state(project_path, image_asset_id, True)


def reopen_image(project_path: str | Path, image_asset_id: str) -> ReviewState:
    """Return an image to Wafer Image Under Review without changing annotations."""

    _load_review_context(project_path, image_asset_id)
    return _write_review_state(project_path, image_asset_id, False)


def load_review_state(project_path: str | Path, image_asset_id: str) -> ReviewState:
    """Read an image review flag without creating or migrating project data."""

    _validate_identifier(image_asset_id)
    project_info = open_project(project_path)
    if project_info.schema_version < _GRID_ANNOTATION_SCHEMA_VERSION:
        raise ReviewError("Image review requires project schema 8 or newer")

    database_path = project_info.path / "project.sqlite"
    try:
        connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise ReviewError(f"Cannot open project database: {database_path}") from error
    try:
        _require_image(connection, image_asset_id, database_path)
        if project_info.schema_version < _REVIEW_SCHEMA_VERSION:
            return ReviewState(image_asset_id, False)
        row = connection.execute(
            "SELECT reviewed FROM image_reviews WHERE image_asset_id = ?",
            (image_asset_id,),
        ).fetchone()
    except sqlite3.Error as error:
        raise ReviewError(f"Invalid image review metadata: {database_path}") from error
    finally:
        connection.close()

    if row is None:
        return ReviewState(image_asset_id, False)
    if row[0] not in (0, 1):
        raise ReviewError(f"Invalid image review state: {database_path}")
    return ReviewState(image_asset_id, bool(row[0]))


def derived_normal_grids(
    project_path: str | Path, image_asset_id: str
) -> tuple[AnnotationGrid, ...]:
    """Return in-area, unlabeled grids only while an image is Reviewed."""

    context = _load_review_context(project_path, image_asset_id)
    if not load_review_state(context.project_path, image_asset_id).reviewed:
        return ()
    explicit = _explicit_annotation_cells(context.project_path, image_asset_id)
    return tuple(
        grid for grid in context.grids if (grid.row, grid.column) not in explicit
    )


def unreviewed_grids(
    project_path: str | Path, image_asset_id: str
) -> tuple[AnnotationGrid, ...]:
    """Return in-area, unlabeled grids while an image remains under review."""

    context = _load_review_context(project_path, image_asset_id)
    if load_review_state(context.project_path, image_asset_id).reviewed:
        return ()
    explicit = _explicit_annotation_cells(context.project_path, image_asset_id)
    return tuple(
        grid for grid in context.grids if (grid.row, grid.column) not in explicit
    )


def _load_review_context(
    project_path: str | Path, image_asset_id: str
) -> _ReviewContext:
    _validate_identifier(image_asset_id)
    project_info = open_project(project_path)
    if project_info.schema_version < _GRID_ANNOTATION_SCHEMA_VERSION:
        raise ReviewError("Image review requires project schema 8 or newer")

    database_path = project_info.path / "project.sqlite"
    try:
        connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise ReviewError(f"Cannot open project database: {database_path}") from error
    try:
        image = connection.execute(
            "SELECT width, height FROM image_assets WHERE image_asset_id = ?",
            (image_asset_id,),
        ).fetchone()
    except sqlite3.Error as error:
        raise ReviewError(f"Invalid image asset metadata: {database_path}") from error
    finally:
        connection.close()
    if image is None:
        raise ReviewError(f"Unknown image asset: {image_asset_id}")

    placement = load_image_grid_placement(project_info.path, image_asset_id)
    if placement is None:
        raise ReviewError(f"Missing grid placement for image: {image_asset_id}")
    profile = next(
        (
            candidate
            for candidate in load_grid_profiles(project_info.path)
            if candidate.grid_profile_id == placement.grid_profile_id
            and candidate.version == placement.grid_profile_version
        ),
        None,
    )
    if profile is None:
        raise ReviewError(f"Missing grid profile for image: {image_asset_id}")

    area = load_effective_wafer_area(project_info.path, image_asset_id)
    if area is None:
        raise ReviewError(f"Missing Effective Wafer Area for image: {image_asset_id}")
    try:
        grids = annotation_grids(
            int(image[0]),
            int(image[1]),
            profile.cell_width,
            profile.cell_height,
            origin_x=placement.origin_x,
            origin_y=placement.origin_y,
        )
        participating = confirmed_participating_grids(grids, area)
    except (TypeError, ValueError) as error:
        raise ReviewError(f"Invalid review geometry for image: {image_asset_id}") from error
    except ProjectError as error:
        raise ReviewError(str(error)) from error
    return _ReviewContext(project_info.path, image_asset_id, participating, grids)


def _write_review_state(
    project_path: str | Path, image_asset_id: str, reviewed: bool
) -> ReviewState:
    project_info = open_project(project_path)
    if project_info.schema_version < _GRID_ANNOTATION_SCHEMA_VERSION:
        raise ReviewError("Image review requires project schema 8 or newer")
    if project_info.schema_version > _DETECTION_SCHEMA_VERSION:
        raise ReviewError(f"Unsupported project schema: {project_info.path / 'project.sqlite'}")

    database_path = project_info.path / "project.sqlite"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if schema_version == _GRID_ANNOTATION_SCHEMA_VERSION:
            connection.execute(_IMAGE_REVIEWS_TABLE_SQL)
            updated = connection.execute(
                "UPDATE project_metadata SET schema_version = ? "
                "WHERE project_id = ? AND schema_version = ?",
                (_REVIEW_SCHEMA_VERSION, project_info.project_id, _GRID_ANNOTATION_SCHEMA_VERSION),
            ).rowcount
            if updated != 1:
                raise ReviewError(f"Invalid project metadata: {database_path}")
            connection.execute(f"PRAGMA user_version = {_REVIEW_SCHEMA_VERSION}")
        elif schema_version not in (
            _REVIEW_SCHEMA_VERSION,
            _TRAINING_SCOPE_SCHEMA_VERSION,
            _DATASET_SNAPSHOT_SCHEMA_VERSION,
            _TRAINING_RUN_SCHEMA_VERSION,
            _EVALUATION_SCHEMA_VERSION,
            _DETECTION_SCHEMA_VERSION,
        ):
            raise ReviewError(f"Unsupported project schema: {database_path}")

        _require_image(connection, image_asset_id, database_path)
        connection.execute(
            "INSERT INTO image_reviews (image_asset_id, reviewed) VALUES (?, ?) "
            "ON CONFLICT(image_asset_id) DO UPDATE SET reviewed = excluded.reviewed",
            (image_asset_id, int(reviewed)),
        )
        connection.commit()
    except sqlite3.Error as error:
        connection.rollback()
        raise ReviewError(f"Invalid image review metadata: {database_path}") from error
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return ReviewState(image_asset_id, reviewed)


def _explicit_annotation_cells(project_path: Path, image_asset_id: str) -> set[tuple[int, int]]:
    database_path = project_path / "project.sqlite"
    try:
        connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise ReviewError(f"Cannot open project database: {database_path}") from error
    try:
        rows = connection.execute(
            "SELECT row, column, class_codes_json FROM grid_annotations "
            "WHERE image_asset_id = ?",
            (image_asset_id,),
        ).fetchall()
    except sqlite3.Error as error:
        raise ReviewError(f"Invalid Grid Annotation metadata: {database_path}") from error
    finally:
        connection.close()

    explicit = set()
    for row, column, serialized in rows:
        try:
            class_codes = json.loads(serialized)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ReviewError(f"Invalid Grid Annotation metadata: {database_path}") from error
        if not isinstance(class_codes, list):
            raise ReviewError(f"Invalid Grid Annotation metadata: {database_path}")
        if class_codes:
            explicit.add((int(row), int(column)))
    return explicit


def _require_image(
    connection: sqlite3.Connection, image_asset_id: str, database_path: Path
) -> None:
    if connection.execute(
        "SELECT 1 FROM image_assets WHERE image_asset_id = ?", (image_asset_id,)
    ).fetchone() is None:
        raise ReviewError(f"Unknown image asset: {image_asset_id}")


def _validate_identifier(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("image_asset_id must be a non-empty string")
