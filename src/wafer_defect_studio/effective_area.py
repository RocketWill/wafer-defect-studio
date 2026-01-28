"""Persisted effective-wafer-area geometry for each imported image."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .project import (
    _EFFECTIVE_WAFER_AREAS_TABLE_SQL,
    _IMAGE_GRID_PLACEMENT_SCHEMA_VERSION,
    _SCHEMA_VERSION,
    ProjectError,
    open_project,
)


@dataclass(frozen=True)
class EllipseGeometry:
    center_x: int
    center_y: int
    radius_x: int
    radius_y: int


@dataclass(frozen=True)
class EffectiveWaferArea:
    image_asset_id: str
    shape: str
    geometry: EllipseGeometry
    confirmed: bool


class EffectiveWaferAreaError(ProjectError):
    """Raised when an effective-wafer-area record is invalid or unavailable."""


def set_effective_ellipse(
    project_path: str | Path,
    image_asset_id: str,
    center_x: int,
    center_y: int,
    radius_x: int,
    radius_y: int,
) -> EffectiveWaferArea:
    """Replace the current image area with an unconfirmed source-aligned ellipse."""

    _identifier(image_asset_id)
    _integer("center_x", center_x)
    _integer("center_y", center_y)
    _integer("radius_x", radius_x)
    _integer("radius_y", radius_y)
    if radius_x <= 0 or radius_y <= 0:
        raise ValueError("ellipse radii must be positive")

    project_info = open_project(project_path)
    if project_info.schema_version < _IMAGE_GRID_PLACEMENT_SCHEMA_VERSION:
        raise EffectiveWaferAreaError("Effective wafer areas require project schema 5 or newer")

    database_path = project_info.path / "project.sqlite"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]

        image = connection.execute(
            "SELECT width, height FROM image_assets WHERE image_asset_id = ?",
            (image_asset_id,),
        ).fetchone()
        if image is None:
            raise EffectiveWaferAreaError(f"Unknown image asset: {image_asset_id}")
        image_width, image_height = int(image[0]), int(image[1])
        if (
            center_x - radius_x < 0
            or center_y - radius_y < 0
            or center_x + radius_x > image_width
            or center_y + radius_y > image_height
        ):
            raise ValueError("ellipse must lie wholly within the source image")

        if schema_version == _IMAGE_GRID_PLACEMENT_SCHEMA_VERSION:
            connection.execute(_EFFECTIVE_WAFER_AREAS_TABLE_SQL)
            updated = connection.execute(
                "UPDATE project_metadata SET schema_version = ? "
                "WHERE project_id = ? AND schema_version = 5",
                (_SCHEMA_VERSION, project_info.project_id),
            ).rowcount
            if updated != 1:
                raise EffectiveWaferAreaError(f"Invalid project metadata: {database_path}")
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        elif schema_version != _SCHEMA_VERSION:
            raise EffectiveWaferAreaError(f"Unsupported project schema: {database_path}")

        geometry = EllipseGeometry(center_x, center_y, radius_x, radius_y)
        connection.execute(
            "INSERT INTO effective_wafer_areas "
            "(image_asset_id, shape, geometry_json, confirmed) "
            "VALUES (?, 'ellipse', ?, 0) "
            "ON CONFLICT(image_asset_id) DO UPDATE SET "
            "shape = excluded.shape, geometry_json = excluded.geometry_json, "
            "confirmed = excluded.confirmed",
            (image_asset_id, _geometry_json(geometry)),
        )
        connection.commit()
    except sqlite3.Error as error:
        connection.rollback()
        raise EffectiveWaferAreaError(
            f"Invalid effective wafer area metadata: {database_path}"
        ) from error
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    return EffectiveWaferArea(image_asset_id, "ellipse", geometry, False)


def load_effective_wafer_area(
    project_path: str | Path, image_asset_id: str
) -> EffectiveWaferArea | None:
    """Read the current image area without writing or migrating the project."""

    _identifier(image_asset_id)
    project_info = open_project(project_path)
    if project_info.schema_version < _IMAGE_GRID_PLACEMENT_SCHEMA_VERSION:
        raise EffectiveWaferAreaError("Effective wafer areas require project schema 5 or newer")
    database_path = project_info.path / "project.sqlite"
    try:
        connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise EffectiveWaferAreaError(f"Cannot open project database: {database_path}") from error
    try:
        image = connection.execute(
            "SELECT width, height FROM image_assets WHERE image_asset_id = ?",
            (image_asset_id,),
        ).fetchone()
        if image is None:
            raise EffectiveWaferAreaError(f"Unknown image asset: {image_asset_id}")
        if project_info.schema_version == _IMAGE_GRID_PLACEMENT_SCHEMA_VERSION:
            return None
        row = connection.execute(
            "SELECT image_asset_id, shape, geometry_json, confirmed "
            "FROM effective_wafer_areas WHERE image_asset_id = ?",
            (image_asset_id,),
        ).fetchone()
    except sqlite3.Error as error:
        raise EffectiveWaferAreaError(
            f"Invalid effective wafer area metadata: {database_path}"
        ) from error
    finally:
        connection.close()

    if row is None:
        return None
    persisted_id, shape, geometry_json, confirmed = row
    if shape != "ellipse":
        raise EffectiveWaferAreaError(f"Unsupported effective wafer area shape: {shape}")
    try:
        values = json.loads(geometry_json)
        if not isinstance(values, dict) or tuple(values) != (
            "center_x",
            "center_y",
            "radius_x",
            "radius_y",
        ):
            raise ValueError
        geometry = EllipseGeometry(
            _stored_integer(values["center_x"]),
            _stored_integer(values["center_y"]),
            _stored_integer(values["radius_x"]),
            _stored_integer(values["radius_y"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise EffectiveWaferAreaError(
            f"Invalid effective wafer area geometry: {database_path}"
        ) from error
    if (
        geometry.radius_x <= 0
        or geometry.radius_y <= 0
        or geometry.center_x - geometry.radius_x < 0
        or geometry.center_y - geometry.radius_y < 0
        or geometry.center_x + geometry.radius_x > int(image[0])
        or geometry.center_y + geometry.radius_y > int(image[1])
    ):
        raise EffectiveWaferAreaError(
            f"Invalid effective wafer area bounds: {database_path}"
        )
    if confirmed not in (0, 1):
        raise EffectiveWaferAreaError(f"Invalid effective wafer area confirmation: {database_path}")
    return EffectiveWaferArea(str(persisted_id), str(shape), geometry, bool(confirmed))


def _geometry_json(geometry: EllipseGeometry) -> str:
    return json.dumps(
        {
            "center_x": geometry.center_x,
            "center_y": geometry.center_y,
            "radius_x": geometry.radius_x,
            "radius_y": geometry.radius_y,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _identifier(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("image_asset_id must be a non-empty string")


def _integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _stored_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("geometry values must be integers")
    return value
