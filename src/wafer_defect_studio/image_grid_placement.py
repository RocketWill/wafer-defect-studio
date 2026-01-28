"""Per-image source-coordinate placement for the current Grid Profile."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .project import (
    _GRID_PROFILE_SCHEMA_VERSION,
    _IMAGE_GRID_PLACEMENTS_TABLE_SQL,
    _IMAGE_GRID_PLACEMENT_SCHEMA_VERSION,
    _SCHEMA_VERSION,
    ProjectError,
    open_project,
)


@dataclass(frozen=True)
class ImageGridPlacement:
    image_asset_id: str
    grid_profile_id: str
    grid_profile_version: int
    origin_x: int
    origin_y: int


class ImageGridPlacementError(ProjectError):
    """Raised when an image-grid placement cannot be read or persisted."""


def set_image_grid_origin(
    project_path: str | Path,
    image_asset_id: str,
    grid_profile_id: str,
    origin_x: int,
    origin_y: int,
) -> ImageGridPlacement:
    """Persist one canonical origin for an image and latest grid profile."""

    _text_identifier("image_asset_id", image_asset_id)
    _text_identifier("grid_profile_id", grid_profile_id)
    _integer("origin_x", origin_x)
    _integer("origin_y", origin_y)

    project_info = open_project(project_path)
    if project_info.schema_version < _GRID_PROFILE_SCHEMA_VERSION:
        raise ImageGridPlacementError("Image grid placement requires project schema 4 or newer")

    database_path = project_info.path / "project.sqlite"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if schema_version == _GRID_PROFILE_SCHEMA_VERSION:
            connection.execute(_IMAGE_GRID_PLACEMENTS_TABLE_SQL)
            updated = connection.execute(
                "UPDATE project_metadata SET schema_version = ? "
                "WHERE project_id = ? AND schema_version = 4",
                (_IMAGE_GRID_PLACEMENT_SCHEMA_VERSION, project_info.project_id),
            ).rowcount
            if updated != 1:
                raise ImageGridPlacementError(f"Invalid project metadata: {database_path}")
            connection.execute(f"PRAGMA user_version = {_IMAGE_GRID_PLACEMENT_SCHEMA_VERSION}")
        elif schema_version < _GRID_PROFILE_SCHEMA_VERSION or schema_version > _SCHEMA_VERSION:
            raise ImageGridPlacementError(f"Unsupported project schema: {database_path}")

        profile = connection.execute(
            "SELECT grid_profile_id, version, cell_width, cell_height "
            "FROM grid_profiles WHERE grid_profile_id = ? "
            "ORDER BY version DESC LIMIT 1",
            (grid_profile_id,),
        ).fetchone()
        if profile is None:
            raise ImageGridPlacementError(f"Unknown grid profile: {grid_profile_id}")
        if origin_x < 0 or origin_y < 0 or origin_x >= profile[2] or origin_y >= profile[3]:
            raise ValueError("origin must be canonical for the grid profile")

        image_exists = connection.execute(
            "SELECT 1 FROM image_assets WHERE image_asset_id = ?",
            (image_asset_id,),
        ).fetchone()
        if image_exists is None:
            raise ImageGridPlacementError(f"Unknown image asset: {image_asset_id}")

        connection.execute(
            "INSERT INTO image_grid_placements "
            "(image_asset_id, grid_profile_id, grid_profile_version, origin_x, origin_y) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(image_asset_id) DO UPDATE SET "
            "grid_profile_id = excluded.grid_profile_id, "
            "grid_profile_version = excluded.grid_profile_version, "
            "origin_x = excluded.origin_x, origin_y = excluded.origin_y",
            (image_asset_id, grid_profile_id, profile[1], origin_x, origin_y),
        )
        connection.commit()
    except sqlite3.Error as error:
        connection.rollback()
        raise ImageGridPlacementError(
            f"Invalid image grid placement metadata: {database_path}"
        ) from error
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    return ImageGridPlacement(image_asset_id, grid_profile_id, profile[1], origin_x, origin_y)


def load_image_grid_placement(
    project_path: str | Path, image_asset_id: str
) -> ImageGridPlacement | None:
    """Read one image placement without writing or migrating the project."""

    _text_identifier("image_asset_id", image_asset_id)
    project_info = open_project(project_path)
    if project_info.schema_version < _GRID_PROFILE_SCHEMA_VERSION:
        raise ImageGridPlacementError("Image grid placement requires project schema 4 or newer")

    database_path = project_info.path / "project.sqlite"
    try:
        connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise ImageGridPlacementError(f"Cannot open project database: {database_path}") from error
    try:
        image_exists = connection.execute(
            "SELECT 1 FROM image_assets WHERE image_asset_id = ?",
            (image_asset_id,),
        ).fetchone()
        if image_exists is None:
            raise ImageGridPlacementError(f"Unknown image asset: {image_asset_id}")
        if project_info.schema_version == _GRID_PROFILE_SCHEMA_VERSION:
            return None
        row = connection.execute(
            "SELECT image_asset_id, grid_profile_id, grid_profile_version, origin_x, origin_y "
            "FROM image_grid_placements WHERE image_asset_id = ?",
            (image_asset_id,),
        ).fetchone()
    except sqlite3.Error as error:
        raise ImageGridPlacementError(
            f"Invalid image grid placement metadata: {database_path}"
        ) from error
    finally:
        connection.close()

    return ImageGridPlacement(*row) if row is not None else None


def _text_identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
