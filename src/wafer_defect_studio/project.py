"""Create and reopen the on-disk project shell."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path


_SCHEMA_VERSION = 6
_IMAGE_ASSET_SCHEMA_VERSION = 3
_GRID_PROFILE_SCHEMA_VERSION = 4
_IMAGE_GRID_PLACEMENT_SCHEMA_VERSION = 5
_SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3, 4, 5, _SCHEMA_VERSION)
_DATABASE_NAME = "project.sqlite"
_PROJECT_DIRECTORIES = ("models", "runs", "exports", "backups", "cache")
_IMAGE_ASSETS_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS image_assets ("
    "image_asset_id TEXT NOT NULL PRIMARY KEY, "
    "path TEXT NOT NULL, "
    "width INTEGER NOT NULL, "
    "height INTEGER NOT NULL, "
    "dtype TEXT NOT NULL, "
    "format TEXT NOT NULL, "
    "fingerprint TEXT NOT NULL, "
    "lossy_source INTEGER NOT NULL DEFAULT 0 CHECK (lossy_source IN (0, 1))"
    ")"
)
_GRID_PROFILES_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS grid_profiles ("
    "grid_profile_id TEXT NOT NULL, "
    "version INTEGER NOT NULL CHECK (version >= 1), "
    "cell_width INTEGER NOT NULL CHECK (cell_width > 0), "
    "cell_height INTEGER NOT NULL CHECK (cell_height > 0), "
    "PRIMARY KEY (grid_profile_id, version)"
    ")"
)
_IMAGE_GRID_PLACEMENTS_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS image_grid_placements ("
    "image_asset_id TEXT NOT NULL PRIMARY KEY, "
    "grid_profile_id TEXT NOT NULL, "
    "grid_profile_version INTEGER NOT NULL CHECK (grid_profile_version >= 1), "
    "origin_x INTEGER NOT NULL CHECK (origin_x >= 0), "
    "origin_y INTEGER NOT NULL CHECK (origin_y >= 0), "
    "FOREIGN KEY (image_asset_id) REFERENCES image_assets(image_asset_id), "
    "FOREIGN KEY (grid_profile_id, grid_profile_version) "
    "REFERENCES grid_profiles(grid_profile_id, version)"
    ")"
)
_EFFECTIVE_WAFER_AREAS_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS effective_wafer_areas ("
    "image_asset_id TEXT NOT NULL PRIMARY KEY, "
    "shape TEXT NOT NULL CHECK (shape IN ('ellipse', 'polygon')), "
    "geometry_json TEXT NOT NULL, "
    "confirmed INTEGER NOT NULL DEFAULT 0 CHECK (confirmed IN (0, 1)), "
    "FOREIGN KEY (image_asset_id) REFERENCES image_assets(image_asset_id)"
    ")"
)
_IMAGE_ASSET_COLUMNS_V2 = (
    "image_asset_id",
    "path",
    "width",
    "height",
    "dtype",
    "format",
    "fingerprint",
)
_IMAGE_ASSET_COLUMNS_V3 = _IMAGE_ASSET_COLUMNS_V2 + ("lossy_source",)
_GRID_PROFILE_COLUMNS = ("grid_profile_id", "version", "cell_width", "cell_height")
_IMAGE_GRID_PLACEMENT_COLUMNS = (
    "image_asset_id",
    "grid_profile_id",
    "grid_profile_version",
    "origin_x",
    "origin_y",
)
_EFFECTIVE_WAFER_AREA_COLUMNS = (
    "image_asset_id",
    "shape",
    "geometry_json",
    "confirmed",
)


class ProjectError(ValueError):
    """Raised when a path is not safe to create or is not a valid project."""


@dataclass(frozen=True)
class ProjectInfo:
    """Immutable identity and location of a project."""

    project_id: str
    schema_version: int
    path: Path


def create_project(path: str | Path) -> ProjectInfo:
    """Create a new project at *path* and return its immutable identity."""

    project_path = _resolved_path(path)
    if project_path.exists():
        if not project_path.is_dir() or any(project_path.iterdir()):
            raise ProjectError(f"Refusing to alter non-empty or non-directory path: {project_path}")
    else:
        project_path.mkdir(parents=True)

    for directory_name in _PROJECT_DIRECTORIES:
        (project_path / directory_name).mkdir()

    project_id = str(uuid.uuid4())
    database_path = project_path / _DATABASE_NAME
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        connection.execute(
            "CREATE TABLE project_metadata ("
            "project_id TEXT NOT NULL PRIMARY KEY, "
            "schema_version INTEGER NOT NULL"
            ")"
        )
        connection.execute(
            "INSERT INTO project_metadata(project_id, schema_version) VALUES (?, ?)",
            (project_id, _SCHEMA_VERSION),
        )
        connection.execute(_IMAGE_ASSETS_TABLE_SQL)
        connection.execute(_GRID_PROFILES_TABLE_SQL)
        connection.execute(_IMAGE_GRID_PLACEMENTS_TABLE_SQL)
        connection.execute(_EFFECTIVE_WAFER_AREAS_TABLE_SQL)
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
        raise
    finally:
        connection.close()

    return ProjectInfo(project_id, _SCHEMA_VERSION, project_path)


def open_project(path: str | Path) -> ProjectInfo:
    """Read and validate an existing project without writing to it."""

    project_path = _resolved_path(path)
    if not project_path.exists():
        raise FileNotFoundError(project_path)
    if not project_path.is_dir():
        raise ProjectError(f"Project path is not a directory: {project_path}")

    database_path = project_path / _DATABASE_NAME
    _validate_layout(project_path, database_path)

    connection = _read_only_connection(database_path)
    try:
        pragma_version = connection.execute("PRAGMA user_version").fetchone()[0]
        metadata = connection.execute(
            "SELECT project_id, schema_version FROM project_metadata"
        ).fetchall()
        image_asset_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(image_assets)")
        )
        grid_profile_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(grid_profiles)")
        )
        image_grid_placement_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(image_grid_placements)")
        )
        effective_wafer_area_columns = tuple(
            row[1]
            for row in connection.execute("PRAGMA table_info(effective_wafer_areas)")
        )
    except sqlite3.Error as error:
        raise ProjectError(f"Invalid project database: {database_path}") from error
    finally:
        connection.close()

    if pragma_version not in _SUPPORTED_SCHEMA_VERSIONS or len(metadata) != 1:
        raise ProjectError(f"Unsupported project schema: {database_path}")

    project_id, metadata_version = metadata[0]
    if metadata_version != pragma_version or not project_id:
        raise ProjectError(f"Invalid project metadata: {database_path}")
    if pragma_version == 2 and image_asset_columns != _IMAGE_ASSET_COLUMNS_V2:
        raise ProjectError(f"Invalid image asset table: {database_path}")
    if pragma_version >= _IMAGE_ASSET_SCHEMA_VERSION and image_asset_columns != _IMAGE_ASSET_COLUMNS_V3:
        raise ProjectError(f"Invalid image asset table: {database_path}")
    if pragma_version >= _GRID_PROFILE_SCHEMA_VERSION and grid_profile_columns != _GRID_PROFILE_COLUMNS:
        raise ProjectError(f"Invalid grid profile table: {database_path}")
    if pragma_version >= 5 and image_grid_placement_columns != _IMAGE_GRID_PLACEMENT_COLUMNS:
        raise ProjectError(f"Invalid image grid placement table: {database_path}")
    if pragma_version == _SCHEMA_VERSION and effective_wafer_area_columns != _EFFECTIVE_WAFER_AREA_COLUMNS:
        raise ProjectError(f"Invalid effective wafer area table: {database_path}")

    return ProjectInfo(project_id, pragma_version, project_path)


def _resolved_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _validate_layout(project_path: Path, database_path: Path) -> None:
    if not database_path.is_file():
        raise ProjectError(f"Missing project database: {database_path}")
    for directory_name in _PROJECT_DIRECTORIES:
        directory = project_path / directory_name
        if not directory.is_dir():
            raise ProjectError(f"Missing project directory: {directory}")


def _read_only_connection(database_path: Path) -> sqlite3.Connection:
    try:
        return sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise ProjectError(f"Cannot open project database: {database_path}") from error
