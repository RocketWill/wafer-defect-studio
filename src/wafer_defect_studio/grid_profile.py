"""Append-only, pixel-sized annotation-grid profiles."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from .project import (
    ProjectError,
    _GRID_PROFILES_TABLE_SQL,
    _SCHEMA_VERSION,
    open_project,
)


class GridProfileError(ProjectError):
    """Raised when a grid profile cannot be read or saved."""


class GridProfileConflictError(GridProfileError):
    """Raised when a revision is based on a stale or unchanged profile."""


@dataclass(frozen=True)
class GridProfile:
    """One immutable version of a project's annotation-grid dimensions."""

    grid_profile_id: str
    version: int
    cell_width: int
    cell_height: int


def save_grid_profile(
    project_path: str | Path,
    cell_width: int,
    cell_height: int,
    previous: GridProfile | None = None,
) -> GridProfile:
    """Create a profile or append one revision after the persisted latest version."""

    _positive_integer("cell_width", cell_width)
    _positive_integer("cell_height", cell_height)
    if previous is not None and not isinstance(previous, GridProfile):
        raise GridProfileConflictError("previous must be a GridProfile")

    project_info = open_project(project_path)
    if project_info.schema_version < 3:
        raise GridProfileError("Grid profiles require project schema 3 or newer")

    database_path = project_info.path / "project.sqlite"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if schema_version == 3:
            connection.execute(_GRID_PROFILES_TABLE_SQL)
            updated = connection.execute(
                "UPDATE project_metadata SET schema_version = ? "
                "WHERE project_id = ? AND schema_version = 3",
                (_SCHEMA_VERSION, project_info.project_id),
            ).rowcount
            if updated != 1:
                raise GridProfileError(f"Invalid project metadata: {database_path}")
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        elif schema_version != _SCHEMA_VERSION:
            raise GridProfileError(f"Unsupported project schema: {database_path}")

        if previous is None:
            profile_id = str(uuid.uuid4())
            version = 1
        else:
            latest = connection.execute(
                "SELECT grid_profile_id, version, cell_width, cell_height "
                "FROM grid_profiles WHERE grid_profile_id = ? "
                "ORDER BY version DESC LIMIT 1",
                (previous.grid_profile_id,),
            ).fetchone()
            expected = (
                previous.grid_profile_id,
                previous.version,
                previous.cell_width,
                previous.cell_height,
            )
            if latest is None or tuple(latest) != expected:
                raise GridProfileConflictError("previous profile is not the persisted latest version")
            if (cell_width, cell_height) == (previous.cell_width, previous.cell_height):
                raise ValueError("grid profile dimensions are unchanged")
            profile_id = previous.grid_profile_id
            version = previous.version + 1

        connection.execute(
            "INSERT INTO grid_profiles "
            "(grid_profile_id, version, cell_width, cell_height) VALUES (?, ?, ?, ?)",
            (profile_id, version, cell_width, cell_height),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    return GridProfile(profile_id, version, cell_width, cell_height)


def load_grid_profiles(project_path: str | Path) -> tuple[GridProfile, ...]:
    """Reopen every persisted profile version in deterministic order."""

    project_info = open_project(project_path)
    if project_info.schema_version < _SCHEMA_VERSION:
        return ()

    database_path = project_info.path / "project.sqlite"
    try:
        connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise GridProfileError(f"Cannot open project database: {database_path}") from error
    try:
        rows = connection.execute(
            "SELECT grid_profile_id, version, cell_width, cell_height "
            "FROM grid_profiles ORDER BY grid_profile_id ASC, version ASC"
        ).fetchall()
    except sqlite3.Error as error:
        raise GridProfileError(f"Invalid grid profile metadata: {database_path}") from error
    finally:
        connection.close()

    return tuple(GridProfile(*row) for row in rows)


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
