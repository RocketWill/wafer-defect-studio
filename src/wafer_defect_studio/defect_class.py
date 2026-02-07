"""Persist ordered project Defect Class definitions."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .project import (
    _DEFECT_CLASS_SCHEMA_VERSION,
    _DEFECT_CLASSES_TABLE_SQL,
    _GRID_ANNOTATION_SCHEMA_VERSION,
    _REVIEW_SCHEMA_VERSION,
    _TRAINING_SCOPE_SCHEMA_VERSION,
    _DATASET_SNAPSHOT_SCHEMA_VERSION,
    _TRAINING_RUN_SCHEMA_VERSION,
    _SCHEMA_VERSION,
    ProjectError,
    open_project,
)


class DefectClassError(ProjectError):
    """Raised when Defect Class metadata cannot be read or saved."""


@dataclass(frozen=True)
class DefectClass:
    """A project class definition retained by stable code."""

    code: str
    name: str
    color: str
    icon: str = ""
    description: str = ""
    order: int = 0
    enabled: bool = True

    @property
    def display_order(self) -> int:
        """Return the persisted ordering value."""

        return self.order


def save_defect_classes(
    project_path: str | Path,
    classes: Iterable[DefectClass],
) -> None:
    """Create or update an ordered set of Defect Classes atomically."""

    values = tuple(classes)
    _validate_classes(values)

    project_info = open_project(project_path)
    if project_info.schema_version < _SCHEMA_VERSION:
        raise DefectClassError("Defect classes require project schema 6 or newer")
    if project_info.schema_version > _TRAINING_RUN_SCHEMA_VERSION:
        raise DefectClassError(f"Unsupported project schema: {project_info.path / 'project.sqlite'}")

    database_path = project_info.path / "project.sqlite"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if schema_version == _SCHEMA_VERSION:
            connection.execute(_DEFECT_CLASSES_TABLE_SQL)
            updated = connection.execute(
                "UPDATE project_metadata SET schema_version = ? "
                "WHERE project_id = ? AND schema_version = ?",
                (_DEFECT_CLASS_SCHEMA_VERSION, project_info.project_id, _SCHEMA_VERSION),
            ).rowcount
            if updated != 1:
                raise DefectClassError(f"Invalid project metadata: {database_path}")
            connection.execute(f"PRAGMA user_version = {_DEFECT_CLASS_SCHEMA_VERSION}")
        elif schema_version not in (
            _DEFECT_CLASS_SCHEMA_VERSION,
            _GRID_ANNOTATION_SCHEMA_VERSION,
            _REVIEW_SCHEMA_VERSION,
            _TRAINING_SCOPE_SCHEMA_VERSION,
            _DATASET_SNAPSHOT_SCHEMA_VERSION,
            _TRAINING_RUN_SCHEMA_VERSION,
        ):
            raise DefectClassError(f"Unsupported project schema: {database_path}")

        connection.executemany(
            "INSERT INTO defect_classes "
            "(code, name, color, icon, description, display_order, enabled) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(code) DO UPDATE SET "
            "name = excluded.name, color = excluded.color, icon = excluded.icon, "
            "description = excluded.description, display_order = excluded.display_order, "
            "enabled = excluded.enabled",
            (
                (
                    defect_class.code,
                    defect_class.name,
                    defect_class.color,
                    defect_class.icon,
                    defect_class.description,
                    defect_class.order,
                    int(defect_class.enabled),
                )
                for defect_class in values
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def load_defect_classes(project_path: str | Path) -> tuple[DefectClass, ...]:
    """Reopen every persisted Defect Class in display order."""

    project_info = open_project(project_path)
    if project_info.schema_version < _DEFECT_CLASS_SCHEMA_VERSION:
        return ()
    if project_info.schema_version > _TRAINING_RUN_SCHEMA_VERSION:
        raise DefectClassError(f"Unsupported project schema: {project_info.path / 'project.sqlite'}")

    database_path = project_info.path / "project.sqlite"
    try:
        connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise DefectClassError(f"Cannot open project database: {database_path}") from error
    try:
        rows = connection.execute(
            "SELECT code, name, color, icon, description, display_order, enabled "
            "FROM defect_classes ORDER BY display_order ASC, code ASC"
        ).fetchall()
    except sqlite3.Error as error:
        raise DefectClassError(f"Invalid defect class metadata: {database_path}") from error
    finally:
        connection.close()

    return tuple(
        DefectClass(
            code=row[0],
            name=row[1],
            color=row[2],
            icon=row[3],
            description=row[4],
            order=row[5],
            enabled=bool(row[6]),
        )
        for row in rows
    )


def _validate_classes(classes: tuple[DefectClass, ...]) -> None:
    codes: set[str] = set()
    for defect_class in classes:
        if not isinstance(defect_class, DefectClass):
            raise ValueError("classes must contain DefectClass values")
        for field_name in ("code", "name", "color", "icon", "description"):
            value = getattr(defect_class, field_name)
            if not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string")
        if not defect_class.code.strip():
            raise ValueError("code must not be empty")
        if not defect_class.name.strip():
            raise ValueError("name must not be empty")
        if not defect_class.color.strip():
            raise ValueError("color must not be empty")
        if defect_class.code in codes:
            raise ValueError(f"duplicate Defect Class code: {defect_class.code}")
        codes.add(defect_class.code)
        if isinstance(defect_class.order, bool) or not isinstance(defect_class.order, int):
            raise ValueError("order must be a non-negative integer")
        if defect_class.order < 0:
            raise ValueError("order must be a non-negative integer")
        if not isinstance(defect_class.enabled, bool):
            raise ValueError("enabled must be a boolean")
