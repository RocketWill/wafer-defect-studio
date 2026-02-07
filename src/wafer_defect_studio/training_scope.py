"""Persist the Training Scope and derive currently eligible images."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .image_asset import SourceHealth, load_image_assets
from .project import (
    _DATA_GROUPS_TABLE_SQL,
    _IMAGE_DATA_GROUPS_TABLE_SQL,
    _REVIEW_SCHEMA_VERSION,
    _DATASET_SNAPSHOT_SCHEMA_VERSION,
    _DATASET_SPLIT_SCHEMA_VERSION,
    _TRAINING_RUN_SCHEMA_VERSION,
    _TRAINING_SCOPE_SCHEMA_VERSION,
    _TRAINING_SCOPE_TABLE_SQL,
    ProjectError,
    open_project,
)


class TrainingScopeError(ProjectError):
    pass


@dataclass(frozen=True)
class DataGroup:
    data_group_id: str
    name: str
    order: int = 0


@dataclass(frozen=True)
class TrainingScope:
    data_group_ids: tuple[str, ...]
    class_codes: tuple[str, ...]


def save_data_groups(project_path: str | Path, groups: Iterable[DataGroup]) -> None:
    values = tuple(groups)
    _validate_groups(values)
    project_info = open_project(project_path)
    database_path = project_info.path / "project.sqlite"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        _ensure_schema(connection, project_info.project_id, database_path)
        connection.executemany(
            "INSERT INTO data_groups VALUES (?, ?, ?) ON CONFLICT(data_group_id) DO UPDATE SET "
            "name = excluded.name, display_order = excluded.display_order",
            ((group.data_group_id, group.name, group.order) for group in values),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def assign_image_to_data_group(
    project_path: str | Path, image_asset_id: str, data_group_id: str
) -> None:
    _identifier(image_asset_id, "image_asset_id")
    _identifier(data_group_id, "data_group_id")
    project_info = open_project(project_path)
    if project_info.schema_version not in (
        _TRAINING_SCOPE_SCHEMA_VERSION,
        _DATASET_SNAPSHOT_SCHEMA_VERSION,
        _TRAINING_RUN_SCHEMA_VERSION,
    ):
        raise TrainingScopeError("Data Groups must be saved before assigning images")
    connection = sqlite3.connect(project_info.path / "project.sqlite")
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "INSERT INTO image_data_groups VALUES (?, ?) ON CONFLICT(image_asset_id) "
            "DO UPDATE SET data_group_id = excluded.data_group_id",
            (image_asset_id, data_group_id),
        )
        connection.commit()
    except sqlite3.IntegrityError as error:
        connection.rollback()
        raise TrainingScopeError("Unknown image or Data Group") from error
    finally:
        connection.close()


def save_training_scope(project_path: str | Path, scope: TrainingScope) -> None:
    if not isinstance(scope, TrainingScope):
        raise ValueError("scope must be a TrainingScope")
    project_info = open_project(project_path)
    if project_info.schema_version not in (
        _TRAINING_SCOPE_SCHEMA_VERSION,
        _DATASET_SNAPSHOT_SCHEMA_VERSION,
        _TRAINING_RUN_SCHEMA_VERSION,
    ):
        raise TrainingScopeError("Data Groups must be saved before Training Scope")
    connection = sqlite3.connect(project_info.path / "project.sqlite")
    try:
        groups = _ordered_selection(
            connection, "data_groups", "data_group_id", scope.data_group_ids
        )
        classes = _ordered_selection(
            connection, "defect_classes", "code", scope.class_codes, enabled=True
        )
        connection.execute(
            "INSERT INTO training_scope VALUES (1, ?, ?) ON CONFLICT(singleton) DO UPDATE SET "
            "data_group_ids_json = excluded.data_group_ids_json, "
            "class_codes_json = excluded.class_codes_json",
            (json.dumps(groups), json.dumps(classes)),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def load_training_scope(project_path: str | Path) -> TrainingScope | None:
    project_info = open_project(project_path)
    if project_info.schema_version < _TRAINING_SCOPE_SCHEMA_VERSION:
        return None
    connection = _read_only(project_info.path / "project.sqlite")
    try:
        row = connection.execute(
            "SELECT data_group_ids_json, class_codes_json FROM training_scope WHERE singleton = 1"
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    try:
        return TrainingScope(tuple(json.loads(row[0])), tuple(json.loads(row[1])))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise TrainingScopeError("Invalid Training Scope metadata") from error


def eligible_image_ids(project_path: str | Path) -> tuple[str, ...]:
    scope = load_training_scope(project_path)
    if scope is None or not scope.data_group_ids:
        return ()
    project_info = open_project(project_path)
    connection = _read_only(project_info.path / "project.sqlite")
    try:
        rows = connection.execute(
            "SELECT image_data_groups.image_asset_id FROM image_data_groups "
            "JOIN image_reviews USING (image_asset_id) WHERE image_reviews.reviewed = 1 "
            f"AND image_data_groups.data_group_id IN ({','.join('?' for _ in scope.data_group_ids)}) "
            "ORDER BY image_data_groups.image_asset_id",
            scope.data_group_ids,
        ).fetchall()
    finally:
        connection.close()
    healthy = {
        reopened.asset.image_asset_id
        for reopened in load_image_assets(project_info.path)
        if reopened.source_health is SourceHealth.AVAILABLE
    }
    return tuple(row[0] for row in rows if row[0] in healthy)


def _ensure_schema(connection: sqlite3.Connection, project_id: str, database_path: Path) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == _REVIEW_SCHEMA_VERSION:
        connection.execute(_DATA_GROUPS_TABLE_SQL)
        connection.execute(_IMAGE_DATA_GROUPS_TABLE_SQL)
        connection.execute(_TRAINING_SCOPE_TABLE_SQL)
        updated = connection.execute(
            "UPDATE project_metadata SET schema_version = ? WHERE project_id = ? AND schema_version = ?",
            (_TRAINING_SCOPE_SCHEMA_VERSION, project_id, _REVIEW_SCHEMA_VERSION),
        ).rowcount
        if updated != 1:
            raise TrainingScopeError(f"Invalid project metadata: {database_path}")
        connection.execute(f"PRAGMA user_version = {_TRAINING_SCOPE_SCHEMA_VERSION}")
    elif version not in (
        _TRAINING_SCOPE_SCHEMA_VERSION,
        _DATASET_SNAPSHOT_SCHEMA_VERSION,
        _DATASET_SPLIT_SCHEMA_VERSION,
        _TRAINING_RUN_SCHEMA_VERSION,
    ):
        raise TrainingScopeError("Training Scope requires project schema 9 or newer")


def _ordered_selection(
    connection: sqlite3.Connection,
    table: str,
    identifier: str,
    selected: tuple[str, ...],
    *,
    enabled: bool = False,
) -> tuple[str, ...]:
    if not isinstance(selected, tuple) or not selected:
        raise ValueError("Training Scope selections must be non-empty tuples")
    if len(set(selected)) != len(selected) or any(not isinstance(x, str) or not x for x in selected):
        raise ValueError("Training Scope selections must contain unique non-empty strings")
    where = " WHERE enabled = 1" if enabled else ""
    rows = connection.execute(
        f"SELECT {identifier} FROM {table}{where} ORDER BY display_order, {identifier}"
    ).fetchall()
    ordered = tuple(row[0] for row in rows if row[0] in selected)
    if set(ordered) != set(selected):
        raise ValueError("Training Scope references an unknown or disabled selection")
    return ordered


def _validate_groups(groups: tuple[DataGroup, ...]) -> None:
    if not groups:
        raise ValueError("groups must not be empty")
    identifiers = set()
    for group in groups:
        if not isinstance(group, DataGroup):
            raise ValueError("groups must contain DataGroup values")
        _identifier(group.data_group_id, "data_group_id")
        _identifier(group.name, "name")
        if group.data_group_id in identifiers:
            raise ValueError("duplicate Data Group identifier")
        identifiers.add(group.data_group_id)
        if isinstance(group.order, bool) or not isinstance(group.order, int) or group.order < 0:
            raise ValueError("order must be a non-negative integer")


def _identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _read_only(database_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
