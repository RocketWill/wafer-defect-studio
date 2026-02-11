"""Persist deterministic image-level Dataset Splits."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from .dataset_snapshot import load_dataset_snapshot
from .project import (
    _DATASET_SPLITS_TABLE_SQL,
    _DATASET_SPLIT_SCHEMA_VERSION,
    _DATASET_SNAPSHOT_SCHEMA_VERSION,
    _TRAINING_RUN_SCHEMA_VERSION,
    _EVALUATION_SCHEMA_VERSION,
    _DETECTION_SCHEMA_VERSION,
    ProjectError,
    open_project,
)


class DatasetSplitError(ProjectError):
    pass


@dataclass(frozen=True)
class DatasetSplit:
    split_id: str
    snapshot_id: str
    seed: int
    train_image_ids: tuple[str, ...]
    validation_image_ids: tuple[str, ...]
    test_image_ids: tuple[str, ...]


def create_dataset_split(
    project_path: str | Path, snapshot_id: str, seed: int
) -> DatasetSplit:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    snapshot = load_dataset_snapshot(project_path, snapshot_id)
    ranked = tuple(
        sorted(
            (source.image_asset_id for source in snapshot.sources),
            key=lambda image_id: hashlib.sha256(
                f"{seed}\0{image_id}".encode("utf-8")
            ).digest(),
        )
    )
    validation_count = len(ranked) // 10
    test_count = len(ranked) // 10
    train_count = len(ranked) - validation_count - test_count
    split_id = hashlib.sha256(f"{snapshot_id}\0{seed}".encode("utf-8")).hexdigest()
    split = DatasetSplit(
        split_id,
        snapshot_id,
        seed,
        ranked[:train_count],
        ranked[train_count : train_count + validation_count],
        ranked[train_count + validation_count :],
    )

    info = open_project(project_path)
    connection = sqlite3.connect(info.path / "project.sqlite")
    try:
        connection.execute("BEGIN IMMEDIATE")
        _ensure_schema(connection, info.project_id)
        row = connection.execute(
            "SELECT payload_json FROM dataset_splits WHERE split_id = ?", (split_id,)
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO dataset_splits VALUES (?, ?, ?, ?)",
                (split_id, snapshot_id, seed, _encode(split)),
            )
        connection.commit()
        return _decode(row[0]) if row else split
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def load_dataset_split(project_path: str | Path, split_id: str) -> DatasetSplit:
    info = open_project(project_path)
    if info.schema_version < _DATASET_SPLIT_SCHEMA_VERSION:
        raise DatasetSplitError(f"Unknown Dataset Split: {split_id}")
    connection = sqlite3.connect(
        (info.path / "project.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    )
    try:
        row = connection.execute(
            "SELECT payload_json FROM dataset_splits WHERE split_id = ?", (split_id,)
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise DatasetSplitError(f"Unknown Dataset Split: {split_id}")
    return _decode(row[0])


def _ensure_schema(connection: sqlite3.Connection, project_id: str) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == _DATASET_SNAPSHOT_SCHEMA_VERSION:
        connection.execute(_DATASET_SPLITS_TABLE_SQL)
        connection.execute(
            "CREATE TRIGGER dataset_splits_no_update BEFORE UPDATE ON dataset_splits "
            "BEGIN SELECT RAISE(ABORT, 'Dataset Splits are immutable'); END"
        )
        connection.execute(
            "CREATE TRIGGER dataset_splits_no_delete BEFORE DELETE ON dataset_splits "
            "BEGIN SELECT RAISE(ABORT, 'Dataset Splits are immutable'); END"
        )
        connection.execute(
            "UPDATE project_metadata SET schema_version = ? WHERE project_id = ?",
            (_DATASET_SPLIT_SCHEMA_VERSION, project_id),
        )
        connection.execute(f"PRAGMA user_version = {_DATASET_SPLIT_SCHEMA_VERSION}")
    elif version not in (
        _DATASET_SPLIT_SCHEMA_VERSION,
        _TRAINING_RUN_SCHEMA_VERSION,
        _EVALUATION_SCHEMA_VERSION,
        _DETECTION_SCHEMA_VERSION,
    ):
        raise DatasetSplitError("Dataset Splits require project schema 11 or newer")


def _encode(split: DatasetSplit) -> str:
    return json.dumps(asdict(split), sort_keys=True, separators=(",", ":"))


def _decode(serialized: str) -> DatasetSplit:
    try:
        value = json.loads(serialized)
        return DatasetSplit(
            value["split_id"],
            value["snapshot_id"],
            value["seed"],
            tuple(value["train_image_ids"]),
            tuple(value["validation_image_ids"]),
            tuple(value["test_image_ids"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DatasetSplitError("Invalid Dataset Split metadata") from error
