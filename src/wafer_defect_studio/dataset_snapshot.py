"""Freeze reproducible, append-only Dataset Snapshot inputs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .defect_class import DefectClass, load_defect_classes
from .normalization import NormalizationBounds
from .project import (
    _DATASET_SNAPSHOTS_TABLE_SQL,
    _DATASET_SPLIT_SCHEMA_VERSION,
    _DATASET_SNAPSHOT_SCHEMA_VERSION,
    _TRAINING_SCOPE_SCHEMA_VERSION,
    _TRAINING_RUN_SCHEMA_VERSION,
    ProjectError,
    open_project,
)
from .training_scope import TrainingScope, eligible_image_ids, load_training_scope


class DatasetSnapshotError(ProjectError):
    pass


@dataclass(frozen=True)
class SamplingPolicy:
    normal_to_positive_ratio: float = 1.0


@dataclass(frozen=True)
class SnapshotSource:
    image_asset_id: str
    data_group_id: str
    fingerprint: str


@dataclass(frozen=True)
class GridVersion:
    image_asset_id: str
    grid_profile_id: str
    grid_profile_version: int
    origin_x: int
    origin_y: int


@dataclass(frozen=True)
class AnnotationVersion:
    image_asset_id: str
    content_hash: str


@dataclass(frozen=True)
class DatasetSnapshot:
    snapshot_id: str
    created_at: str
    scope: TrainingScope
    classes: tuple[DefectClass, ...]
    sources: tuple[SnapshotSource, ...]
    grid_versions: tuple[GridVersion, ...]
    annotation_versions: tuple[AnnotationVersion, ...]
    normalization_bounds: tuple[NormalizationBounds, ...]
    sampling_policy: SamplingPolicy


def create_dataset_snapshot(
    project_path: str | Path,
    normalization_bounds: tuple[NormalizationBounds, ...],
    sampling_policy: SamplingPolicy,
) -> DatasetSnapshot:
    """Freeze all current eligible Training Scope inputs in one append-only row."""

    scope = load_training_scope(project_path)
    if scope is None or not scope.data_group_ids or not scope.class_codes:
        raise DatasetSnapshotError("Training Scope is incomplete")
    if not isinstance(normalization_bounds, tuple) or not normalization_bounds:
        raise ValueError("normalization_bounds must be a non-empty tuple")
    if not all(isinstance(item, NormalizationBounds) for item in normalization_bounds):
        raise ValueError("normalization_bounds must contain NormalizationBounds values")
    if (
        not isinstance(sampling_policy, SamplingPolicy)
        or isinstance(sampling_policy.normal_to_positive_ratio, bool)
        or not isinstance(sampling_policy.normal_to_positive_ratio, (int, float))
        or sampling_policy.normal_to_positive_ratio < 0
    ):
        raise ValueError("sampling_policy must have a non-negative Normal ratio")

    info = open_project(project_path)
    database = info.path / "project.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("BEGIN IMMEDIATE")
        placeholders = ",".join("?" for _ in scope.data_group_ids)
        selected = tuple(
            row[0]
            for row in connection.execute(
                "SELECT image_asset_id FROM image_data_groups WHERE data_group_id IN ("
                + placeholders
                + ") ORDER BY image_asset_id",
                scope.data_group_ids,
            )
        )
        eligible = eligible_image_ids(info.path)
        if not selected or selected != eligible:
            raise DatasetSnapshotError(
                "Every image in the Training Scope must be reviewed with an unchanged source"
            )

        class_by_code = {item.code: item for item in load_defect_classes(info.path)}
        try:
            classes = tuple(class_by_code[code] for code in scope.class_codes)
        except KeyError as error:
            raise DatasetSnapshotError("Training Scope class schema is incomplete") from error
        if not all(item.enabled for item in classes):
            raise DatasetSnapshotError("Training Scope contains an archived Defect Class")

        source_rows = connection.execute(
            "SELECT i.image_asset_id, g.data_group_id, i.fingerprint, i.dtype "
            "FROM image_assets i JOIN image_data_groups g USING (image_asset_id) "
            "WHERE i.image_asset_id IN (" + ",".join("?" for _ in eligible) + ") "
            "ORDER BY i.image_asset_id",
            eligible,
        ).fetchall()
        bound_dtypes = {item.dtype for item in normalization_bounds}
        if {row[3] for row in source_rows} != bound_dtypes:
            raise DatasetSnapshotError("Normalization bounds do not cover the Training Scope")
        sources = tuple(SnapshotSource(*row[:3]) for row in source_rows)

        grid_rows = connection.execute(
            "SELECT image_asset_id, grid_profile_id, grid_profile_version, origin_x, origin_y "
            "FROM image_grid_placements WHERE image_asset_id IN ("
            + ",".join("?" for _ in eligible)
            + ") ORDER BY image_asset_id",
            eligible,
        ).fetchall()
        if len(grid_rows) != len(eligible):
            raise DatasetSnapshotError("Training Scope has incomplete grid metadata")
        grids = tuple(GridVersion(*row) for row in grid_rows)
        annotations = tuple(
            AnnotationVersion(image_id, _annotation_hash(connection, image_id))
            for image_id in eligible
        )
        snapshot = DatasetSnapshot(
            str(uuid.uuid4()),
            datetime.now(timezone.utc).isoformat(),
            scope,
            classes,
            sources,
            grids,
            annotations,
            normalization_bounds,
            SamplingPolicy(float(sampling_policy.normal_to_positive_ratio)),
        )
        _ensure_schema(connection, info.project_id, database)
        connection.execute(
            "INSERT INTO dataset_snapshots VALUES (?, ?, ?)",
            (snapshot.snapshot_id, snapshot.created_at, _encode(snapshot)),
        )
        connection.commit()
        return snapshot
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def load_dataset_snapshot(project_path: str | Path, snapshot_id: str) -> DatasetSnapshot:
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("snapshot_id must be a non-empty string")
    info = open_project(project_path)
    if info.schema_version < _DATASET_SNAPSHOT_SCHEMA_VERSION:
        raise DatasetSnapshotError(f"Unknown Dataset Snapshot: {snapshot_id}")
    connection = sqlite3.connect(
        (info.path / "project.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    )
    try:
        row = connection.execute(
            "SELECT payload_json FROM dataset_snapshots WHERE snapshot_id = ?", (snapshot_id,)
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise DatasetSnapshotError(f"Unknown Dataset Snapshot: {snapshot_id}")
    return _decode(row[0])


def _annotation_hash(connection: sqlite3.Connection, image_id: str) -> str:
    rows = connection.execute(
        "SELECT row, column, class_codes_json FROM grid_annotations "
        "WHERE image_asset_id = ? ORDER BY row, column",
        (image_id,),
    ).fetchall()
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ensure_schema(connection: sqlite3.Connection, project_id: str, database: Path) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == _TRAINING_SCOPE_SCHEMA_VERSION:
        connection.execute(_DATASET_SNAPSHOTS_TABLE_SQL)
        connection.execute(
            "CREATE TRIGGER dataset_snapshots_no_update BEFORE UPDATE ON dataset_snapshots "
            "BEGIN SELECT RAISE(ABORT, 'Dataset Snapshots are immutable'); END"
        )
        connection.execute(
            "CREATE TRIGGER dataset_snapshots_no_delete BEFORE DELETE ON dataset_snapshots "
            "BEGIN SELECT RAISE(ABORT, 'Dataset Snapshots are immutable'); END"
        )
        updated = connection.execute(
            "UPDATE project_metadata SET schema_version = ? "
            "WHERE project_id = ? AND schema_version = ?",
            (_DATASET_SNAPSHOT_SCHEMA_VERSION, project_id, _TRAINING_SCOPE_SCHEMA_VERSION),
        ).rowcount
        if updated != 1:
            raise DatasetSnapshotError(f"Invalid project metadata: {database}")
        connection.execute(f"PRAGMA user_version = {_DATASET_SNAPSHOT_SCHEMA_VERSION}")
    elif version not in (
        _DATASET_SNAPSHOT_SCHEMA_VERSION,
        _DATASET_SPLIT_SCHEMA_VERSION,
        _TRAINING_RUN_SCHEMA_VERSION,
    ):
        raise DatasetSnapshotError("Dataset Snapshots require project schema 10 or newer")


def _encode(snapshot: DatasetSnapshot) -> str:
    return json.dumps(asdict(snapshot), sort_keys=True, separators=(",", ":"))


def _decode(serialized: str) -> DatasetSnapshot:
    try:
        value = json.loads(serialized)
        return DatasetSnapshot(
            value["snapshot_id"],
            value["created_at"],
            TrainingScope(
                tuple(value["scope"]["data_group_ids"]),
                tuple(value["scope"]["class_codes"]),
            ),
            tuple(DefectClass(**item) for item in value["classes"]),
            tuple(SnapshotSource(**item) for item in value["sources"]),
            tuple(GridVersion(**item) for item in value["grid_versions"]),
            tuple(AnnotationVersion(**item) for item in value["annotation_versions"]),
            tuple(NormalizationBounds(**item) for item in value["normalization_bounds"]),
            SamplingPolicy(**value["sampling_policy"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DatasetSnapshotError("Invalid Dataset Snapshot metadata") from error
