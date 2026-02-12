"""Create and reopen the on-disk project shell."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path


_SCHEMA_VERSION = 6
_DEFECT_CLASS_SCHEMA_VERSION = 7
_GRID_ANNOTATION_SCHEMA_VERSION = 8
_REVIEW_SCHEMA_VERSION = 9
_TRAINING_SCOPE_SCHEMA_VERSION = 10
_DATASET_SNAPSHOT_SCHEMA_VERSION = 11
_DATASET_SPLIT_SCHEMA_VERSION = 12
_TRAINING_RUN_SCHEMA_VERSION = 13
_EVALUATION_SCHEMA_VERSION = 14
_DETECTION_SCHEMA_VERSION = 15
_PROPOSAL_SCHEMA_VERSION = 16
_IMAGE_ASSET_SCHEMA_VERSION = 3
_GRID_PROFILE_SCHEMA_VERSION = 4
_IMAGE_GRID_PLACEMENT_SCHEMA_VERSION = 5
_SUPPORTED_SCHEMA_VERSIONS = (
    1,
    2,
    3,
    4,
    5,
    _SCHEMA_VERSION,
    _DEFECT_CLASS_SCHEMA_VERSION,
    _GRID_ANNOTATION_SCHEMA_VERSION,
    _REVIEW_SCHEMA_VERSION,
    _TRAINING_SCOPE_SCHEMA_VERSION,
    _DATASET_SNAPSHOT_SCHEMA_VERSION,
    _DATASET_SPLIT_SCHEMA_VERSION,
    _TRAINING_RUN_SCHEMA_VERSION,
    _EVALUATION_SCHEMA_VERSION,
    _DETECTION_SCHEMA_VERSION,
    _PROPOSAL_SCHEMA_VERSION,
)
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
_DEFECT_CLASSES_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS defect_classes ("
    "code TEXT NOT NULL PRIMARY KEY, "
    "name TEXT NOT NULL, "
    "color TEXT NOT NULL, "
    "icon TEXT NOT NULL DEFAULT '', "
    "description TEXT NOT NULL DEFAULT '', "
    "display_order INTEGER NOT NULL CHECK (display_order >= 0), "
    "enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1))"
    ")"
)
_GRID_ANNOTATIONS_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS grid_annotations ("
    "image_asset_id TEXT NOT NULL, "
    "row INTEGER NOT NULL, "
    "column INTEGER NOT NULL, "
    "class_codes_json TEXT NOT NULL, "
    "PRIMARY KEY (image_asset_id, row, column), "
    "FOREIGN KEY (image_asset_id) REFERENCES image_assets(image_asset_id)"
    ")"
)
_IMAGE_REVIEWS_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS image_reviews ("
    "image_asset_id TEXT NOT NULL PRIMARY KEY, "
    "reviewed INTEGER NOT NULL CHECK (reviewed IN (0, 1)), "
    "FOREIGN KEY (image_asset_id) REFERENCES image_assets(image_asset_id)"
    ")"
)
_DATA_GROUPS_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS data_groups ("
    "data_group_id TEXT NOT NULL PRIMARY KEY, name TEXT NOT NULL, "
    "display_order INTEGER NOT NULL CHECK (display_order >= 0))"
)
_IMAGE_DATA_GROUPS_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS image_data_groups ("
    "image_asset_id TEXT NOT NULL PRIMARY KEY, data_group_id TEXT NOT NULL, "
    "FOREIGN KEY (image_asset_id) REFERENCES image_assets(image_asset_id), "
    "FOREIGN KEY (data_group_id) REFERENCES data_groups(data_group_id))"
)
_TRAINING_SCOPE_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS training_scope ("
    "singleton INTEGER NOT NULL PRIMARY KEY CHECK (singleton = 1), "
    "data_group_ids_json TEXT NOT NULL, class_codes_json TEXT NOT NULL)"
)
_DATASET_SNAPSHOTS_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS dataset_snapshots ("
    "snapshot_id TEXT NOT NULL PRIMARY KEY, created_at TEXT NOT NULL, payload_json TEXT NOT NULL)"
)
_DATASET_SPLITS_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS dataset_splits ("
    "split_id TEXT NOT NULL PRIMARY KEY, snapshot_id TEXT NOT NULL, seed INTEGER NOT NULL, "
    "payload_json TEXT NOT NULL, FOREIGN KEY (snapshot_id) REFERENCES dataset_snapshots(snapshot_id))"
)
_TRAINING_RUNS_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS training_runs ("
    "run_id TEXT NOT NULL PRIMARY KEY, "
    "created_at TEXT NOT NULL, "
    "snapshot_id TEXT NOT NULL, "
    "split_id TEXT NOT NULL, "
    "parent_run_id TEXT, "
    "config_json TEXT NOT NULL, "
    "environment_json TEXT NOT NULL, "
    "metrics_json TEXT NOT NULL, "
    "log_text TEXT NOT NULL DEFAULT '', "
    "terminal_status TEXT NOT NULL CHECK(terminal_status IN "
    "('created', 'running', 'completed', 'cancelled', 'failed', 'interrupted')), "
    "terminal_message TEXT NOT NULL DEFAULT '', "
    "staging_path TEXT, "
    "artifact_path TEXT"
    ")"
)
_EVALUATION_RUNS_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS evaluation_runs ("
    "evaluation_id TEXT NOT NULL PRIMARY KEY, "
    "training_run_id TEXT NOT NULL, "
    "snapshot_id TEXT NOT NULL, "
    "split_id TEXT NOT NULL, "
    "created_at TEXT NOT NULL, "
    "environment_json TEXT NOT NULL, "
    "criteria_json TEXT NOT NULL, "
    "metrics_json TEXT NOT NULL, "
    "thresholds_json TEXT NOT NULL, "
    "target_satisfied INTEGER NOT NULL CHECK (target_satisfied IN (0, 1)), "
    "notes TEXT NOT NULL"
    ")"
)
_EVALUATION_DECISIONS_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS evaluation_decisions ("
    "decision_id TEXT NOT NULL PRIMARY KEY, "
    "evaluation_id TEXT NOT NULL, "
    "status TEXT NOT NULL CHECK (status IN ('candidate', 'validated', 'approved')), "
    "actor TEXT NOT NULL, "
    "decided_at TEXT NOT NULL, "
    "criteria_json TEXT NOT NULL, "
    "target_satisfied INTEGER NOT NULL CHECK (target_satisfied IN (0, 1)), "
    "notes TEXT NOT NULL, "
    "FOREIGN KEY (evaluation_id) REFERENCES evaluation_runs(evaluation_id)"
    ")"
)
_DETECTION_PROFILES_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS detection_profiles ("
    "profile_id TEXT NOT NULL PRIMARY KEY, "
    "created_at TEXT NOT NULL, "
    "evaluation_id TEXT NOT NULL, "
    "training_run_id TEXT NOT NULL, "
    "model_fingerprint TEXT NOT NULL, "
    "settings_json TEXT NOT NULL, "
    "FOREIGN KEY (evaluation_id) REFERENCES evaluation_runs(evaluation_id))"
)
_DETECTION_RUNS_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS detection_runs ("
    "detection_run_id TEXT NOT NULL PRIMARY KEY, "
    "created_at TEXT NOT NULL, "
    "evaluation_id TEXT NOT NULL, "
    "training_run_id TEXT NOT NULL, "
    "profile_id TEXT NOT NULL, "
    "model_fingerprint TEXT NOT NULL, "
    "source_fingerprints_json TEXT NOT NULL, "
    "provenance_json TEXT NOT NULL, "
    "status TEXT NOT NULL CHECK (status IN ('created', 'running', 'completed', 'cancelled', 'failed', 'interrupted')), "
    "staging_path TEXT, "
    "artifact_path TEXT, "
    "FOREIGN KEY (evaluation_id) REFERENCES evaluation_runs(evaluation_id), "
    "FOREIGN KEY (profile_id) REFERENCES detection_profiles(profile_id))"
)
_DEFECT_PROPOSALS_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS defect_proposals ("
    "proposal_id TEXT NOT NULL PRIMARY KEY, "
    "detection_run_id TEXT NOT NULL, "
    "profile_id TEXT NOT NULL, "
    "class_name TEXT NOT NULL, "
    "source_rect_json TEXT NOT NULL, "
    "area INTEGER NOT NULL CHECK (area > 0), "
    "peak_confidence REAL NOT NULL CHECK (peak_confidence >= 0 AND peak_confidence <= 1), "
    "mean_confidence REAL NOT NULL CHECK (mean_confidence >= 0 AND mean_confidence <= 1), "
    "provenance_json TEXT NOT NULL, "
    "FOREIGN KEY (detection_run_id) REFERENCES detection_runs(detection_run_id), "
    "FOREIGN KEY (profile_id) REFERENCES detection_profiles(profile_id)"
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
_DEFECT_CLASS_COLUMNS = (
    "code",
    "name",
    "color",
    "icon",
    "description",
    "display_order",
    "enabled",
)
_GRID_ANNOTATION_COLUMNS = (
    "image_asset_id",
    "row",
    "column",
    "class_codes_json",
)
_IMAGE_REVIEW_COLUMNS = ("image_asset_id", "reviewed")
_DATA_GROUP_COLUMNS = ("data_group_id", "name", "display_order")
_IMAGE_DATA_GROUP_COLUMNS = ("image_asset_id", "data_group_id")
_TRAINING_SCOPE_COLUMNS = ("singleton", "data_group_ids_json", "class_codes_json")
_DATASET_SNAPSHOT_COLUMNS = ("snapshot_id", "created_at", "payload_json")
_DATASET_SPLIT_COLUMNS = ("split_id", "snapshot_id", "seed", "payload_json")
_TRAINING_RUN_COLUMNS = (
    "run_id",
    "created_at",
    "snapshot_id",
    "split_id",
    "parent_run_id",
    "config_json",
    "environment_json",
    "metrics_json",
    "log_text",
    "terminal_status",
    "terminal_message",
    "staging_path",
    "artifact_path",
)
_EVALUATION_RUN_COLUMNS = (
    "evaluation_id",
    "training_run_id",
    "snapshot_id",
    "split_id",
    "created_at",
    "environment_json",
    "criteria_json",
    "metrics_json",
    "thresholds_json",
    "target_satisfied",
    "notes",
)
_EVALUATION_DECISION_COLUMNS = (
    "decision_id",
    "evaluation_id",
    "status",
    "actor",
    "decided_at",
    "criteria_json",
    "target_satisfied",
    "notes",
)
_DETECTION_PROFILE_COLUMNS = (
    "profile_id",
    "created_at",
    "evaluation_id",
    "training_run_id",
    "model_fingerprint",
    "settings_json",
)
_DETECTION_RUN_COLUMNS = (
    "detection_run_id",
    "created_at",
    "evaluation_id",
    "training_run_id",
    "profile_id",
    "model_fingerprint",
    "source_fingerprints_json",
    "provenance_json",
    "status",
    "staging_path",
    "artifact_path",
)
_DEFECT_PROPOSAL_COLUMNS = (
    "proposal_id",
    "detection_run_id",
    "profile_id",
    "class_name",
    "source_rect_json",
    "area",
    "peak_confidence",
    "mean_confidence",
    "provenance_json",
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
        defect_class_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(defect_classes)")
        )
        grid_annotation_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(grid_annotations)")
        )
        image_review_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(image_reviews)")
        )
        data_group_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(data_groups)")
        )
        image_data_group_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(image_data_groups)")
        )
        training_scope_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(training_scope)")
        )
        dataset_snapshot_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(dataset_snapshots)")
        )
        dataset_split_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(dataset_splits)")
        )
        training_run_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(training_runs)")
        )
        evaluation_run_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(evaluation_runs)")
        )
        evaluation_decision_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(evaluation_decisions)")
        )
        detection_profile_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(detection_profiles)")
        )
        detection_run_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(detection_runs)")
        )
        defect_proposal_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(defect_proposals)")
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
    if pragma_version >= _SCHEMA_VERSION and effective_wafer_area_columns != _EFFECTIVE_WAFER_AREA_COLUMNS:
        raise ProjectError(f"Invalid effective wafer area table: {database_path}")
    if pragma_version >= _DEFECT_CLASS_SCHEMA_VERSION and defect_class_columns != _DEFECT_CLASS_COLUMNS:
        raise ProjectError(f"Invalid defect class table: {database_path}")
    if (
        pragma_version >= _GRID_ANNOTATION_SCHEMA_VERSION
        and grid_annotation_columns != _GRID_ANNOTATION_COLUMNS
    ):
        raise ProjectError(f"Invalid grid annotation table: {database_path}")
    if pragma_version >= _REVIEW_SCHEMA_VERSION and image_review_columns != _IMAGE_REVIEW_COLUMNS:
        raise ProjectError(f"Invalid image review table: {database_path}")
    if pragma_version >= _TRAINING_SCOPE_SCHEMA_VERSION:
        if data_group_columns != _DATA_GROUP_COLUMNS:
            raise ProjectError(f"Invalid data group table: {database_path}")
        if image_data_group_columns != _IMAGE_DATA_GROUP_COLUMNS:
            raise ProjectError(f"Invalid image data group table: {database_path}")
        if training_scope_columns != _TRAINING_SCOPE_COLUMNS:
            raise ProjectError(f"Invalid training scope table: {database_path}")
    if (
        pragma_version >= _DATASET_SNAPSHOT_SCHEMA_VERSION
        and dataset_snapshot_columns != _DATASET_SNAPSHOT_COLUMNS
    ):
        raise ProjectError(f"Invalid Dataset Snapshot table: {database_path}")
    if pragma_version >= _DATASET_SPLIT_SCHEMA_VERSION and dataset_split_columns != _DATASET_SPLIT_COLUMNS:
        raise ProjectError(f"Invalid Dataset Split table: {database_path}")
    if pragma_version >= _TRAINING_RUN_SCHEMA_VERSION and training_run_columns != _TRAINING_RUN_COLUMNS:
        raise ProjectError(f"Invalid Training Run table: {database_path}")
    if pragma_version >= _EVALUATION_SCHEMA_VERSION:
        if evaluation_run_columns != _EVALUATION_RUN_COLUMNS:
            raise ProjectError(f"Invalid Evaluation table: {database_path}")
        if evaluation_decision_columns != _EVALUATION_DECISION_COLUMNS:
            raise ProjectError(f"Invalid Evaluation decision table: {database_path}")
    if pragma_version >= _DETECTION_SCHEMA_VERSION:
        if detection_profile_columns != _DETECTION_PROFILE_COLUMNS:
            raise ProjectError(f"Invalid Detection Profile table: {database_path}")
        if detection_run_columns != _DETECTION_RUN_COLUMNS:
            raise ProjectError(f"Invalid Detection Run table: {database_path}")
    if pragma_version >= _PROPOSAL_SCHEMA_VERSION:
        if defect_proposal_columns != _DEFECT_PROPOSAL_COLUMNS:
            raise ProjectError(f"Invalid Defect Proposal table: {database_path}")

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
