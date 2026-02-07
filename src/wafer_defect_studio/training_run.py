"""GUI-side persistence and publication for immutable Training Runs.

Workers only receive value messages and write files to a run-specific staging
directory.  This module is the project-service boundary: it is the only code
that opens a writable project database and it publishes artifacts only after a
manifest and every declared checksum validate.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .project import (
    _DATASET_SPLIT_SCHEMA_VERSION,
    _TRAINING_RUN_SCHEMA_VERSION,
    _TRAINING_RUNS_TABLE_SQL,
    ProjectError,
    open_project,
)


_INITIAL_STATUS = "created"
_STATUSES = {"created", "running", "completed", "cancelled", "failed", "interrupted"}
_TERMINAL_STATUSES = {"completed", "cancelled", "failed", "interrupted"}
_IMMUTABLE_COLUMNS = (
    "run_id",
    "created_at",
    "snapshot_id",
    "split_id",
    "parent_run_id",
    "config_json",
    "environment_json",
)


class TrainingRunError(ProjectError):
    """Raised when a Training Run cannot be persisted or published."""


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Reproducible model and optimization values captured by a Training Run."""

    snapshot_id: str
    split_id: str
    architecture: str = "resnet18"
    class_count: int = 1
    augmentation: Mapping[str, Any] = field(default_factory=dict)
    batch_size: int = 1
    device: str = "auto"
    epochs: int = 1
    learning_rate: float = 0.001
    seed: int = 0
    weights_policy: str = "none"

    def __post_init__(self) -> None:
        for name in ("snapshot_id", "split_id", "architecture", "device", "weights_policy"):
            _require_text(getattr(self, name), name)
        if self.architecture.lower() != "resnet18":
            raise TrainingRunError("only the resnet18 architecture is supported")
        if self.weights_policy not in {"none", "imagenet"}:
            raise TrainingRunError("weights_policy must be 'none' or 'imagenet'")
        for name in ("class_count", "batch_size", "epochs"):
            _require_positive_int(getattr(self, name), name)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TrainingRunError("seed must be an integer")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or self.learning_rate <= 0
        ):
            raise TrainingRunError("learning_rate must be a positive number")
        _json_mapping(self.augmentation, "augmentation")
        object.__setattr__(self, "augmentation", dict(self.augmentation))

    @property
    def model_config(self) -> dict[str, Any]:
        """Return the explicit model portion of the persisted config."""

        return {
            "architecture": self.architecture,
            "class_count": self.class_count,
            "weights_policy": self.weights_policy,
        }

    @property
    def optimization_config(self) -> dict[str, Any]:
        """Return the explicit optimization portion of the persisted config."""

        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "device": self.device,
        }


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """Persisted run identity, immutable inputs, and service-owned terminal state."""

    run_id: str
    created_at: str
    config: RunConfig
    parent_run_id: str | None
    environment: dict[str, Any]
    metrics: dict[str, Any]
    log: str
    status: str
    terminal_message: str
    staging_path: Path | None
    artifact_path: Path | None

    @property
    def snapshot_id(self) -> str:
        return self.config.snapshot_id

    @property
    def split_id(self) -> str:
        return self.config.split_id

    @property
    def terminal_state(self) -> str:
        return self.status


def create_training_run(
    project_path: str | Path,
    config: RunConfig,
    *,
    parent_run_id: str | None = None,
    environment: Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> TrainingRun:
    """Create one run row and its service-owned staging directory."""

    if not isinstance(config, RunConfig):
        raise ValueError("config must be a RunConfig")
    _validate_optional_id(parent_run_id, "parent_run_id")
    chosen_id = run_id or str(uuid.uuid4())
    _validate_run_id(chosen_id)
    environment_value = _json_mapping(
        {} if environment is None else environment,
        "environment",
    )
    info = open_project(project_path)
    if info.schema_version < _DATASET_SPLIT_SCHEMA_VERSION:
        raise TrainingRunError("Training Runs require project schema 12 or newer")
    database_path = info.path / "project.sqlite"
    staging_path = info.path / "runs" / ".staging" / chosen_id
    if staging_path.exists() or (info.path / "runs" / chosen_id).exists():
        raise TrainingRunError(f"Training Run already exists: {chosen_id}")
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging_path.mkdir()
        created_at = datetime.now(timezone.utc).isoformat()
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            _ensure_schema(connection, info.project_id, database_path)
            connection.execute(
                "INSERT INTO training_runs ("
                "run_id, created_at, snapshot_id, split_id, parent_run_id, config_json, "
                "environment_json, metrics_json, log_text, terminal_status, terminal_message, "
                "staging_path, artifact_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    chosen_id,
                    created_at,
                    config.snapshot_id,
                    config.split_id,
                    parent_run_id,
                    _encode(config),
                    json.dumps(environment_value, sort_keys=True, separators=(",", ":")),
                    "{}",
                    "",
                    _INITIAL_STATUS,
                    "",
                    str(staging_path),
                    None,
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
    except Exception:
        if staging_path.is_dir():
            _remove_empty_directory(staging_path)
        raise
    return TrainingRun(
        chosen_id,
        created_at,
        config,
        parent_run_id,
        environment_value,
        {},
        "",
        _INITIAL_STATUS,
        "",
        staging_path,
        None,
    )


def load_training_run(project_path: str | Path, run_id: str) -> TrainingRun:
    """Read one run through SQLite's read-only URI."""

    _validate_run_id(run_id)
    info = open_project(project_path)
    if info.schema_version < _TRAINING_RUN_SCHEMA_VERSION:
        raise TrainingRunError(f"Unknown Training Run: {run_id}")
    database_path = info.path / "project.sqlite"
    connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT run_id, created_at, snapshot_id, split_id, parent_run_id, config_json, "
            "environment_json, metrics_json, log_text, terminal_status, terminal_message, "
            "staging_path, artifact_path FROM training_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise TrainingRunError(f"Unknown Training Run: {run_id}")
    try:
        config_value = json.loads(row[5])
        config = RunConfig(**config_value)
        environment = _decode_mapping(row[6], "environment")
        metrics = _decode_mapping(row[7], "metrics")
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise TrainingRunError(f"Invalid Training Run metadata: {database_path}") from error
    return TrainingRun(
        row[0],
        row[1],
        config,
        row[4],
        environment,
        metrics,
        row[8],
        row[9],
        row[10],
        Path(row[11]) if row[11] else None,
        Path(row[12]) if row[12] else None,
    )


def update_training_run_terminal(
    project_path: str | Path,
    run_id: str,
    status: str,
    *,
    metrics: Mapping[str, Any] | None = None,
    log: str | None = None,
    message: str = "",
    terminal_message: str | None = None,
    environment: Mapping[str, Any] | None = None,
    staging_path: str | Path | None = None,
) -> TrainingRun:
    """Record a service-owned state transition and publish completed artifacts."""

    _validate_run_id(run_id)
    if status not in _STATUSES:
        raise TrainingRunError(f"unsupported Training Run status: {status!r}")
    if not isinstance(message, str):
        raise ValueError("message must be a string")
    if terminal_message is not None:
        if not isinstance(terminal_message, str):
            raise ValueError("terminal_message must be a string")
        message = terminal_message
    info = open_project(project_path)
    if info.schema_version < _TRAINING_RUN_SCHEMA_VERSION:
        raise TrainingRunError("Training Run schema is not available")
    current = load_training_run(info.path, run_id)
    if current.status in _TERMINAL_STATUSES and status != current.status:
        raise TrainingRunError("terminal Training Runs are immutable")
    metrics_value = current.metrics if metrics is None else _json_mapping(metrics, "metrics")
    environment_value = (
        current.environment
        if environment is None
        else _json_mapping(environment, "environment")
    )
    if environment is not None and environment_value != current.environment:
        raise TrainingRunError("Training Run environment is immutable")
    log_value = current.log if log is None else log
    if not isinstance(log_value, str):
        raise ValueError("log must be a string")

    published_path: Path | None = current.artifact_path
    stage_value = Path(staging_path).expanduser().resolve() if staging_path else current.staging_path
    if status == "completed":
        if stage_value is None:
            raise TrainingRunError("completed Training Runs require staged artifacts")
        published_path = publish_staged_artifacts(info.path, run_id, stage_value)

    database_path = info.path / "project.sqlite"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "UPDATE training_runs SET metrics_json = ?, log_text = ?, "
            "terminal_status = ?, terminal_message = ?, staging_path = ?, artifact_path = ? "
            "WHERE run_id = ?",
            (
                json.dumps(metrics_value, sort_keys=True, separators=(",", ":")),
                log_value,
                status,
                message,
                str(stage_value) if stage_value else None,
                str(published_path) if published_path else None,
                run_id,
            ),
        )
        if connection.execute("SELECT changes()").fetchone()[0] != 1:
            raise TrainingRunError(f"Unknown Training Run: {run_id}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return load_training_run(info.path, run_id)


def validate_staged_artifacts(staging_path: str | Path) -> tuple[Path, ...]:
    """Validate manifest-declared files and SHA-256 checksums."""

    staging = Path(staging_path).expanduser().resolve()
    if not staging.is_dir():
        raise TrainingRunError(f"Missing artifact staging directory: {staging}")
    manifest_path = staging / "manifest.json"
    if not manifest_path.is_file():
        raise TrainingRunError("staged artifacts require manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise TrainingRunError("invalid staged artifact manifest") from error
    if not isinstance(manifest, dict):
        raise TrainingRunError("artifact manifest must be a JSON object")
    entries = _manifest_entries(manifest)
    if not entries:
        raise TrainingRunError("artifact manifest must declare at least one file")
    names = {name for name, _ in entries}
    required = manifest.get("required_files", manifest.get("required", tuple(names)))
    if not isinstance(required, (list, tuple)) or any(not isinstance(item, str) for item in required):
        raise TrainingRunError("manifest required_files must be a list of paths")
    if not set(required).issubset(names):
        raise TrainingRunError("manifest required file is not checksum-declared")
    validated: list[Path] = []
    for relative, expected in entries:
        candidate = _safe_artifact_path(staging, relative)
        if not candidate.is_file():
            raise TrainingRunError(f"missing staged artifact: {relative}")
        actual = _sha256(candidate)
        if actual != expected.lower():
            raise TrainingRunError(f"staged artifact checksum mismatch: {relative}")
        validated.append(candidate)
    return tuple(validated)


def publish_staged_artifacts(
    project_path: str | Path,
    run_id: str,
    staging_path: str | Path,
) -> Path:
    """Atomically move one validated staging directory under ``runs/<run_id>``."""

    _validate_run_id(run_id)
    info = open_project(project_path)
    staging = Path(staging_path).expanduser().resolve()
    validate_staged_artifacts(staging)
    destination = (info.path / "runs" / run_id).resolve()
    if destination.exists():
        raise TrainingRunError(f"published artifacts already exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(staging, destination)
    except OSError as error:
        raise TrainingRunError(f"cannot atomically publish staged artifacts: {destination}") from error
    return destination


# Short aliases keep the project-service seam discoverable without exposing a
# database handle to worker code.
create_run = create_training_run
load_run = load_training_run
update_run_terminal = update_training_run_terminal


def _ensure_schema(connection: sqlite3.Connection, project_id: str, database_path: Path) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == _DATASET_SPLIT_SCHEMA_VERSION:
        connection.execute(_TRAINING_RUNS_TABLE_SQL)
        immutable = ", ".join(_IMMUTABLE_COLUMNS)
        connection.execute(
            "CREATE TRIGGER IF NOT EXISTS training_runs_no_immutable_update "
            f"BEFORE UPDATE OF {immutable} ON training_runs "
            "BEGIN SELECT RAISE(ABORT, 'Training Run configuration is immutable'); END"
        )
        connection.execute(
            "CREATE TRIGGER IF NOT EXISTS training_runs_no_delete BEFORE DELETE ON training_runs "
            "BEGIN SELECT RAISE(ABORT, 'Training Runs are immutable'); END"
        )
        connection.execute(
            "CREATE TRIGGER IF NOT EXISTS training_runs_terminal_once "
            "BEFORE UPDATE OF terminal_status ON training_runs "
            "WHEN OLD.terminal_status IN ('completed', 'cancelled', 'failed', 'interrupted') "
            "AND NEW.terminal_status <> OLD.terminal_status "
            "BEGIN SELECT RAISE(ABORT, 'terminal Training Runs are immutable'); END"
        )
        updated = connection.execute(
            "UPDATE project_metadata SET schema_version = ? "
            "WHERE project_id = ? AND schema_version = ?",
            (_TRAINING_RUN_SCHEMA_VERSION, project_id, _DATASET_SPLIT_SCHEMA_VERSION),
        ).rowcount
        if updated != 1:
            raise TrainingRunError(f"Invalid project metadata: {database_path}")
        connection.execute(f"PRAGMA user_version = {_TRAINING_RUN_SCHEMA_VERSION}")
    elif version != _TRAINING_RUN_SCHEMA_VERSION:
        raise TrainingRunError("Training Runs require project schema 12 or 13")


def _manifest_entries(manifest: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    raw = manifest.get("files", manifest.get("artifacts", manifest.get("checksums")))
    entries: list[tuple[str, str]] = []
    if isinstance(raw, Mapping):
        raw = [{"path": key, "sha256": value} for key, value in raw.items()]
    if not isinstance(raw, list):
        raise TrainingRunError("manifest files must be a list or path/checksum object")
    for item in raw:
        if isinstance(item, Mapping):
            name = item.get("path", item.get("relative_path", item.get("name")))
            checksum = item.get("sha256", item.get("checksum"))
        else:
            raise TrainingRunError("manifest file entries must be objects")
        if not isinstance(name, str) or not name or not isinstance(checksum, str):
            raise TrainingRunError("manifest entries require path and sha256")
        if len(checksum) != 64:
            raise TrainingRunError(f"invalid SHA-256 checksum: {name}")
        try:
            int(checksum, 16)
        except ValueError as error:
            raise TrainingRunError(f"invalid SHA-256 checksum: {name}") from error
        entries.append((name, checksum))
    if len({name for name, _ in entries}) != len(entries):
        raise TrainingRunError("manifest contains duplicate artifact paths")
    return tuple(entries)


def _safe_artifact_path(staging: Path, relative: str) -> Path:
    candidate = (staging / relative).resolve()
    try:
        candidate.relative_to(staging)
    except ValueError as error:
        raise TrainingRunError(f"artifact path escapes staging directory: {relative}") from error
    return candidate


def _encode(config: RunConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def _json_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result = dict(value)
    try:
        json.dumps(result, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain JSON values") from error
    return result


def _decode_mapping(serialized: str, name: str) -> dict[str, Any]:
    value = json.loads(serialized)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _validate_run_id(value: object) -> None:
    _require_text(value, "run_id")
    if any(character in str(value) for character in ("/", "\\", "..")):
        raise ValueError("run_id must be a simple path-safe identifier")


def _validate_optional_id(value: object, name: str) -> None:
    if value is not None:
        _validate_run_id(value)
        if name != "parent_run_id":
            _require_text(value, name)


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_empty_directory(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


__all__ = [
    "RunConfig",
    "TrainingRun",
    "TrainingRunError",
    "create_run",
    "create_training_run",
    "load_run",
    "load_training_run",
    "publish_staged_artifacts",
    "update_run_terminal",
    "update_training_run_terminal",
    "validate_staged_artifacts",
]
