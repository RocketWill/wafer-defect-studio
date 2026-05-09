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
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .project import (
    _DATASET_SPLIT_SCHEMA_VERSION,
    _TRAINING_RUN_SCHEMA_VERSION,
    _EVALUATION_SCHEMA_VERSION,
    _DETECTION_SCHEMA_VERSION,
    _TRAINING_RUNS_TABLE_SQL,
    ProjectError,
    open_project,
)
from .environment import collect_training_environment


_INITIAL_STATUS = "created"
_STATUSES = {"created", "running", "completed", "cancelled", "failed", "interrupted"}
_TERMINAL_STATUSES = {"completed", "cancelled", "failed", "interrupted"}
REQUIRED_ENVIRONMENT_KEYS = (
    "python",
    "pytorch",
    "torchvision",
    "cuda",
    "cuda_driver",
    "os",
    "gpu",
    "packages",
)
_OOM_ERROR_CODES = {"out_of_memory", "oom"}
_CHECKPOINT_NAME = "model.pt"
_CHECKPOINT_FORMAT_V1 = "wafer_defect_studio.resnet18.v1"
_CHECKPOINT_FORMAT_V2 = "wafer_defect_studio.resnet18.v2"
_CHECKPOINT_FORMAT_V3 = "wafer_defect_studio.resnet18.v3"
_CHECKPOINT_FORMAT_V4 = "wafer_defect_studio.resnet18.v4"
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
    patch_size: int | None = None
    patch_stride: int | None = None
    bag_pooling: str | None = None
    training_policy: str = "legacy"

    def __post_init__(self) -> None:
        for name in ("snapshot_id", "split_id", "architecture", "device", "weights_policy"):
            _require_text(getattr(self, name), name)
        if self.architecture.lower() != "resnet18":
            raise TrainingRunError("only the resnet18 architecture is supported")
        if self.weights_policy not in {"none", "imagenet"}:
            raise TrainingRunError("weights_policy must be 'none' or 'imagenet'")
        if self.training_policy not in {"legacy", "spatial_mil_v4"}:
            raise TrainingRunError(
                "training_policy must be 'legacy' or 'spatial_mil_v4'"
            )
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
        patch_values = (self.patch_size, self.patch_stride)
        if any(value is not None for value in patch_values):
            if self.patch_size is None or self.patch_stride is None:
                raise TrainingRunError(
                    "patch_size and patch_stride must be provided together"
                )
            _require_positive_int(self.patch_size, "patch_size")
            _require_positive_int(self.patch_stride, "patch_stride")
            if self.patch_stride > self.patch_size:
                raise TrainingRunError("patch_stride cannot exceed patch_size")
            if self.training_policy == "legacy" and self.bag_pooling != "max":
                raise TrainingRunError("bag_pooling must be max for Patch Classification")
        if self.training_policy == "spatial_mil_v4" and self.patch_size is None:
            raise TrainingRunError("spatial_mil_v4 requires patch geometry")
        if self.training_policy == "spatial_mil_v4" and self.bag_pooling is not None:
            raise TrainingRunError("spatial_mil_v4 bag_pooling must be null")
        if self.training_policy == "legacy" and self.patch_size is None and self.bag_pooling is not None:
            raise TrainingRunError(
                "patch_size and patch_stride must be provided with bag_pooling"
            )

    @property
    def model_config(self) -> dict[str, Any]:
        """Return the explicit model portion of the persisted config."""

        return {
            "architecture": self.architecture,
            "class_count": self.class_count,
            "weights_policy": self.weights_policy,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "bag_pooling": self.bag_pooling,
            "training_policy": self.training_policy,
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
    environment_value = _merge_environment(
        collect_training_environment(),
        environment,
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
    error_code: str | None = None,
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
    if error_code is not None:
        _require_text(error_code, "error_code")
    info = open_project(project_path)
    if info.schema_version < _TRAINING_RUN_SCHEMA_VERSION:
        raise TrainingRunError("Training Run schema is not available")
    current = load_training_run(info.path, run_id)
    if current.status in _TERMINAL_STATUSES and status != current.status:
        raise TrainingRunError("terminal Training Runs are immutable")
    metrics_value = current.metrics if metrics is None else _json_mapping(metrics, "metrics")
    if error_code is not None:
        metrics_value = dict(metrics_value)
        metrics_value["error_code"] = error_code
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
        _require_complete_environment(current.environment)
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


def resume_training_run(
    project_path: str | Path,
    parent_run_id: str,
    *,
    new_run_id: str | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> TrainingRun:
    """Create a new child run from a terminal run with a valid checkpoint."""

    parent = load_training_run(project_path, parent_run_id)
    if parent.status not in _TERMINAL_STATUSES:
        raise TrainingRunError("resume requires a terminal parent Training Run")
    if not _has_checkpoint(parent):
        raise TrainingRunError("resume requires a validated model.pt checkpoint")
    if environment is not None:
        _require_complete_environment(environment)
    config = _child_config(parent.config, config_overrides)
    inherited_environment = parent.environment if environment is None else environment
    return create_training_run(
        project_path,
        config,
        parent_run_id=parent.run_id,
        environment=inherited_environment,
        run_id=new_run_id,
    )


def clone_after_oom(
    project_path: str | Path,
    parent_run_id: str,
    *,
    batch_size: int,
    new_run_id: str | None = None,
    environment: Mapping[str, Any] | None = None,
) -> TrainingRun:
    """Create an explicit smaller-batch child for a failed OOM run."""

    parent = load_training_run(project_path, parent_run_id)
    if parent.status != "failed":
        raise TrainingRunError("OOM clone requires a failed parent Training Run")
    if not _is_oom_run(parent):
        raise TrainingRunError("OOM clone requires an out_of_memory failure")
    _require_positive_int(batch_size, "batch_size")
    if batch_size >= parent.config.batch_size:
        raise TrainingRunError("OOM clone batch_size must be smaller than the parent")
    if environment is not None:
        _require_complete_environment(environment)
    config = _child_config(parent.config, {"batch_size": batch_size})
    inherited_environment = parent.environment if environment is None else environment
    return create_training_run(
        project_path,
        config,
        parent_run_id=parent.run_id,
        environment=inherited_environment,
        run_id=new_run_id,
    )


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
    validated = validate_staged_artifacts(staging)
    checkpoint = next((path for path in validated if path.name == _CHECKPOINT_NAME), None)
    if checkpoint is not None:
        validate_project_checkpoint(checkpoint)
    destination = (info.path / "runs" / run_id).resolve()
    if destination.exists():
        raise TrainingRunError(f"published artifacts already exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(staging, destination)
    except OSError as error:
        raise TrainingRunError(f"cannot atomically publish staged artifacts: {destination}") from error
    return destination


def validate_project_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_class_codes: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Validate the immutable metadata required by a project checkpoint."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise TrainingRunError(f"missing project checkpoint: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise TrainingRunError(f"unable to load project checkpoint: {path}") from error
    if not isinstance(value, Mapping):
        raise TrainingRunError("project checkpoint must be a mapping")
    required = {
        "checkpoint_format",
        "architecture",
        "class_count",
        "class_codes",
        "normalization_bounds",
        "input_size",
        "state_dict",
    }
    missing = required - set(value)
    if missing:
        raise TrainingRunError(
            f"project checkpoint metadata is incomplete: {', '.join(sorted(missing))}"
        )
    if value["checkpoint_format"] not in {
        _CHECKPOINT_FORMAT_V1,
        _CHECKPOINT_FORMAT_V2,
        _CHECKPOINT_FORMAT_V3,
        _CHECKPOINT_FORMAT_V4,
    }:
        raise TrainingRunError("unsupported project checkpoint format")
    if value["checkpoint_format"] in {_CHECKPOINT_FORMAT_V2, _CHECKPOINT_FORMAT_V3}:
        if value.get("feature_stride") != 16:
            version = value["checkpoint_format"].rsplit(".", 1)[-1]
            raise TrainingRunError(f"resnet18.{version} checkpoint feature_stride must be 16")
    if value["checkpoint_format"] == _CHECKPOINT_FORMAT_V3:
        for name in ("patch_size", "patch_stride"):
            item = value.get(name)
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                raise TrainingRunError(f"resnet18.v3 checkpoint {name} must be a positive integer")
        if value["patch_stride"] > value["patch_size"]:
            raise TrainingRunError(
                "resnet18.v3 checkpoint patch_stride cannot exceed patch_size"
            )
        if value.get("bag_pooling") != "max":
            raise TrainingRunError("resnet18.v3 checkpoint bag_pooling must be max")
    if value["checkpoint_format"] == _CHECKPOINT_FORMAT_V4:
        if value.get("feature_stride") != 4:
            raise TrainingRunError("resnet18.v4 checkpoint feature_stride must be 4")
        for name in ("patch_size", "patch_stride"):
            item = value.get(name)
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                raise TrainingRunError(
                    f"resnet18.v4 checkpoint {name} must be a positive integer"
                )
        if value["patch_stride"] > value["patch_size"]:
            raise TrainingRunError(
                "resnet18.v4 checkpoint patch_stride cannot exceed patch_size"
            )
        if value.get("training_policy") != "spatial_mil_v4":
            raise TrainingRunError(
                "resnet18.v4 checkpoint training_policy must be spatial_mil_v4"
            )
        if value.get("loss_weights") != {
            "positive_spatial_mil": 1.0,
            "absent_class_hard_negative": 1.0,
            "overlap_consistency": 1.0,
        }:
            raise TrainingRunError("resnet18.v4 checkpoint loss_weights are invalid")
        if value.get("positive_class_weighting") != {
            "formula": "negative_bag_count / positive_bag_count",
            "minimum": 1.0,
            "maximum": 10.0,
        }:
            raise TrainingRunError(
                "resnet18.v4 checkpoint positive_class_weighting is invalid"
            )
    expected_architecture = (
        "resnet18_spatial_logits"
        if value["checkpoint_format"] == _CHECKPOINT_FORMAT_V4
        else "resnet18"
    )
    if value["architecture"] != expected_architecture:
        raise TrainingRunError(
            f"project checkpoint architecture must be {expected_architecture}"
        )
    class_codes = value["class_codes"]
    if (
        not isinstance(class_codes, (list, tuple))
        or not class_codes
        or any(not isinstance(code, str) or not code for code in class_codes)
        or len(set(class_codes)) != len(class_codes)
    ):
        raise TrainingRunError("project checkpoint class_codes must be ordered unique strings")
    class_count = value["class_count"]
    if isinstance(class_count, bool) or not isinstance(class_count, int) or class_count != len(class_codes):
        raise TrainingRunError("project checkpoint class_count does not match class_codes")
    if expected_class_codes is not None and tuple(class_codes) != tuple(expected_class_codes):
        raise TrainingRunError("project checkpoint class order does not match the selected snapshot")
    bounds = value["normalization_bounds"]
    if not isinstance(bounds, (list, tuple)) or not bounds:
        raise TrainingRunError("project checkpoint normalization_bounds are incomplete")
    for item in bounds:
        if not isinstance(item, Mapping):
            raise TrainingRunError("project checkpoint normalization bounds are invalid")
        if set(item) != {
            "dtype",
            "source_min",
            "source_max",
            "low",
            "high",
            "low_percentile",
            "high_percentile",
        }:
            raise TrainingRunError("project checkpoint normalization bounds are incomplete")
    input_size = value["input_size"]
    if (
        not isinstance(input_size, Mapping)
        or set(input_size) != {"width", "height"}
        or any(
            isinstance(input_size[name], bool)
            or not isinstance(input_size[name], int)
            or input_size[name] < 1
            for name in ("width", "height")
        )
    ):
        raise TrainingRunError("project checkpoint input_size is invalid")
    state_dict = value["state_dict"]
    if (
        not isinstance(state_dict, Mapping)
        or not state_dict
        or any(not isinstance(name, str) or not isinstance(tensor, torch.Tensor) for name, tensor in state_dict.items())
    ):
        raise TrainingRunError("project checkpoint state_dict is invalid")
    return dict(value)


# Short aliases keep the project-service seam discoverable without exposing a
# database handle to worker code.
create_run = create_training_run
load_run = load_training_run
update_run_terminal = update_training_run_terminal
resume_run = resume_training_run
clone_run_after_oom = clone_after_oom


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
    elif version not in (
        _TRAINING_RUN_SCHEMA_VERSION,
        _EVALUATION_SCHEMA_VERSION,
        _DETECTION_SCHEMA_VERSION,
    ):
        raise TrainingRunError("Training Runs require project schema 12 or newer")


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


def _merge_environment(
    collected: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = _json_mapping(collected, "environment")
    if overrides is None:
        return merged
    explicit = _json_mapping(overrides, "environment")
    if "packages" in explicit:
        base_packages = merged.get("packages", {})
        override_packages = explicit["packages"]
        if not isinstance(base_packages, Mapping) or not isinstance(override_packages, Mapping):
            raise ValueError("environment packages must be mappings")
        explicit = dict(explicit)
        explicit["packages"] = {**base_packages, **override_packages}
    merged.update(explicit)
    return _json_mapping(merged, "environment")


def _require_complete_environment(environment: Mapping[str, Any]) -> None:
    missing = []
    for key in REQUIRED_ENVIRONMENT_KEYS:
        value = environment.get(key)
        if key == "packages":
            if not isinstance(value, Mapping) or not value:
                missing.append(key)
        elif not isinstance(value, str) or not value.strip():
            missing.append(key)
    if missing:
        raise TrainingRunError(
            "completed Training Runs require complete environment provenance; "
            f"missing or empty keys: {', '.join(missing)}"
        )


def _decode_mapping(serialized: str, name: str) -> dict[str, Any]:
    value = json.loads(serialized)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _child_config(
    parent: RunConfig,
    overrides: Mapping[str, Any] | None,
) -> RunConfig:
    if overrides is None:
        override_values: dict[str, Any] = {}
    elif isinstance(overrides, Mapping):
        override_values = dict(overrides)
    else:
        raise ValueError("config_overrides must be a mapping")
    fields_by_name = {item.name for item in fields(RunConfig)}
    unknown = set(override_values) - fields_by_name
    if unknown:
        raise ValueError(f"unsupported config override: {sorted(unknown)!r}")
    preserved = {"snapshot_id", "split_id", "class_count"}
    changed_preserved = preserved.intersection(override_values)
    if changed_preserved:
        raise TrainingRunError(
            "child Training Runs must preserve snapshot_id, split_id, and class_count"
        )
    values = asdict(parent)
    values.update(override_values)
    return RunConfig(**values)


def _has_checkpoint(run: TrainingRun) -> bool:
    for candidate in (run.artifact_path, run.staging_path):
        if candidate is None or not candidate.is_dir():
            continue
        try:
            validated = validate_staged_artifacts(candidate)
        except TrainingRunError:
            continue
        if any(path.name == _CHECKPOINT_NAME for path in validated):
            return True
    return False


def _is_oom_run(run: TrainingRun) -> bool:
    code = run.metrics.get("error_code")
    if isinstance(code, str) and code.lower() in _OOM_ERROR_CODES:
        return True
    normalized = run.terminal_message.lower().replace("-", "_").replace(" ", "_")
    return "out_of_memory" in normalized or "oom" in normalized.split("_")


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
    "clone_after_oom",
    "clone_run_after_oom",
    "load_run",
    "load_training_run",
    "publish_staged_artifacts",
    "update_run_terminal",
    "update_training_run_terminal",
    "validate_staged_artifacts",
    "validate_project_checkpoint",
    "resume_run",
    "resume_training_run",
]
