"""Immutable Detection Profile and Detection Run persistence.

Detection execution is deliberately outside this module.  The project service
only records a reproducible, approved input contract and validates source
identity before a worker is allowed to stage maps.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluation_run import (
    EvaluationRunError,
    load_evaluation,
    load_evaluation_decisions,
)
from .project import (
    _DETECTION_PROFILES_TABLE_SQL,
    _DETECTION_RUNS_TABLE_SQL,
    _DETECTION_SCHEMA_VERSION,
    _EVALUATION_SCHEMA_VERSION,
    ProjectError,
    open_project,
)
from .training_run import TrainingRunError, load_training_run


_STATUSES = {"created", "running", "completed", "cancelled", "failed", "interrupted"}


class DetectionRunError(ProjectError):
    """Raised when a Detection Profile or Detection Run is invalid."""


@dataclass(frozen=True, slots=True)
class SourceFingerprint:
    """Source-image identity captured by a Detection Run."""

    image_asset_id: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class DetectionProfile:
    """An immutable set of native-resolution detection and map settings."""

    profile_id: str
    created_at: str
    evaluation_id: str
    training_run_id: str
    model_fingerprint: str
    window_size: tuple[int, int]
    stride: tuple[int, int]
    overlap: tuple[float, float]
    reflect_padding: bool
    thresholds: dict[str, Any]
    center_weighting: Any
    map_generation: dict[str, Any]

    @property
    def approved_evaluation_id(self) -> str:
        """Compatibility spelling for the required Approved Evaluation."""

        return self.evaluation_id

    @property
    def window_width(self) -> int:
        return self.window_size[0]

    @property
    def window_height(self) -> int:
        return self.window_size[1]

    @property
    def stride_x(self) -> int:
        return self.stride[0]

    @property
    def stride_y(self) -> int:
        return self.stride[1]

    @property
    def class_thresholds(self) -> dict[str, Any]:
        return self.thresholds


@dataclass(frozen=True, slots=True)
class DetectionRun:
    """One immutable application of an Approved model to source images."""

    run_id: str
    created_at: str
    evaluation_id: str
    training_run_id: str
    profile_id: str
    model_fingerprint: str
    source_fingerprints: dict[str, str]
    provenance: dict[str, Any]
    status: str
    staging_path: Path | None
    artifact_path: Path | None

    @property
    def detection_run_id(self) -> str:
        return self.run_id

    @property
    def approved_evaluation_id(self) -> str:
        return self.evaluation_id

    @property
    def source_fingerprint(self) -> str | None:
        return next(iter(self.source_fingerprints.values()), None) if len(self.source_fingerprints) == 1 else None


def create_detection_profile(
    project_path: str | Path,
    *,
    evaluation_id: str | None = None,
    approved_evaluation_id: str | None = None,
    training_run_id: str | None = None,
    window_size: Sequence[int] = (256, 256),
    stride: Sequence[int] | None = None,
    overlap: Sequence[float] | float | None = None,
    reflect_padding: bool = False,
    thresholds: Mapping[str, Any] | None = None,
    center_weighting: Any = "uniform",
    map_generation: Mapping[str, Any] | None = None,
    model_fingerprint: str | None = None,
    profile_id: str | None = None,
    created_at: str | None = None,
    previous: DetectionProfile | None = None,
    window_width: int | None = None,
    window_height: int | None = None,
    stride_x: int | None = None,
    stride_y: int | None = None,
    padding_mode: str | None = None,
    class_thresholds: Mapping[str, Any] | None = None,
    aggregation: Any | None = None,
    map_settings: Mapping[str, Any] | None = None,
) -> DetectionProfile:
    """Persist one profile after verifying an explicitly Approved Evaluation.

    Profiles are identity records rather than revisions.  Calling this with
    changed settings therefore creates a new UUID (or requires a new explicit
    ``profile_id``); existing rows can never be replaced.
    """

    chosen_evaluation = _coalesce_id(evaluation_id, approved_evaluation_id, "evaluation_id")
    _identifier(chosen_evaluation, "evaluation_id")
    if training_run_id is not None:
        _identifier(training_run_id, "training_run_id")
    if window_width is not None or window_height is not None:
        if window_width is None or window_height is None:
            raise ValueError("window_width and window_height must be provided together")
        window_size = (window_width, window_height)
    if stride_x is not None or stride_y is not None:
        if stride_x is None or stride_y is None:
            raise ValueError("stride_x and stride_y must be provided together")
        stride = (stride_x, stride_y)
    if padding_mode is not None:
        if padding_mode != "reflect":
            raise ValueError("padding_mode must be 'reflect'")
        reflect_padding = True
    if class_thresholds is not None:
        thresholds = class_thresholds
    if aggregation is not None:
        center_weighting = aggregation
    if map_settings is not None:
        map_generation = map_settings
    settings = _normalize_settings(
        window_size,
        stride,
        overlap,
        reflect_padding,
        thresholds,
        center_weighting,
        map_generation,
    )
    if previous is not None:
        if not isinstance(previous, DetectionProfile):
            raise ValueError("previous must be a DetectionProfile")
        if _profile_settings(previous) == settings:
            raise ValueError("Detection Profile settings are unchanged")
    chosen_id = profile_id or str(uuid.uuid4())
    _identifier(chosen_id, "profile_id")
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    _identifier(timestamp, "created_at")

    info = open_project(project_path)
    evaluation, training, fingerprint = _approved_model(
        info.path, chosen_evaluation, training_run_id, model_fingerprint
    )
    database = info.path / "project.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        _ensure_schema(connection, info.project_id, database)
        connection.execute(
            "INSERT INTO detection_profiles VALUES (?, ?, ?, ?, ?, ?)",
            (
                chosen_id,
                timestamp,
                evaluation.evaluation_id,
                training.run_id,
                fingerprint,
                _encode(settings),
            ),
        )
        connection.commit()
    except sqlite3.IntegrityError as error:
        connection.rollback()
        raise DetectionRunError("Detection Profile identity already exists or is invalid") from error
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return DetectionProfile(
        chosen_id,
        timestamp,
        evaluation.evaluation_id,
        training.run_id,
        fingerprint,
        settings["window_size"],
        settings["stride"],
        settings["overlap"],
        settings["reflect_padding"],
        settings["thresholds"],
        settings["center_weighting"],
        settings["map_generation"],
    )


def load_detection_profile(project_path: str | Path, profile_id: str) -> DetectionProfile:
    """Load one immutable Detection Profile."""

    _identifier(profile_id, "profile_id")
    info = open_project(project_path)
    if info.schema_version < _DETECTION_SCHEMA_VERSION:
        raise DetectionRunError(f"Unknown Detection Profile: {profile_id}")
    connection = _read_only(info.path / "project.sqlite")
    try:
        row = connection.execute(
            "SELECT profile_id, created_at, evaluation_id, training_run_id, "
            "model_fingerprint, settings_json FROM detection_profiles WHERE profile_id = ?",
            (profile_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise DetectionRunError(f"Unknown Detection Profile: {profile_id}")
    try:
        settings = _decode_settings(row[5])
        return DetectionProfile(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            settings["window_size"],
            settings["stride"],
            settings["overlap"],
            settings["reflect_padding"],
            settings["thresholds"],
            settings["center_weighting"],
            settings["map_generation"],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DetectionRunError("Invalid Detection Profile metadata") from error


def create_detection_run(
    project_path: str | Path,
    *,
    profile_id: str,
    evaluation_id: str | None = None,
    approved_evaluation_id: str | None = None,
    source_fingerprints: Mapping[str, str] | Sequence[Mapping[str, str]] | None = None,
    source_fingerprint: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    status: str = "created",
    staging_path: str | Path | None = None,
    artifact_path: str | Path | None = None,
    model_fingerprint: str | None = None,
    run_id: str | None = None,
    detection_run_id: str | None = None,
    created_at: str | None = None,
) -> DetectionRun:
    """Persist one immutable Detection Run after all identity guards pass."""

    _identifier(profile_id, "profile_id")
    chosen_evaluation = _coalesce_id(evaluation_id, approved_evaluation_id, "evaluation_id")
    if chosen_evaluation is not None:
        _identifier(chosen_evaluation, "evaluation_id")
    chosen_id = _coalesce_id(run_id, detection_run_id, "run_id") or str(uuid.uuid4())
    _identifier(chosen_id, "run_id")
    if status not in _STATUSES:
        raise DetectionRunError(f"unsupported Detection Run status: {status!r}")
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    _identifier(timestamp, "created_at")
    source_values = _normalize_sources(source_fingerprints, source_fingerprint)
    provenance_value = _json_object(provenance or {}, "provenance")

    info = open_project(project_path)
    profile = load_detection_profile(info.path, profile_id)
    if chosen_evaluation is not None and chosen_evaluation != profile.evaluation_id:
        raise DetectionRunError("Detection Run must use the profile's Approved Evaluation")
    chosen_evaluation = profile.evaluation_id
    evaluation, training, fingerprint = _approved_model(
        info.path, chosen_evaluation, profile.training_run_id, model_fingerprint
    )
    if fingerprint != profile.model_fingerprint:
        raise DetectionRunError("Approved model changed since the Detection Profile was created")
    _validate_sources(info.path, source_values)
    provenance_value = {
        **provenance_value,
        "evaluation_id": evaluation.evaluation_id,
        "training_run_id": training.run_id,
        "profile_id": profile.profile_id,
        "model_fingerprint": fingerprint,
        "source_coordinate_system": provenance_value.get(
            "source_coordinate_system", "source-image-pixels"
        ),
    }
    database = info.path / "project.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        _ensure_schema(connection, info.project_id, database)
        connection.execute(
            "INSERT INTO detection_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                chosen_id,
                timestamp,
                evaluation.evaluation_id,
                training.run_id,
                profile.profile_id,
                fingerprint,
                _encode(source_values),
                _encode(provenance_value),
                status,
                _path_value(staging_path),
                _path_value(artifact_path),
            ),
        )
        connection.commit()
    except sqlite3.IntegrityError as error:
        connection.rollback()
        raise DetectionRunError("Detection Run identity already exists or is invalid") from error
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return DetectionRun(
        chosen_id,
        timestamp,
        evaluation.evaluation_id,
        training.run_id,
        profile.profile_id,
        fingerprint,
        source_values,
        provenance_value,
        status,
        _path_object(staging_path),
        _path_object(artifact_path),
    )


def load_detection_run(project_path: str | Path, run_id: str) -> DetectionRun:
    """Load one immutable Detection Run."""

    _identifier(run_id, "run_id")
    info = open_project(project_path)
    if info.schema_version < _DETECTION_SCHEMA_VERSION:
        raise DetectionRunError(f"Unknown Detection Run: {run_id}")
    connection = _read_only(info.path / "project.sqlite")
    try:
        row = connection.execute(
            "SELECT detection_run_id, created_at, evaluation_id, training_run_id, profile_id, "
            "model_fingerprint, source_fingerprints_json, provenance_json, status, "
            "staging_path, artifact_path FROM detection_runs WHERE detection_run_id = ?",
            (run_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise DetectionRunError(f"Unknown Detection Run: {run_id}")
    try:
        sources = json.loads(row[6])
        provenance = json.loads(row[7])
        if not isinstance(sources, dict) or not isinstance(provenance, dict):
            raise ValueError
        return DetectionRun(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            {str(key): _text(value, "source fingerprint") for key, value in sources.items()},
            provenance,
            row[8],
            Path(row[9]) if row[9] else None,
            Path(row[10]) if row[10] else None,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise DetectionRunError("Invalid Detection Run metadata") from error


def _approved_model(
    project_path: Path,
    evaluation_id: str,
    training_run_id: str | None,
    expected_fingerprint: str | None,
) -> tuple[Any, Any, str]:
    try:
        evaluation = load_evaluation(project_path, evaluation_id)
        decisions = load_evaluation_decisions(project_path, evaluation_id)
    except (EvaluationRunError, ValueError) as error:
        raise DetectionRunError(str(error)) from error
    if not decisions or decisions[-1].status != "approved":
        raise DetectionRunError("Detection requires an Approved Evaluation")
    if training_run_id is not None and training_run_id != evaluation.training_run_id:
        raise DetectionRunError("Evaluation and Training Run identities do not match")
    try:
        training = load_training_run(project_path, evaluation.training_run_id)
    except (TrainingRunError, ValueError) as error:
        raise DetectionRunError(str(error)) from error
    fingerprint = _model_fingerprint(training)
    if expected_fingerprint is not None:
        # A caller may provide the Model Bundle's stable fingerprint.  The
        # fallback fingerprint is only used when no bundle identity exists;
        # the profile/run equality guard still rejects a changed identity.
        fingerprint = _text(expected_fingerprint, "model_fingerprint")
    return evaluation, training, fingerprint


def _model_fingerprint(training: Any) -> str:
    payload = {
        "run_id": training.run_id,
        "config": asdict(training.config),
        "environment": training.environment,
        "artifact_path": str(training.artifact_path) if training.artifact_path else None,
    }
    return hashlib.sha256(_encode(payload).encode("utf-8")).hexdigest()


def _validate_sources(project_path: Path, sources: Mapping[str, str]) -> None:
    connection = _read_only(project_path / "project.sqlite")
    try:
        for image_id, expected in sources.items():
            row = connection.execute(
                "SELECT fingerprint FROM image_assets WHERE image_asset_id = ?", (image_id,)
            ).fetchone()
            if row is None:
                raise DetectionRunError(f"Unknown source image: {image_id}")
            if row[0] != expected:
                raise DetectionRunError(f"Source image fingerprint changed: {image_id}")
    finally:
        connection.close()


def _ensure_schema(connection: sqlite3.Connection, project_id: str, database: Path) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == _EVALUATION_SCHEMA_VERSION:
        connection.execute(_DETECTION_PROFILES_TABLE_SQL)
        connection.execute(_DETECTION_RUNS_TABLE_SQL)
        connection.execute(
            "CREATE TRIGGER detection_profiles_no_update BEFORE UPDATE ON detection_profiles "
            "BEGIN SELECT RAISE(ABORT, 'Detection Profiles are immutable'); END"
        )
        connection.execute(
            "CREATE TRIGGER detection_profiles_no_delete BEFORE DELETE ON detection_profiles "
            "BEGIN SELECT RAISE(ABORT, 'Detection Profiles are immutable'); END"
        )
        connection.execute(
            "CREATE TRIGGER detection_runs_no_update BEFORE UPDATE ON detection_runs "
            "BEGIN SELECT RAISE(ABORT, 'Detection Runs are immutable'); END"
        )
        connection.execute(
            "CREATE TRIGGER detection_runs_no_delete BEFORE DELETE ON detection_runs "
            "BEGIN SELECT RAISE(ABORT, 'Detection Runs are immutable'); END"
        )
        updated = connection.execute(
            "UPDATE project_metadata SET schema_version = ? "
            "WHERE project_id = ? AND schema_version = ?",
            (_DETECTION_SCHEMA_VERSION, project_id, _EVALUATION_SCHEMA_VERSION),
        ).rowcount
        if updated != 1:
            raise DetectionRunError(f"Invalid project metadata: {database}")
        connection.execute(f"PRAGMA user_version = {_DETECTION_SCHEMA_VERSION}")
    elif version != _DETECTION_SCHEMA_VERSION:
        raise DetectionRunError("Detection Profiles require project schema 14 or 15")


def _normalize_settings(
    window_size: Sequence[int],
    stride: Sequence[int] | None,
    overlap: Sequence[float] | float | None,
    reflect_padding: bool,
    thresholds: Mapping[str, Any] | None,
    center_weighting: Any,
    map_generation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    window = _pair(window_size, "window_size", integer=True)
    if stride is None:
        if overlap is None:
            stride_value = window
        else:
            overlap_value = _overlap_pair(overlap)
            stride_value = tuple(
                max(1, int(round(window[index] * (1.0 - overlap_value[index]))))
                for index in range(2)
            )
    else:
        stride_value = _pair(stride, "stride", integer=True)
    if any(value > window[index] for index, value in enumerate(stride_value)):
        raise ValueError("stride must not exceed window_size")
    if overlap is None:
        overlap_value = tuple(
            (window[index] - stride_value[index]) / window[index] for index in range(2)
        )
    else:
        overlap_value = _overlap_pair(overlap)
        expected = tuple(
            (window[index] - stride_value[index]) / window[index] for index in range(2)
        )
        if any(abs(overlap_value[index] - expected[index]) > 1e-9 for index in range(2)):
            raise ValueError("overlap does not match window_size and stride")
    if not isinstance(reflect_padding, bool):
        raise ValueError("reflect_padding must be a boolean")
    threshold_value = _json_object(thresholds or {}, "thresholds")
    center_value = _json_value(center_weighting, "center_weighting")
    map_value = _json_object(map_generation or {}, "map_generation")
    return {
        "window_size": window,
        "stride": stride_value,
        "overlap": overlap_value,
        "reflect_padding": reflect_padding,
        "thresholds": threshold_value,
        "center_weighting": center_value,
        "map_generation": map_value,
    }


def _decode_settings(serialized: str) -> dict[str, Any]:
    value = json.loads(serialized)
    if not isinstance(value, dict):
        raise ValueError("settings must be a JSON object")
    return _normalize_settings(
        value["window_size"],
        value["stride"],
        value["overlap"],
        value["reflect_padding"],
        value["thresholds"],
        value["center_weighting"],
        value["map_generation"],
    )


def _profile_settings(profile: DetectionProfile) -> dict[str, Any]:
    return {
        "window_size": profile.window_size,
        "stride": profile.stride,
        "overlap": profile.overlap,
        "reflect_padding": profile.reflect_padding,
        "thresholds": profile.thresholds,
        "center_weighting": profile.center_weighting,
        "map_generation": profile.map_generation,
    }


def _normalize_sources(
    sources: Mapping[str, str] | Sequence[Mapping[str, str]] | None,
    source_fingerprint: str | None,
) -> dict[str, str]:
    if sources is None:
        if source_fingerprint is None:
            raise ValueError("source_fingerprints are required")
        raise ValueError("source_fingerprint requires one source_fingerprints mapping")
    if isinstance(sources, Mapping):
        values = dict(sources)
    elif isinstance(sources, Sequence) and not isinstance(sources, (str, bytes, bytearray)):
        values = {}
        for item in sources:
            if isinstance(item, SourceFingerprint):
                item = asdict(item)
            if not isinstance(item, Mapping):
                raise ValueError("source_fingerprints entries must be mappings")
            image_id = item.get("image_asset_id", item.get("image_id"))
            fingerprint = item.get("fingerprint")
            _identifier(image_id, "image_asset_id")
            _text(fingerprint, "fingerprint")
            values[str(image_id)] = str(fingerprint)
    else:
        raise ValueError("source_fingerprints must be a mapping or sequence")
    if not values:
        raise ValueError("source_fingerprints must not be empty")
    normalized: dict[str, str] = {}
    for image_id, fingerprint in values.items():
        _identifier(image_id, "image_asset_id")
        _text(fingerprint, "fingerprint")
        normalized[str(image_id)] = str(fingerprint)
    if source_fingerprint is not None and len(normalized) != 1:
        raise ValueError("source_fingerprint is only valid for one source")
    return dict(sorted(normalized.items()))


def _pair(value: Sequence[int], name: str, *, integer: bool) -> tuple[int, int]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"{name} values must be positive integers")
        result.append(item)
    return tuple(result)  # type: ignore[return-value]


def _overlap_pair(value: Sequence[float] | float) -> tuple[float, float]:
    values = (value, value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence) or len(values) != 2:
        raise ValueError("overlap must contain two fractions")
    result: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not 0 <= float(item) < 1:
            raise ValueError("overlap values must be in [0, 1)")
        result.append(float(item))
    return tuple(result)  # type: ignore[return-value]


def _json_object(value: object, name: str) -> dict[str, Any]:
    result = _json_value(value, name)
    if not isinstance(result, dict):
        raise ValueError(f"{name} must be a JSON object")
    return result


def _json_value(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        result = {str(key): _json_value(item, name) for key, item in value.items()}
    elif isinstance(value, (tuple, list)):
        result = [_json_value(item, name) for item in value]
    elif value is None or isinstance(value, (str, int, float, bool)):
        result = value
    else:
        raise ValueError(f"{name} must contain JSON-compatible values")
    try:
        json.dumps(result, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain JSON-compatible values") from error
    return result


def _encode(value: object) -> str:
    return json.dumps(_json_value(value, "metadata"), sort_keys=True, separators=(",", ":"))


def _read_only(database: Path) -> sqlite3.Connection:
    try:
        return sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise DetectionRunError(f"Cannot open project database: {database}") from error


def _coalesce_id(first: str | None, second: str | None, name: str) -> str | None:
    if first is not None and second is not None and first != second:
        raise ValueError(f"{name} aliases must identify the same record")
    return first if first is not None else second


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if any(token in value for token in ("/", "\\", "..")):
        raise ValueError(f"{name} must be a path-safe identifier")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _path_value(value: str | Path | None) -> str | None:
    return str(Path(value).expanduser().resolve()) if value is not None else None


def _path_object(value: str | Path | None) -> Path | None:
    return Path(value).expanduser().resolve() if value is not None else None


# Discoverable aliases follow the existing project-service naming convention.
create_profile = create_detection_profile
load_profile = load_detection_profile
create_run = create_detection_run
load_run = load_detection_run
save_detection_profile = create_detection_profile
get_detection_profile = load_detection_profile
save_detection_run = create_detection_run
get_detection_run = load_detection_run


__all__ = [
    "DetectionProfile",
    "DetectionRun",
    "DetectionRunError",
    "SourceFingerprint",
    "create_detection_profile",
    "load_detection_profile",
    "create_detection_run",
    "load_detection_run",
    "create_profile",
    "load_profile",
    "create_run",
    "load_run",
    "save_detection_profile",
    "get_detection_profile",
    "save_detection_run",
    "get_detection_run",
]
