"""Spawn-safe Grid Evaluation worker and GUI-side staged publication.

The worker receives only value data from an immutable test split.  It never
opens SQLite or receives a project-service object; it writes one checksum-
declared ``metrics.json`` artifact to a staging directory.  The GUI-side
service validates that stage before calling :func:`create_evaluation`.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from uuid import uuid4
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .evaluation_run import EvaluationRecord, EvaluationRunError, create_evaluation
from .grid_evaluation import ClassMetrics, GridMetrics, compute_grid_metrics
from .model_scoring import score_training_bundle
from .thresholds import (
    DEFAULT_MAX_FPR_TARGET,
    DEFAULT_MIN_RECALL_TARGET,
    ThresholdPolicy,
    optimize_thresholds,
)
from .training_run import TrainingRunError, load_training_run, validate_staged_artifacts


PROTOCOL_VERSION = 1
_STAGE_FORMAT = "grid_evaluation_stage_v1"
_TERMINAL_STATUSES = {"completed", "cancelled", "failed", "interrupted"}


class EvaluationWorkerError(RuntimeError):
    """Raised when an Evaluation worker request or stage is invalid."""


class EvaluationProtocolError(ValueError):
    """Raised when a worker message is not a supported value message."""


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """Serializable test-split values passed across the process boundary."""

    request_id: str
    run_id: str
    snapshot_id: str
    split_id: str
    y_true: Sequence[Sequence[int]]
    y_score: Sequence[Sequence[float]]
    staging_path: str | Path
    class_names: Sequence[str] | None = None
    thresholds: float | Sequence[float] = 0.5
    policies: Any = None
    criteria: Mapping[str, Any] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("request_id", "run_id", "snapshot_id", "split_id"):
            _require_text(getattr(self, name), name)
        truth = _binary_matrix(self.y_true, "y_true")
        scores = _score_matrix(self.y_score)
        if truth.shape != scores.shape:
            raise EvaluationWorkerError("y_true and y_score must have the same 2-D shape")
        if truth.shape[0] == 0 or truth.shape[1] == 0:
            raise EvaluationWorkerError("y_true and y_score must contain data")
        object.__setattr__(self, "y_true", _matrix_values(truth.astype(np.int8)))
        object.__setattr__(self, "y_score", _matrix_values(scores))
        if self.class_names is not None:
            names = tuple(self.class_names)
            if len(names) != truth.shape[1] or any(
                not isinstance(name, str) or not name.strip() for name in names
            ):
                raise EvaluationWorkerError("class_names must match the class count")
            if len(set(names)) != len(names):
                raise EvaluationWorkerError("class_names must be unique")
            object.__setattr__(self, "class_names", names)
        object.__setattr__(
            self,
            "thresholds",
            _threshold_value(self.thresholds, truth.shape[1]),
        )
        object.__setattr__(self, "criteria", _json_mapping(self.criteria, "criteria"))
        object.__setattr__(
            self,
            "environment",
            _json_mapping(self.environment, "environment"),
        )
        if not isinstance(self.notes, str):
            raise EvaluationWorkerError("notes must be a string")
        if isinstance(self.staging_path, Path):
            object.__setattr__(self, "staging_path", str(self.staging_path))
        _require_text(self.staging_path, "staging_path")

    def to_payload(self) -> dict[str, Any]:
        """Return JSON-compatible values without a project handle."""

        return {
            "request_id": self.request_id,
            "run_id": self.run_id,
            "snapshot_id": self.snapshot_id,
            "split_id": self.split_id,
            "y_true": [list(row) for row in self.y_true],
            "y_score": [list(row) for row in self.y_score],
            "staging_path": str(self.staging_path),
            "class_names": list(self.class_names) if self.class_names is not None else None,
            "thresholds": list(self.thresholds) if isinstance(self.thresholds, tuple) else self.thresholds,
            "policies": _json_value(self.policies, "policies"),
            "criteria": dict(self.criteria),
            "environment": dict(self.environment),
            "notes": self.notes,
        }

    def to_json(self) -> str:
        return json.dumps(
            {"version": PROTOCOL_VERSION, "type": "evaluation_request", "payload": self.to_payload()},
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "EvaluationRequest":
        expected = {
            "request_id",
            "run_id",
            "snapshot_id",
            "split_id",
            "y_true",
            "y_score",
            "staging_path",
            "class_names",
            "thresholds",
            "policies",
            "criteria",
            "environment",
            "notes",
        }
        if set(value) != expected:
            raise EvaluationProtocolError("evaluation request fields are invalid")
        try:
            return cls(**dict(value))
        except (TypeError, ValueError) as error:
            if isinstance(error, EvaluationWorkerError):
                raise
            raise EvaluationProtocolError("invalid evaluation request") from error


@dataclass(frozen=True, slots=True)
class EvaluationProgress:
    """Truthful progress emitted by the worker process."""

    message_type: ClassVar[str] = "evaluation_progress"
    request_id: str
    phase: str
    completed: int
    total: int
    message: str = ""

    def __post_init__(self) -> None:
        _require_text(self.request_id, "request_id")
        _require_text(self.phase, "phase")
        _require_non_negative_int(self.completed, "completed")
        _require_positive_int(self.total, "total")
        if self.completed > self.total:
            raise EvaluationProtocolError("completed cannot exceed total")
        if not isinstance(self.message, str):
            raise EvaluationProtocolError("message must be a string")


@dataclass(frozen=True, slots=True)
class EvaluationTerminal:
    """Truthful completed, cancelled, or failed worker state."""

    message_type: ClassVar[str] = "evaluation_terminal"
    request_id: str
    status: str
    message: str = ""
    error_code: str | None = None
    artifact_staging_path: str | Path | None = None

    def __post_init__(self) -> None:
        _require_text(self.request_id, "request_id")
        if self.status not in _TERMINAL_STATUSES:
            raise EvaluationProtocolError("unsupported Evaluation terminal status")
        if not isinstance(self.message, str):
            raise EvaluationProtocolError("message must be a string")
        if self.error_code is not None:
            _require_text(self.error_code, "error_code")
        if isinstance(self.artifact_staging_path, Path):
            object.__setattr__(self, "artifact_staging_path", str(self.artifact_staging_path))


@dataclass(slots=True)
class EvaluationWorkerHandle:
    """Parent-side controls and output queue for one Evaluation worker."""

    process: mp.Process
    queue: Any
    cancel_event: Any
    request_id: str

    def cancel(self, reason: str = "user") -> bool:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        if self.cancel_event.is_set():
            return False
        self.cancel_event.set()
        return True

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self.process.join(timeout)

    @property
    def exitcode(self) -> int | None:
        return self.process.exitcode


def encode_message(message: EvaluationRequest | EvaluationProgress | EvaluationTerminal) -> str:
    """Encode a deterministic versioned worker message."""

    if isinstance(message, EvaluationRequest):
        message_type = "evaluation_request"
        payload = message.to_payload()
    elif isinstance(message, EvaluationProgress):
        message_type = message.message_type
        payload = {
            "request_id": message.request_id,
            "phase": message.phase,
            "completed": message.completed,
            "total": message.total,
            "message": message.message,
        }
    elif isinstance(message, EvaluationTerminal):
        message_type = message.message_type
        payload = {
            "request_id": message.request_id,
            "status": message.status,
            "message": message.message,
            "error_code": message.error_code,
            "artifact_staging_path": message.artifact_staging_path,
        }
    else:
        raise EvaluationProtocolError("unsupported Evaluation message")
    return json.dumps(
        {"version": PROTOCOL_VERSION, "type": message_type, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
    )


def decode_message(serialized: str) -> EvaluationRequest | EvaluationProgress | EvaluationTerminal:
    """Decode and validate one worker message."""

    if not isinstance(serialized, str):
        raise EvaluationProtocolError("serialized message must be a string")
    try:
        envelope = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise EvaluationProtocolError("invalid Evaluation message JSON") from error
    if not isinstance(envelope, dict) or set(envelope) != {"version", "type", "payload"}:
        raise EvaluationProtocolError("invalid Evaluation message envelope")
    if envelope["version"] != PROTOCOL_VERSION:
        raise EvaluationProtocolError("unsupported Evaluation message version")
    payload = envelope["payload"]
    if not isinstance(payload, dict):
        raise EvaluationProtocolError("Evaluation message payload must be an object")
    try:
        if envelope["type"] == "evaluation_request":
            return EvaluationRequest.from_payload(payload)
        if envelope["type"] == EvaluationProgress.message_type:
            expected = {"request_id", "phase", "completed", "total", "message"}
            if set(payload) != expected:
                raise EvaluationProtocolError("invalid Evaluation progress fields")
            return EvaluationProgress(**payload)
        if envelope["type"] == EvaluationTerminal.message_type:
            expected = {
                "request_id",
                "status",
                "message",
                "error_code",
                "artifact_staging_path",
            }
            if set(payload) != expected:
                raise EvaluationProtocolError("invalid Evaluation terminal fields")
            return EvaluationTerminal(**payload)
    except (TypeError, ValueError) as error:
        if isinstance(error, EvaluationProtocolError):
            raise
        raise EvaluationProtocolError("invalid Evaluation message payload") from error
    raise EvaluationProtocolError(f"unknown Evaluation message type: {envelope['type']!r}")


def start_evaluation_worker(
    request: EvaluationRequest,
    *,
    step_delay: float = 0.0,
    context: mp.context.BaseContext | None = None,
) -> EvaluationWorkerHandle:
    """Start an isolated Evaluation process using the platform spawn method."""

    if not isinstance(request, EvaluationRequest):
        raise EvaluationWorkerError("request must be an EvaluationRequest value")
    _validate_delay(step_delay)
    worker_context = context or mp.get_context("spawn")
    output_queue = worker_context.Queue()
    cancel_event = worker_context.Event()
    process = worker_context.Process(
        target=run_evaluation_worker,
        args=(request.to_json(), output_queue, cancel_event),
        kwargs={"step_delay": step_delay},
        name=f"wafer-evaluation-{request.request_id}",
        daemon=True,
    )
    process.start()
    return EvaluationWorkerHandle(process, output_queue, cancel_event, request.request_id)


def build_checkpoint_evaluation_request(
    project_path: str | Path,
    training_run_id: str,
    staging_path: str | Path,
    *,
    request_id: str | None = None,
    split: str = "test",
    thresholds: float | Sequence[float] = 0.5,
    policies: Any = None,
    criteria: Mapping[str, Any] | None = None,
    notes: str = "",
    device: str = "cpu",
) -> EvaluationRequest:
    """Build an Evaluation request from a completed run's checkpoint bundle."""

    try:
        run = load_training_run(project_path, training_run_id)
    except TrainingRunError as error:
        raise EvaluationWorkerError(str(error)) from error
    if run.status != "completed":
        raise EvaluationWorkerError("checkpoint Evaluation requires a completed Training Run")
    if run.artifact_path is None or not run.artifact_path.is_dir():
        raise EvaluationWorkerError("completed Training Run has no published artifact directory")
    bundle_path = run.artifact_path / "training_input_bundle.json"
    checkpoint_path = run.artifact_path / "model.pt"
    try:
        scores = score_training_bundle(
            bundle_path,
            checkpoint_path,
            split=split,
            device=device,
        )
    except ValueError as error:
        raise EvaluationWorkerError(f"unable to score Training Run {training_run_id}: {error}") from error
    return EvaluationRequest(
        request_id or str(uuid4()),
        run.run_id,
        run.snapshot_id,
        run.split_id,
        scores.y_true.tolist(),
        scores.y_score.tolist(),
        staging_path,
        class_names=scores.class_codes,
        thresholds=thresholds,
        policies=policies,
        criteria={} if criteria is None else criteria,
        environment=run.environment,
        notes=notes,
    )


def run_evaluation_worker(
    request: EvaluationRequest | Mapping[str, Any] | str,
    output_queue: Any,
    cancel_event: Any,
    *,
    step_delay: float = 0.0,
) -> None:
    """Compute and stage one immutable test-split Evaluation."""

    _validate_delay(step_delay)
    request_id = _request_id(request)
    request_value: EvaluationRequest | None = None
    try:
        request_value = _coerce_request(request)
        staging = Path(request_value.staging_path).expanduser().resolve()
        _emit(
            output_queue,
            EvaluationProgress(request_value.request_id, "evaluate", 0, 2, "Starting Grid Evaluation"),
        )
        if _is_cancelled(cancel_event):
            _emit_cancelled(output_queue, request_value)
            return
        if step_delay:
            time.sleep(step_delay)
        if _is_cancelled(cancel_event):
            _emit_cancelled(output_queue, request_value)
            return
        metrics, thresholds = _evaluate_values(request_value)
        _emit(
            output_queue,
            EvaluationProgress(request_value.request_id, "evaluate", 1, 2, "Computed Grid Evaluation metrics"),
        )
        if step_delay:
            time.sleep(step_delay)
        if _is_cancelled(cancel_event):
            _cleanup_stage(staging)
            _emit_cancelled(output_queue, request_value)
            return
        _write_staged_metrics(request_value, metrics, thresholds, staging)
        _emit(
            output_queue,
            EvaluationProgress(request_value.request_id, "stage", 2, 2, "Staged checksummed metrics"),
        )
        _emit(
            output_queue,
            EvaluationTerminal(
                request_value.request_id,
                "completed",
                "Evaluation completed; Training Run unchanged.",
                artifact_staging_path=str(staging),
            ),
        )
    except Exception as error:  # pragma: no cover - exercised by boundary tests
        if request_value is not None:
            _cleanup_stage(Path(request_value.staging_path).expanduser().resolve())
        _emit(
            output_queue,
            EvaluationTerminal(
                request_id,
                "failed",
                f"Evaluation worker failed: {error}",
                error_code="worker_error",
            ),
        )


def create_evaluation_from_staged(
    project_path: str | Path,
    staging_path: str | Path,
    *,
    training_run_id: str | None = None,
    actor: str = "system",
    evaluation_id: str | None = None,
) -> EvaluationRecord:
    """Validate one worker stage, then publish it through the project service."""

    staging = Path(staging_path).expanduser().resolve()
    try:
        validated = validate_staged_artifacts(staging)
    except TrainingRunError as error:
        raise EvaluationWorkerError(f"invalid staged Evaluation metrics: {error}") from error
    metrics_path = next((path for path in validated if path.name == "metrics.json"), None)
    if metrics_path is None:
        raise EvaluationWorkerError("staged Evaluation must declare metrics.json")
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise EvaluationWorkerError("invalid staged Evaluation metrics JSON") from error
    _validate_stage_payload(payload)
    run_id = payload["run_id"]
    if training_run_id is not None and training_run_id != run_id:
        raise EvaluationWorkerError("staged Evaluation Training Run does not match the request")
    try:
        run = load_training_run(project_path, run_id)
    except TrainingRunError as error:
        raise EvaluationWorkerError(str(error)) from error
    if payload["snapshot_id"] != run.snapshot_id or payload["split_id"] != run.split_id:
        raise EvaluationWorkerError("staged Evaluation provenance does not match the Training Run")
    try:
        return create_evaluation(
            project_path,
            training_run_id=run_id,
            metrics=payload["metrics"],
            thresholds=payload["thresholds"],
            criteria=payload["criteria"],
            environment=payload["environment"],
            notes=payload["notes"],
            actor=actor,
            evaluation_id=evaluation_id,
        )
    except (EvaluationRunError, TrainingRunError, ValueError) as error:
        raise EvaluationWorkerError(f"cannot publish staged Evaluation: {error}") from error


# Descriptive aliases keep the GUI service seam easy to discover.
publish_staged_evaluation = create_evaluation_from_staged
create_evaluation_from_stage = create_evaluation_from_staged


def _evaluate_values(
    request: EvaluationRequest,
) -> tuple[dict[str, Any], dict[str, Any]]:
    class_names = request.class_names
    if request.policies is None:
        grid = compute_grid_metrics(
            request.y_true,
            request.y_score,
            thresholds=request.thresholds,
            class_names=class_names,
        )
        rows = tuple(
            _threshold_row(metric, "fixed", None, None)
            for metric in grid.per_class
        )
        return _metrics_payload(grid), {"per_class": list(rows)}

    min_recall_target = _criteria_target(
        request.criteria,
        ("minimum_recall_target", "min_recall_target"),
        DEFAULT_MIN_RECALL_TARGET,
    )
    max_fpr_target = _criteria_target(
        request.criteria,
        ("maximum_fpr_target", "max_fpr_target"),
        DEFAULT_MAX_FPR_TARGET,
    )
    optimization = optimize_thresholds(
        request.y_true,
        request.y_score,
        policies=request.policies,
        min_recall_target=min_recall_target,
        max_fpr_target=max_fpr_target,
        class_names=class_names,
    )
    grid = compute_grid_metrics(
        request.y_true,
        request.y_score,
        thresholds=optimization.thresholds,
        class_names=class_names,
    )
    rows = tuple(
        _threshold_row(
            metric,
            selection.policy.value,
            selection.target,
            selection.target_satisfied,
        )
        for metric, selection in zip(grid.per_class, optimization.per_class)
    )
    return _metrics_payload(grid), {
        "per_class": list(rows),
        "min_recall_target": optimization.min_recall_target,
        "max_fpr_target": optimization.max_fpr_target,
    }


def _threshold_row(
    metric: ClassMetrics,
    policy: str,
    target: float | None,
    target_satisfied: bool | None,
) -> dict[str, Any]:
    row = {
        "class_name": metric.class_name,
        "policy": policy,
        "threshold": metric.threshold,
        "metrics": _class_metrics_payload(metric),
        "target": target,
    }
    if target_satisfied is not None:
        row["target_satisfied"] = target_satisfied
    return row


def _metrics_payload(metrics: GridMetrics) -> dict[str, Any]:
    return {
        "macro_f1": metrics.macro_f1,
        "per_class": [_class_metrics_payload(metric) for metric in metrics.per_class],
    }


def _class_metrics_payload(metric: ClassMetrics) -> dict[str, Any]:
    return {
        "class_name": metric.class_name,
        "precision": metric.precision,
        "recall": metric.recall,
        "f1": metric.f1,
        "support": metric.support,
        "fpr": metric.fpr,
        "true_positive": metric.true_positive,
        "true_negative": metric.true_negative,
        "false_positive": metric.false_positive,
        "false_negative": metric.false_negative,
        "threshold": metric.threshold,
    }


def _write_staged_metrics(
    request: EvaluationRequest,
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    staging: Path,
) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    metrics_path = staging / "metrics.json"
    manifest_path = staging / "manifest.json"
    payload = {
        "format": _STAGE_FORMAT,
        "request_id": request.request_id,
        "run_id": request.run_id,
        "snapshot_id": request.snapshot_id,
        "split_id": request.split_id,
        "metrics": dict(metrics),
        "thresholds": dict(thresholds),
        "criteria": dict(request.criteria),
        "environment": dict(request.environment),
        "notes": request.notes,
    }
    metrics_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "required_files": ["metrics.json"],
                "files": [{"path": "metrics.json", "sha256": _sha256(metrics_path)}],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def _validate_stage_payload(value: Any) -> None:
    if not isinstance(value, Mapping) or value.get("format") != _STAGE_FORMAT:
        raise EvaluationWorkerError("invalid staged Evaluation format")
    required = {
        "format",
        "request_id",
        "run_id",
        "snapshot_id",
        "split_id",
        "metrics",
        "thresholds",
        "criteria",
        "environment",
        "notes",
    }
    if set(value) != required:
        raise EvaluationWorkerError("staged Evaluation fields are invalid")
    for name in ("request_id", "run_id", "snapshot_id", "split_id"):
        _require_text(value[name], name)
    if not isinstance(value["notes"], str):
        raise EvaluationWorkerError("staged Evaluation notes must be text")
    for name in ("metrics", "thresholds", "criteria", "environment"):
        if not isinstance(value[name], Mapping):
            raise EvaluationWorkerError(f"staged Evaluation {name} must be an object")
    metrics = value["metrics"]
    macro_f1 = metrics.get("macro_f1")
    if not isinstance(macro_f1, (int, float)) or isinstance(macro_f1, bool) or not 0.0 <= float(macro_f1) <= 1.0:
        raise EvaluationWorkerError("staged Evaluation macro_f1 is invalid")
    rows = metrics.get("per_class")
    if not isinstance(rows, list) or not rows:
        raise EvaluationWorkerError("staged Evaluation must contain per-class metrics")
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise EvaluationWorkerError("staged per-class metrics must be objects")
        name = row.get("class_name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise EvaluationWorkerError("staged per-class names must be unique text")
        names.add(name)
        for metric_name in ("precision", "recall", "f1", "fpr"):
            metric_value = row.get(metric_name)
            if not isinstance(metric_value, (int, float)) or isinstance(metric_value, bool) or not 0.0 <= float(metric_value) <= 1.0:
                raise EvaluationWorkerError(f"staged metric {metric_name} is invalid")
        for count_name in ("support", "true_positive", "true_negative", "false_positive", "false_negative"):
            count = row.get(count_name)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise EvaluationWorkerError(f"staged metric {count_name} is invalid")


def _coerce_request(value: EvaluationRequest | Mapping[str, Any] | str) -> EvaluationRequest:
    if isinstance(value, EvaluationRequest):
        return value
    if isinstance(value, str):
        decoded = decode_message(value)
        if not isinstance(decoded, EvaluationRequest):
            raise EvaluationProtocolError("serialized value is not an EvaluationRequest")
        return decoded
    if isinstance(value, Mapping):
        return EvaluationRequest.from_payload(value)
    raise EvaluationWorkerError("request must be an EvaluationRequest, payload, or JSON value")


def _emit(output_queue: Any, message: EvaluationProgress | EvaluationTerminal) -> None:
    output_queue.put(encode_message(message))


def _emit_cancelled(output_queue: Any, request: EvaluationRequest) -> None:
    _emit(
        output_queue,
        EvaluationTerminal(
            request.request_id,
            "cancelled",
            "Evaluation cancelled; no Evaluation record was written.",
        ),
    )


def _cleanup_stage(staging: Path) -> None:
    for name in ("metrics.json", "manifest.json"):
        try:
            (staging / name).unlink(missing_ok=True)
        except OSError:
            pass


def _is_cancelled(cancel_event: Any) -> bool:
    return cancel_event is not None and bool(cancel_event.is_set())


def _request_id(value: object) -> str:
    if isinstance(value, EvaluationRequest):
        return value.request_id
    if isinstance(value, Mapping):
        raw = value.get("request_id")
        return raw if isinstance(raw, str) and raw.strip() else "unknown"
    return "unknown"


def _binary_matrix(value: object, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise EvaluationWorkerError(f"{name} must be a numeric 2-D matrix") from error
    if matrix.ndim != 2 or matrix.dtype.kind not in "biuf":
        raise EvaluationWorkerError(f"{name} must be a numeric 2-D matrix")
    if not np.all(np.isfinite(matrix)) or not np.all((matrix == 0) | (matrix == 1)):
        raise EvaluationWorkerError(f"{name} must contain only 0 or 1 values")
    return matrix


def _score_matrix(value: object) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise EvaluationWorkerError("y_score must be a numeric 2-D matrix") from error
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)) or not np.all((matrix >= 0.0) & (matrix <= 1.0)):
        raise EvaluationWorkerError("y_score must contain finite values from 0 to 1")
    return matrix


def _matrix_values(matrix: np.ndarray) -> tuple[tuple[int | float, ...], ...]:
    return tuple(tuple(item.item() for item in row) for row in matrix)


def _threshold_value(value: object, class_count: int) -> float | tuple[float, ...]:
    if isinstance(value, bool) or isinstance(value, (str, bytes)):
        raise EvaluationWorkerError("thresholds must be a number or one value per class")
    if isinstance(value, (int, float)):
        values = (float(value),)
        result: float | tuple[float, ...] = float(value)
    else:
        try:
            values = tuple(float(item) for item in value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise EvaluationWorkerError("thresholds must be a number or one value per class") from error
        if len(values) != class_count:
            raise EvaluationWorkerError("thresholds must match the class count")
        result = values
    if not all(isfinite(item) and 0.0 <= item <= 1.0 for item in values):
        raise EvaluationWorkerError("thresholds must contain finite values from 0 to 1")
    return result


def _criteria_target(criteria: Mapping[str, Any], names: tuple[str, ...], default: float) -> float:
    for name in names:
        value = criteria.get(name)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise EvaluationWorkerError(f"{name} must be a finite value from 0 to 1")
            return float(value)
    return default


def _json_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationWorkerError(f"{name} must be a mapping")
    result = {str(key): _json_value(item, name) for key, item in value.items()}
    return result


def _json_value(value: object, name: str) -> Any:
    if isinstance(value, ThresholdPolicy):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item, name) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item, name) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not isfinite(value):
            raise EvaluationWorkerError(f"{name} must contain finite JSON values")
        return value
    raise EvaluationWorkerError(f"{name} must contain JSON-compatible values")


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationWorkerError(f"{name} must be non-empty text")


def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvaluationProtocolError(f"{name} must be a positive integer")


def _require_non_negative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationProtocolError(f"{name} must be a non-negative integer")


def _validate_delay(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)) or value < 0:
        raise EvaluationWorkerError("step_delay must be a finite non-negative number")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "EvaluationProgress",
    "EvaluationProtocolError",
    "EvaluationRequest",
    "EvaluationTerminal",
    "EvaluationWorkerError",
    "EvaluationWorkerHandle",
    "build_checkpoint_evaluation_request",
    "create_evaluation_from_stage",
    "create_evaluation_from_staged",
    "decode_message",
    "encode_message",
    "publish_staged_evaluation",
    "run_evaluation_worker",
    "start_evaluation_worker",
]
