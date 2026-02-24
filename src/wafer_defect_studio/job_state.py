"""Pure, truthful lifecycle values for long-running background jobs.

The state machine intentionally has no worker, database, or widget dependency.
Callers receive a new immutable snapshot for every transition and must publish
it through their own persistence or UI seam.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any


class JobStateError(ValueError):
    """Raised when a Job snapshot transition would be misleading."""


class JobKind(str, Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"
    DETECTION = "detection"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


_TERMINAL_STATUSES = frozenset(
    (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.INTERRUPTED)
)


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    """Immutable state and diagnostics for one job attempt."""

    job_id: str
    kind: JobKind
    status: JobStatus = JobStatus.QUEUED
    phase: str = "queued"
    completed: int = 0
    total: int = 0
    eta_seconds: float | None = None
    message: str = ""
    log_path: str | Path | None = None
    heartbeat_at: str | datetime | None = None
    staged_artifact_path: str | Path | None = None
    attempt: int = 1
    parent_job_id: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _identifier(self.job_id, "job_id"))
        object.__setattr__(self, "kind", _enum(self.kind, JobKind, "kind"))
        object.__setattr__(self, "status", _enum(self.status, JobStatus, "status"))
        object.__setattr__(self, "phase", _text(self.phase, "phase"))
        _non_negative_int(self.completed, "completed")
        _non_negative_int(self.total, "total")
        if self.completed > self.total:
            raise JobStateError("completed cannot exceed total")
        _optional_non_negative_number(self.eta_seconds, "eta_seconds")
        if not isinstance(self.message, str):
            raise JobStateError("message must be a string")
        object.__setattr__(self, "log_path", _optional_path(self.log_path, "log_path"))
        object.__setattr__(self, "heartbeat_at", _optional_timestamp(self.heartbeat_at))
        object.__setattr__(
            self,
            "staged_artifact_path",
            _optional_path(self.staged_artifact_path, "staged_artifact_path"),
        )
        _positive_int(self.attempt, "attempt")
        if self.parent_job_id is not None:
            object.__setattr__(self, "parent_job_id", _identifier(self.parent_job_id, "parent_job_id"))
        if self.error_code is not None:
            object.__setattr__(self, "error_code", _identifier(self.error_code, "error_code"))

    @property
    def is_terminal(self) -> bool:
        """Whether no further state transition is allowed."""

        return self.status in _TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        """Return stable persistence-ready values without mutating this snapshot."""

        return {
            "job_id": self.job_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "phase": self.phase,
            "completed": self.completed,
            "total": self.total,
            "eta_seconds": self.eta_seconds,
            "message": self.message,
            "log_path": self.log_path,
            "heartbeat_at": self.heartbeat_at,
            "staged_artifact_path": self.staged_artifact_path,
            "attempt": self.attempt,
            "parent_job_id": self.parent_job_id,
            "error_code": self.error_code,
        }


def start_job(
    snapshot: JobSnapshot,
    *,
    phase: str | None = None,
    message: str | None = None,
    heartbeat_at: str | datetime | None = None,
) -> JobSnapshot:
    """Transition one queued job to running."""

    _require_snapshot(snapshot)
    if snapshot.status is not JobStatus.QUEUED:
        raise JobStateError("only queued jobs can start")
    return replace(
        snapshot,
        status=JobStatus.RUNNING,
        phase=phase or ("starting" if snapshot.phase == "queued" else snapshot.phase),
        message=snapshot.message if message is None else message,
        heartbeat_at=_heartbeat(heartbeat_at),
    )


def update_job_progress(
    snapshot: JobSnapshot,
    *,
    completed: int | None = None,
    total: int | None = None,
    phase: str | None = None,
    eta_seconds: float | None = None,
    message: str | None = None,
    heartbeat_at: str | datetime | None = None,
) -> JobSnapshot:
    """Update progress and heartbeat while a job is non-terminal."""

    _require_snapshot(snapshot)
    if snapshot.is_terminal:
        raise JobStateError("terminal job snapshots are immutable")
    return replace(
        snapshot,
        completed=snapshot.completed if completed is None else completed,
        total=snapshot.total if total is None else total,
        phase=snapshot.phase if phase is None else phase,
        eta_seconds=snapshot.eta_seconds if eta_seconds is None else eta_seconds,
        message=snapshot.message if message is None else message,
        heartbeat_at=_heartbeat(heartbeat_at),
    )


def request_job_cancel(
    snapshot: JobSnapshot,
    *,
    message: str = "Cancellation requested",
    heartbeat_at: str | datetime | None = None,
) -> JobSnapshot:
    """Request cooperative cancellation of a running job.

    Repeating the request while already cancelling is idempotent and refreshes
    the heartbeat.  A queued job has not started a worker and is left to the
    caller's queue policy rather than being mislabeled as cancelled.
    """

    _require_snapshot(snapshot)
    if snapshot.status is JobStatus.CANCELLING:
        return replace(snapshot, message=message, heartbeat_at=_heartbeat(heartbeat_at))
    if snapshot.status is not JobStatus.RUNNING:
        raise JobStateError("only running jobs can enter cancelling")
    return replace(
        snapshot,
        status=JobStatus.CANCELLING,
        message=message,
        heartbeat_at=_heartbeat(heartbeat_at),
    )


def finish_job(
    snapshot: JobSnapshot,
    status: JobStatus | str,
    *,
    artifact_validated: bool = False,
    message: str | None = None,
    error_code: str | None = None,
    staged_artifact_path: str | Path | None = None,
    heartbeat_at: str | datetime | None = None,
) -> JobSnapshot:
    """Move a non-terminal job to a truthful terminal state.

    ``completed`` is deliberately gated by an explicit validated-artifact
    acknowledgement.  Worker death or unknown termination should use
    ``interrupted`` and can never appear as ``completed``.
    """

    _require_snapshot(snapshot)
    target = _enum(status, JobStatus, "status")
    if target not in _TERMINAL_STATUSES:
        raise JobStateError("finish_job requires completed, failed, or interrupted")
    if snapshot.is_terminal:
        raise JobStateError("terminal job snapshots are immutable")
    if target is JobStatus.COMPLETED:
        if snapshot.status is not JobStatus.RUNNING:
            raise JobStateError("a cancelling job cannot be marked completed")
        if not isinstance(artifact_validated, bool) or not artifact_validated:
            raise JobStateError("completed requires artifact_validated=True")
    elif not isinstance(artifact_validated, bool):
        raise JobStateError("artifact_validated must be a boolean")
    if error_code is not None:
        _identifier(error_code, "error_code")
    terminal_message = message
    if terminal_message is None:
        terminal_message = {
            JobStatus.COMPLETED: "Completed",
            JobStatus.FAILED: "Failed",
            JobStatus.INTERRUPTED: "Interrupted",
        }[target]
    return replace(
        snapshot,
        status=target,
        message=terminal_message,
        error_code=error_code,
        staged_artifact_path=(
            snapshot.staged_artifact_path
            if staged_artifact_path is None
            else staged_artifact_path
        ),
        eta_seconds=None,
        heartbeat_at=_heartbeat(heartbeat_at),
    )


def interrupt_job(
    snapshot: JobSnapshot,
    *,
    error_code: str | None = None,
    message: str | None = None,
    heartbeat_at: str | datetime | None = None,
) -> JobSnapshot:
    """Convenience transition for forced or unknown worker termination."""

    return finish_job(
        snapshot,
        JobStatus.INTERRUPTED,
        error_code=error_code,
        message=message,
        heartbeat_at=heartbeat_at,
    )


def _require_snapshot(value: Any) -> None:
    if not isinstance(value, JobSnapshot):
        raise TypeError("snapshot must be a JobSnapshot")


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise JobStateError(f"{name} must be a supported value") from error


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JobStateError(f"{name} must be a non-empty string")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JobStateError(f"{name} must be a non-empty string")
    return value


def _non_negative_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JobStateError(f"{name} must be a non-negative integer")


def _positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise JobStateError(f"{name} must be a positive integer")


def _optional_non_negative_number(value: Any, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JobStateError(f"{name} must be a finite non-negative number or null")
    if not isfinite(float(value)) or float(value) < 0:
        raise JobStateError(f"{name} must be a finite non-negative number or null")


def _optional_path(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise JobStateError(f"{name} must be a non-empty path or null")
    return value


def _optional_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, str) and value.strip():
        return value
    raise JobStateError("heartbeat_at must be an ISO timestamp, datetime, or null")


def _heartbeat(value: str | datetime | None) -> str:
    return _optional_timestamp(value) or datetime.now(timezone.utc).isoformat()


__all__ = [
    "JobKind",
    "JobSnapshot",
    "JobStateError",
    "JobStatus",
    "finish_job",
    "interrupt_job",
    "request_job_cancel",
    "start_job",
    "update_job_progress",
]
