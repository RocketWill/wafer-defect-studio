"""Small parent-side adapter for truthful worker lifecycle messages.

The bridge deliberately knows only the value fields shared by the workers.
It does not import a worker implementation, open SQLite, or own a process.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from .job_state import (
    JobSnapshot,
    JobStateError,
    JobStatus,
    finish_job,
    request_job_cancel,
    start_job,
    update_job_progress,
)


class WorkerBridgeError(ValueError):
    """Raised when a worker event cannot be applied truthfully."""


def apply_worker_message(
    snapshot: JobSnapshot,
    message: object,
    *,
    validate_artifact: Callable[[str], object] | None = None,
    publish_artifact: Callable[[str], object] | None = None,
) -> JobSnapshot:
    """Apply one generic progress or terminal worker message.

    Worker implementations send value-only dataclasses (or mappings at a
    process boundary).  Completed is intentionally a two-callback gate:
    staging must validate and then publish before the state can be completed.
    """

    _require_snapshot(snapshot)
    request_id = _field(message, "request_id")
    if request_id is not None and request_id != snapshot.job_id:
        raise WorkerBridgeError(
            f"worker message request_id {request_id!r} does not match {snapshot.job_id!r}"
        )

    raw_status = _field(message, "status")
    if raw_status is None:
        return _apply_progress(snapshot, message)
    status = _status_value(raw_status)
    if status == "completed":
        return _apply_completed(snapshot, message, validate_artifact, publish_artifact)
    if status == "cancelled":
        return _finish_non_completed(
            snapshot,
            JobStatus.INTERRUPTED,
            message,
            default_error="cancelled",
        )
    if status == "interrupted":
        return _finish_non_completed(
            snapshot,
            JobStatus.INTERRUPTED,
            message,
            default_error="worker_interrupted",
        )
    if status == "failed":
        return _finish_non_completed(
            snapshot,
            JobStatus.FAILED,
            message,
            default_error="worker_failed",
        )
    raise WorkerBridgeError(f"unsupported worker terminal status: {status!r}")


def request_worker_cancel(
    snapshot: JobSnapshot,
    handle: object,
    reason: str = "user",
) -> JobSnapshot:
    """Ask an injected worker handle to cancel and enter ``cancelling``."""

    _require_snapshot(snapshot)
    if not isinstance(reason, str) or not reason.strip():
        raise WorkerBridgeError("reason must be a non-empty string")
    if snapshot.status is JobStatus.CANCELLING:
        return snapshot
    if snapshot.status is not JobStatus.RUNNING:
        raise WorkerBridgeError("only running jobs can be cancelled")
    cancel = getattr(handle, "cancel", None)
    if not callable(cancel):
        raise WorkerBridgeError("worker handle does not expose cancel(reason)")
    try:
        accepted = cancel(reason)
    except Exception as error:  # pragma: no cover - defensive process seam
        raise WorkerBridgeError(f"worker cancellation request failed: {error}") from error
    if not accepted:
        raise WorkerBridgeError("worker cancellation request was not accepted")
    try:
        return request_job_cancel(snapshot, message=f"Cancellation requested: {reason}")
    except (JobStateError, TypeError, ValueError) as error:
        raise WorkerBridgeError(str(error)) from error


def _apply_progress(snapshot: JobSnapshot, message: object) -> JobSnapshot:
    phase = _field(message, "phase")
    completed = _field(message, "completed")
    total = _field(message, "total")
    if completed is None:
        completed = _field(message, "step")
    if total is None:
        total = _field(message, "total_steps")
    if phase is None:
        phase = snapshot.phase
    if completed is None:
        completed = snapshot.completed
    if total is None:
        total = snapshot.total
    message_text = _field(message, "message")
    eta_seconds = _field(message, "eta_seconds")
    heartbeat_at = _field(message, "heartbeat_at")
    try:
        current = snapshot
        if current.status is JobStatus.QUEUED:
            current = start_job(
                current,
                phase=phase,
                message=message_text,
                heartbeat_at=heartbeat_at,
            )
        return update_job_progress(
            current,
            completed=completed,
            total=total,
            phase=phase,
            eta_seconds=eta_seconds,
            message=message_text,
            heartbeat_at=heartbeat_at,
        )
    except (JobStateError, TypeError, ValueError) as error:
        raise WorkerBridgeError(f"invalid worker progress: {error}") from error


def _apply_completed(
    snapshot: JobSnapshot,
    message: object,
    validate_artifact: Callable[[str], object] | None,
    publish_artifact: Callable[[str], object] | None,
) -> JobSnapshot:
    """Validate and publish staging before allowing a completed transition."""

    current = _ensure_started(snapshot, message)
    if current.status is JobStatus.CANCELLING:
        return _finish_non_completed(
            current,
            JobStatus.INTERRUPTED,
            message,
            default_error="cancelled",
        )
    staging = _staging_path(message)
    if staging is None:
        return _failed(current, message, "corrupt_staging", "Missing staged artifact path")
    if validate_artifact is None:
        return _failed(current, message, "corrupt_staging", "No artifact validator was provided", staging)
    try:
        valid = bool(validate_artifact(staging))
    except Exception as error:  # pragma: no cover - callback boundary
        return _failed(current, message, "corrupt_staging", f"Artifact validation failed: {error}", staging)
    if not valid:
        return _failed(current, message, "corrupt_staging", "Staged artifact validation failed", staging)
    if publish_artifact is None:
        return _failed(
            current,
            message,
            "artifact_publish_failed",
            "No artifact publisher was provided",
            staging,
        )
    try:
        published = publish_artifact(staging)
    except Exception as error:  # pragma: no cover - callback boundary
        return _failed(
            current,
            message,
            "artifact_publish_failed",
            f"Artifact publication failed: {error}",
            staging,
        )
    published_path = _published_path(published, staging)
    if published_path is None:
        return _failed(
            current,
            message,
            "artifact_publish_failed",
            "Artifact publisher did not return a published path",
            staging,
        )
    try:
        return finish_job(
            current,
            JobStatus.COMPLETED,
            artifact_validated=True,
            message=_message_text(message, "Completed"),
            error_code=None,
            staged_artifact_path=published_path,
            heartbeat_at=_field(message, "heartbeat_at"),
        )
    except (JobStateError, TypeError, ValueError) as error:
        raise WorkerBridgeError(f"could not complete worker job: {error}") from error


def _ensure_started(snapshot: JobSnapshot, message: object) -> JobSnapshot:
    if snapshot.status is not JobStatus.QUEUED:
        return snapshot
    try:
        return start_job(
            snapshot,
            phase=_field(message, "phase"),
            message=_field(message, "message"),
            heartbeat_at=_field(message, "heartbeat_at"),
        )
    except (JobStateError, TypeError, ValueError) as error:
        raise WorkerBridgeError(f"could not start worker job: {error}") from error


def _finish_non_completed(
    snapshot: JobSnapshot,
    status: JobStatus,
    message: object,
    *,
    default_error: str,
) -> JobSnapshot:
    current = _ensure_started(snapshot, message)
    error_code = _field(message, "error_code") or default_error
    try:
        return finish_job(
            current,
            status,
            message=_message_text(message, status.value.title()),
            error_code=error_code,
            staged_artifact_path=_field(message, "artifact_staging_path"),
            heartbeat_at=_field(message, "heartbeat_at"),
        )
    except (JobStateError, TypeError, ValueError) as error:
        raise WorkerBridgeError(f"could not apply worker terminal state: {error}") from error


def _failed(
    snapshot: JobSnapshot,
    message: object,
    error_code: str,
    text: str,
    staging: str | None = None,
) -> JobSnapshot:
    try:
        return finish_job(
            snapshot,
            JobStatus.FAILED,
            message=_message_text(message, text),
            error_code=error_code,
            staged_artifact_path=staging,
            heartbeat_at=_field(message, "heartbeat_at"),
        )
    except (JobStateError, TypeError, ValueError) as error:
        raise WorkerBridgeError(f"could not mark worker job failed: {error}") from error


def _staging_path(message: object) -> str | None:
    value = _field(message, "artifact_staging_path")
    if value is None:
        value = _field(message, "staging_path")
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _published_path(value: object, staging: str) -> str | None:
    if value is True:
        return staging
    if value is False or value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value if value.strip() else None
    return str(value) if value else None


def _message_text(message: object, default: str) -> str:
    value = _field(message, "message")
    return value if isinstance(value, str) and value else default


def _status_value(value: object) -> str:
    raw = getattr(value, "value", value)
    if not isinstance(raw, str):
        raise WorkerBridgeError(f"worker status must be text, got {raw!r}")
    return raw.strip().lower()


def _field(message: object, name: str, default: Any = None) -> Any:
    if isinstance(message, Mapping):
        return message.get(name, default)
    return getattr(message, name, default)


def _require_snapshot(value: object) -> None:
    if not isinstance(value, JobSnapshot):
        raise WorkerBridgeError("snapshot must be a JobSnapshot")


__all__ = [
    "WorkerBridgeError",
    "apply_worker_message",
    "request_worker_cancel",
]
