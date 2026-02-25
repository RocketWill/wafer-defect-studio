"""Failure diagnostics and explicit, non-automatic recovery plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .job_state import JobKind, JobSnapshot, JobStatus


class JobActionError(ValueError):
    """Raised when a requested recovery action would be unsafe."""


class FailureKind(str, Enum):
    OUT_OF_MEMORY = "out_of_memory"
    MISSING_SOURCE = "missing_source"
    DISK_FULL = "disk_full"
    CORRUPT_STAGING = "corrupt_staging"
    WORKER_ERROR = "worker_error"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


class RecoveryAction(str, Enum):
    RESTART = "restart"
    RETRY = "retry"
    INSPECT_LOG = "inspect_log"
    CLONE_ADJUSTED = "clone_adjusted"


@dataclass(frozen=True, slots=True)
class FailureDiagnostic:
    """Immutable failure context shown to a recovery UI or audit log."""

    kind: FailureKind
    job_id: str
    error_code: str | None
    message: str
    log_path: str | Path | None
    staged_artifact_path: str | Path | None
    source_path: str | Path | None = None


@dataclass(frozen=True, slots=True)
class RecoveryPlan:
    """An explicit recovery choice and an optional new queued attempt."""

    action: RecoveryAction
    diagnostic: FailureDiagnostic
    new_job: JobSnapshot | None = None
    config_overrides: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.config_overrides is not None:
            if not isinstance(self.config_overrides, Mapping):
                raise JobActionError("config_overrides must be a mapping or null")
            object.__setattr__(
                self,
                "config_overrides",
                MappingProxyType(dict(self.config_overrides)),
            )

    @property
    def queued_job(self) -> JobSnapshot | None:
        """Alias for callers that name the planned attempt explicitly."""

        return self.new_job

    @property
    def new_snapshot(self) -> JobSnapshot | None:
        """Alias for the optional queued snapshot."""

        return self.new_job


_SAFE_ACTIONS: dict[FailureKind, tuple[RecoveryAction, ...]] = {
    FailureKind.OUT_OF_MEMORY: (
        RecoveryAction.CLONE_ADJUSTED,
        RecoveryAction.INSPECT_LOG,
    ),
    FailureKind.MISSING_SOURCE: (
        RecoveryAction.INSPECT_LOG,
        RecoveryAction.RESTART,
    ),
    FailureKind.DISK_FULL: (
        RecoveryAction.INSPECT_LOG,
        RecoveryAction.RETRY,
    ),
    FailureKind.CORRUPT_STAGING: (
        RecoveryAction.RETRY,
        RecoveryAction.INSPECT_LOG,
    ),
    FailureKind.INTERRUPTED: (
        RecoveryAction.RESTART,
        RecoveryAction.INSPECT_LOG,
    ),
    FailureKind.WORKER_ERROR: (
        RecoveryAction.RESTART,
        RecoveryAction.INSPECT_LOG,
    ),
    FailureKind.UNKNOWN: (RecoveryAction.INSPECT_LOG,),
}


def diagnose_failure(
    snapshot: JobSnapshot,
    source_path: str | Path | None = None,
) -> FailureDiagnostic:
    """Classify one immutable snapshot without touching source or staging."""

    _require_snapshot(snapshot)
    kind = _failure_kind(snapshot)
    return FailureDiagnostic(
        kind=kind,
        job_id=snapshot.job_id,
        error_code=snapshot.error_code,
        message=snapshot.message,
        log_path=snapshot.log_path,
        staged_artifact_path=snapshot.staged_artifact_path,
        source_path=source_path,
    )


def available_recovery_actions(
    snapshot: JobSnapshot,
    source_path: str | Path | None = None,
) -> tuple[RecoveryAction, ...]:
    """Return safe explicit actions for a failed or interrupted snapshot."""

    _require_snapshot(snapshot)
    if snapshot.status not in (JobStatus.FAILED, JobStatus.INTERRUPTED):
        return ()
    return _SAFE_ACTIONS[diagnose_failure(snapshot, source_path).kind]


def plan_recovery(
    snapshot: JobSnapshot,
    action: RecoveryAction | str,
    new_job_id: str,
    *,
    config_overrides: Mapping[str, Any] | None = None,
    current_batch_size: int | None = None,
) -> RecoveryPlan:
    """Build an immutable plan; no job is persisted or automatically queued."""

    _require_snapshot(snapshot)
    if snapshot.status not in (JobStatus.FAILED, JobStatus.INTERRUPTED):
        raise JobActionError("only failed or interrupted jobs can be recovered")
    try:
        recovery_action = _coerce_action(action)
    except (TypeError, ValueError) as error:
        raise JobActionError(f"unsupported recovery action: {action!r}") from error
    if not isinstance(new_job_id, str) or not new_job_id.strip():
        raise JobActionError("new_job_id must be a non-empty string")
    if new_job_id == snapshot.job_id:
        raise JobActionError("new_job_id must differ from the failed job")
    diagnostic = diagnose_failure(snapshot)
    if recovery_action not in _SAFE_ACTIONS[diagnostic.kind]:
        raise JobActionError(
            f"recovery action {recovery_action.value!r} is not safe for {diagnostic.kind.value!r}"
        )
    overrides = _copy_overrides(config_overrides)
    if recovery_action is RecoveryAction.CLONE_ADJUSTED:
        _require_smaller_batch(overrides, current_batch_size, diagnostic.kind)
    if recovery_action is RecoveryAction.INSPECT_LOG:
        return RecoveryPlan(recovery_action, diagnostic, config_overrides=overrides)
    queued = JobSnapshot(
        job_id=new_job_id,
        kind=snapshot.kind,
        status=JobStatus.QUEUED,
        phase="queued",
        completed=0,
        total=snapshot.total,
        eta_seconds=None,
        message=f"Recovery {recovery_action.value} for {snapshot.job_id}",
        log_path=None,
        heartbeat_at=None,
        staged_artifact_path=None,
        attempt=snapshot.attempt + 1,
        parent_job_id=snapshot.job_id,
        error_code=None,
    )
    return RecoveryPlan(recovery_action, diagnostic, queued, overrides)


def _failure_kind(snapshot: JobSnapshot) -> FailureKind:
    code = (snapshot.error_code or "").strip().lower()
    message = snapshot.message.strip().lower()
    combined = f"{code} {message}"
    if _contains(combined, "out_of_memory", "out of memory", "cuda oom", "oom"):
        return FailureKind.OUT_OF_MEMORY
    if _contains(
        combined,
        "missing_source",
        "source_missing",
        "missing source",
        "file_not_found",
        "file not found",
        "no such file",
    ):
        return FailureKind.MISSING_SOURCE
    if _contains(
        combined,
        "disk_full",
        "disk full",
        "no space left",
        "enospc",
        "insufficient disk",
    ):
        return FailureKind.DISK_FULL
    if _contains(
        combined,
        "corrupt_staging",
        "staging_corrupt",
        "corrupt staging",
        "corrupt artifact",
        "checksum mismatch",
    ):
        return FailureKind.CORRUPT_STAGING
    if snapshot.status is JobStatus.INTERRUPTED or _contains(
        combined,
        "interrupted",
        "cancelled",
        "canceled",
        "worker_killed",
    ):
        return FailureKind.INTERRUPTED
    if _contains(combined, "worker_error", "worker_failed", "worker error"):
        return FailureKind.WORKER_ERROR
    return FailureKind.UNKNOWN


def _contains(value: str, *needles: str) -> bool:
    return any(needle in value for needle in needles)


def _copy_overrides(value: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise JobActionError("config_overrides must be a mapping or null")
    return dict(value)


def _require_smaller_batch(
    overrides: Mapping[str, Any] | None,
    current_batch_size: int | None,
    kind: FailureKind,
) -> None:
    if kind is not FailureKind.OUT_OF_MEMORY:
        raise JobActionError("clone_adjusted is reserved for out-of-memory failures")
    if isinstance(current_batch_size, bool) or not isinstance(current_batch_size, int):
        raise JobActionError("current_batch_size must be a positive integer")
    if current_batch_size < 1:
        raise JobActionError("current_batch_size must be a positive integer")
    candidate = None if overrides is None else overrides.get("batch_size")
    if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < 1:
        raise JobActionError("OOM clone requires a positive batch_size override")
    if candidate >= current_batch_size:
        raise JobActionError("OOM clone batch_size must be smaller than the current batch size")


def _coerce_action(value: RecoveryAction | str) -> RecoveryAction:
    if isinstance(value, RecoveryAction):
        return value
    return RecoveryAction(value)


def _require_snapshot(value: object) -> None:
    if not isinstance(value, JobSnapshot):
        raise JobActionError("snapshot must be a JobSnapshot")


__all__ = [
    "FailureDiagnostic",
    "FailureKind",
    "JobActionError",
    "RecoveryAction",
    "RecoveryPlan",
    "available_recovery_actions",
    "diagnose_failure",
    "plan_recovery",
]
