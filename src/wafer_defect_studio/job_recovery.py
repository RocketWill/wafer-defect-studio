"""Safe stale-job recovery and read-only staging reconciliation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any

from .job_state import JobSnapshot, JobStateError, JobStatus, interrupt_job
from .job_store import JobStoreError, list_jobs, save_job
from .project import open_project


class StagingDisposition(str, Enum):
    REFERENCED = "referenced"
    UNREFERENCED = "unreferenced"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class StagingRecord:
    """Read-only classification of one referenced or discovered staging path."""

    path: Path
    job_id: str | None
    disposition: StagingDisposition

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.job_id is not None and (not isinstance(self.job_id, str) or not self.job_id.strip()):
            raise ValueError("job_id must be a non-empty string or None")
        try:
            object.__setattr__(self, "disposition", StagingDisposition(self.disposition))
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported staging disposition") from error


def recover_stale_jobs(
    project_path: str | Path,
    *,
    now: datetime | str,
    stale_after_seconds: int | float = 60,
    process_is_alive: Callable[[JobSnapshot], bool] | None = None,
) -> tuple[JobSnapshot, ...]:
    """Mark only stale active jobs interrupted and persist that transition.

    A missing or malformed heartbeat is treated as stale.  An injected process
    checker may additionally report a worker as dead; queued and terminal jobs
    are never touched.  No staging path is removed or altered.
    """

    current = _timestamp(now, "now")
    threshold = _threshold(stale_after_seconds)
    if process_is_alive is not None and not callable(process_is_alive):
        raise TypeError("process_is_alive must be callable or None")

    recovered: list[JobSnapshot] = []
    for job in list_jobs(project_path):
        if job.status not in (JobStatus.RUNNING, JobStatus.CANCELLING):
            continue
        heartbeat_stale = _heartbeat_stale(job.heartbeat_at, current, threshold)
        process_dead = process_is_alive is not None and process_is_alive(job) is False
        if not heartbeat_stale and not process_dead:
            continue
        reason = "stale_worker" if heartbeat_stale else "worker_not_alive"
        message = (
            "Worker heartbeat is stale; job marked Interrupted."
            if heartbeat_stale
            else "Worker process is not alive; job marked Interrupted."
        )
        try:
            interrupted = interrupt_job(
                job,
                error_code=reason,
                message=message,
                heartbeat_at=current,
            )
            recovered.append(save_job(project_path, interrupted))
        except (JobStoreError, JobStateError, TypeError, ValueError):
            raise
    return tuple(recovered)


def scan_staging(project_path: str | Path) -> tuple[StagingRecord, ...]:
    """Classify staging paths without deleting or mutating any artifact."""

    info = open_project(project_path)
    jobs = list_jobs(project_path)
    root = info.path / "runs" / ".staging"
    references: dict[Path, str] = {}
    for job in jobs:
        if job.staged_artifact_path is None:
            continue
        path = _staging_path(info.path, root, job.staged_artifact_path)
        references[path] = job.job_id

    records: list[StagingRecord] = []
    for path, job_id in sorted(references.items(), key=lambda item: str(item[0])):
        records.append(
            StagingRecord(
                path,
                job_id,
                StagingDisposition.REFERENCED if path.exists() else StagingDisposition.MISSING,
            )
        )

    if root.is_dir():
        existing = sorted((path for path in root.rglob("*") if path.exists()), key=str)
        for path in existing:
            normalized = _resolved(path)
            if normalized in references:
                continue
            records.append(StagingRecord(normalized, None, StagingDisposition.UNREFERENCED))
    return tuple(sorted(records, key=lambda record: str(record.path)))


def _staging_path(project_root: Path, staging_root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return _resolved(candidate)
    project_candidate = _resolved(project_root / candidate)
    if project_candidate.exists() or str(candidate).replace("\\", "/").startswith("runs/"):
        return project_candidate
    staging_candidate = _resolved(staging_root / candidate)
    return staging_candidate


def _heartbeat_stale(value: str | None, now: datetime, threshold: float) -> bool:
    if value is None:
        return True
    try:
        heartbeat = _timestamp(value, "heartbeat_at")
    except ValueError:
        return True
    return (now - heartbeat).total_seconds() >= threshold


def _timestamp(value: datetime | str, name: str) -> datetime:
    if isinstance(value, datetime):
        chosen = value
    elif isinstance(value, str) and value.strip():
        serialized = value.strip()
        if serialized.endswith("Z"):
            serialized = serialized[:-1] + "+00:00"
        try:
            chosen = datetime.fromisoformat(serialized)
        except ValueError as error:
            raise ValueError(f"{name} must be an ISO UTC timestamp") from error
    else:
        raise ValueError(f"{name} must be an ISO UTC timestamp or datetime")
    if chosen.tzinfo is None:
        chosen = chosen.replace(tzinfo=timezone.utc)
    return chosen.astimezone(timezone.utc)


def _threshold(value: int | float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("stale_after_seconds must be a finite non-negative number")
    result = float(value)
    if not isfinite(result) or result < 0:
        raise ValueError("stale_after_seconds must be a finite non-negative number")
    return result


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


__all__ = [
    "StagingDisposition",
    "StagingRecord",
    "recover_stale_jobs",
    "scan_staging",
]
