"""GUI-side SQLite persistence for :mod:`wafer_defect_studio.job_state`."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .job_state import JobKind, JobSnapshot, JobStateError, JobStatus
from .project import (
    _JOB_COLUMNS,
    _JOBS_TABLE_SQL,
    _JOB_SCHEMA_VERSION,
    _PROPOSAL_CONVERSION_SCHEMA_VERSION,
    ProjectError,
    open_project,
)


class JobStoreError(ProjectError):
    """Raised when a Job metadata operation cannot be committed safely."""


def create_job(project_path: str | Path, snapshot: JobSnapshot) -> JobSnapshot:
    """Persist a new job, migrating a schema-18 project to schema 19 once."""

    _validate_snapshot(snapshot)
    info = open_project(project_path)
    database = info.path / "project.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("BEGIN IMMEDIATE")
        _ensure_schema(connection, info.project_id, database)
        timestamp = _utc_now()
        connection.execute(
            "INSERT INTO jobs ("
            "job_id, kind, status, attempt, parent_job_id, phase, completed, total, "
            "eta_seconds, message, log_path, heartbeat_at, staged_artifact_path, "
            "error_code, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _values(snapshot, timestamp, timestamp),
        )
        connection.commit()
    except sqlite3.IntegrityError as error:
        connection.rollback()
        raise JobStoreError(f"Job identity is already used or invalid: {snapshot.job_id}") from error
    except JobStoreError:
        connection.rollback()
        raise
    except (sqlite3.Error, JobStateError, TypeError, ValueError) as error:
        connection.rollback()
        raise JobStoreError(f"Invalid Job metadata: {database}") from error
    finally:
        connection.close()
    return snapshot


def load_job(project_path: str | Path, job_id: str) -> JobSnapshot:
    """Load one immutable Job snapshot through a read-only SQLite connection."""

    job_id = _identifier(job_id, "job_id")
    info = _jobs_project(project_path)
    database = info.path / "project.sqlite"
    connection = _read_only(database)
    try:
        row = connection.execute(
            "SELECT " + ", ".join(_JOB_COLUMNS) + " FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    except sqlite3.Error as error:
        raise JobStoreError(f"Invalid Job metadata: {database}") from error
    finally:
        connection.close()
    if row is None:
        raise JobStoreError(f"Unknown Job: {job_id}")
    return _decode(row, database)


def list_jobs(
    project_path: str | Path,
    *,
    kind: JobKind | str | None = None,
    status: JobStatus | str | None = None,
) -> tuple[JobSnapshot, ...]:
    """Load Jobs deterministically, optionally filtering by kind or status."""

    info = _jobs_project(project_path)
    chosen_kind = _optional_enum(kind, JobKind, "kind")
    chosen_status = _optional_enum(status, JobStatus, "status")
    database = info.path / "project.sqlite"
    connection = _read_only(database)
    try:
        clauses: list[str] = []
        parameters: list[str] = []
        if chosen_kind is not None:
            clauses.append("kind = ?")
            parameters.append(chosen_kind.value)
        if chosen_status is not None:
            clauses.append("status = ?")
            parameters.append(chosen_status.value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = connection.execute(
            "SELECT " + ", ".join(_JOB_COLUMNS) + " FROM jobs" + where + " ORDER BY job_id",
            parameters,
        ).fetchall()
    except sqlite3.Error as error:
        raise JobStoreError(f"Invalid Job metadata: {database}") from error
    finally:
        connection.close()
    return tuple(_decode(row, database) for row in rows)


def save_job(project_path: str | Path, snapshot: JobSnapshot) -> JobSnapshot:
    """Save a non-terminal update or first terminal transition atomically."""

    _validate_snapshot(snapshot)
    info = _jobs_project(project_path)
    database = info.path / "project.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT status FROM jobs WHERE job_id = ?", (snapshot.job_id,)
        ).fetchone()
        if existing is None:
            raise JobStoreError(f"Unknown Job: {snapshot.job_id}")
        try:
            existing_status = JobStatus(existing[0])
        except (TypeError, ValueError) as error:
            raise JobStoreError(f"Invalid Job status: {snapshot.job_id}") from error
        if existing_status in _TERMINAL_STATUSES:
            raise JobStoreError(f"Terminal Job is immutable: {snapshot.job_id}")
        updated = connection.execute(
            "UPDATE jobs SET kind = ?, status = ?, attempt = ?, parent_job_id = ?, "
            "phase = ?, completed = ?, total = ?, eta_seconds = ?, message = ?, "
            "log_path = ?, heartbeat_at = ?, staged_artifact_path = ?, error_code = ?, "
            "updated_at = ? WHERE job_id = ?",
            (
                snapshot.kind.value,
                snapshot.status.value,
                snapshot.attempt,
                snapshot.parent_job_id,
                snapshot.phase,
                snapshot.completed,
                snapshot.total,
                snapshot.eta_seconds,
                snapshot.message,
                snapshot.log_path,
                snapshot.heartbeat_at,
                snapshot.staged_artifact_path,
                snapshot.error_code,
                _utc_now(),
                snapshot.job_id,
            ),
        ).rowcount
        if updated != 1:
            raise JobStoreError(f"Job update did not affect exactly one row: {snapshot.job_id}")
        connection.commit()
    except JobStoreError:
        connection.rollback()
        raise
    except sqlite3.IntegrityError as error:
        connection.rollback()
        raise JobStoreError(f"Invalid Job metadata: {snapshot.job_id}") from error
    except (sqlite3.Error, JobStateError, TypeError, ValueError) as error:
        connection.rollback()
        raise JobStoreError(f"Invalid Job metadata: {database}") from error
    finally:
        connection.close()
    return snapshot


_TERMINAL_STATUSES = frozenset(
    (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.INTERRUPTED)
)


def _ensure_schema(connection: sqlite3.Connection, project_id: str, database: Path) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == _PROPOSAL_CONVERSION_SCHEMA_VERSION:
        connection.execute(_JOBS_TABLE_SQL)
        updated = connection.execute(
            "UPDATE project_metadata SET schema_version = ? "
            "WHERE project_id = ? AND schema_version = ?",
            (_JOB_SCHEMA_VERSION, project_id, _PROPOSAL_CONVERSION_SCHEMA_VERSION),
        ).rowcount
        if updated != 1:
            raise JobStoreError(f"Invalid project metadata: {database}")
        connection.execute(f"PRAGMA user_version = {_JOB_SCHEMA_VERSION}")
    elif version == _JOB_SCHEMA_VERSION:
        connection.execute(_JOBS_TABLE_SQL)
    else:
        raise JobStoreError(
            f"Job persistence requires schema {_PROPOSAL_CONVERSION_SCHEMA_VERSION} or {_JOB_SCHEMA_VERSION}: {database}"
        )


def _jobs_project(project_path: str | Path):
    info = open_project(project_path)
    if info.schema_version != _JOB_SCHEMA_VERSION:
        raise JobStoreError(f"Job persistence requires schema {_JOB_SCHEMA_VERSION}: {info.path / 'project.sqlite'}")
    return info


def _read_only(database: Path) -> sqlite3.Connection:
    try:
        return sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise JobStoreError(f"Cannot open Job database: {database}") from error


def _decode(row: Sequence[Any], database: Path) -> JobSnapshot:
    try:
        return JobSnapshot(
            job_id=row[0],
            kind=row[1],
            status=row[2],
            attempt=row[3],
            parent_job_id=row[4],
            phase=row[5],
            completed=row[6],
            total=row[7],
            eta_seconds=row[8],
            message=row[9],
            log_path=row[10],
            heartbeat_at=row[11],
            staged_artifact_path=row[12],
            error_code=row[13],
        )
    except (JobStateError, TypeError, ValueError, IndexError) as error:
        raise JobStoreError(f"Invalid Job metadata: {database}") from error


def _values(snapshot: JobSnapshot, created_at: str, updated_at: str) -> tuple[Any, ...]:
    return (
        snapshot.job_id,
        snapshot.kind.value,
        snapshot.status.value,
        snapshot.attempt,
        snapshot.parent_job_id,
        snapshot.phase,
        snapshot.completed,
        snapshot.total,
        snapshot.eta_seconds,
        snapshot.message,
        snapshot.log_path,
        snapshot.heartbeat_at,
        snapshot.staged_artifact_path,
        snapshot.error_code,
        created_at,
        updated_at,
    )


def _validate_snapshot(snapshot: Any) -> None:
    if not isinstance(snapshot, JobSnapshot):
        raise TypeError("snapshot must be a JobSnapshot")


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JobStoreError(f"{name} must be a non-empty string")
    return value


def _optional_enum(value: Any, enum_type: type[JobKind] | type[JobStatus], name: str):
    if value is None:
        return None
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise JobStoreError(f"{name} must be a supported Job value") from error


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "JobStoreError",
    "create_job",
    "list_jobs",
    "load_job",
    "save_job",
]
