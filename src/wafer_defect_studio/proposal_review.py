"""Append-only review history for immutable :class:`DefectProposal` rows."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .detection_windows import Rect
from .project import (
    _DETECTION_SCHEMA_VERSION,
    _PROPOSAL_REVIEW_REVISIONS_TABLE_SQL,
    _PROPOSAL_REVIEW_SCHEMA_VERSION,
    _PROPOSAL_SCHEMA_VERSION,
    ProjectError,
    open_project,
)
from .proposal_generation import _json_safe
from .proposal_store import load_defect_proposal


_REVIEW_STATUSES = ("unreviewed", "accepted", "rejected", "corrected")


class ProposalReviewError(ProjectError):
    """Raised when a Defect Proposal review cannot be appended or loaded."""


class ProposalReviewStatus(str, Enum):
    """The statuses represented by one append-only proposal review revision."""

    UNREVIEWED = "unreviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CORRECTED = "corrected"


@dataclass(frozen=True, slots=True)
class ProposalReviewRevision:
    """One immutable status and geometry revision for a Defect Proposal."""

    proposal_id: str
    revision_number: int
    status: str
    source_rect: Rect
    provenance: dict[str, Any]

    @property
    def revision(self) -> int:
        """Short spelling for the monotonic revision number."""

        return self.revision_number

    @property
    def geometry(self) -> Rect:
        """Source-image geometry represented by this review revision."""

        return self.source_rect

    @property
    def rect(self) -> Rect:
        """Short geometry alias for review consumers."""

        return self.source_rect

    @property
    def source_image_rect(self) -> Rect:
        """Explicit source-image-coordinate geometry alias."""

        return self.source_rect

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""

        return {
            "proposal_id": self.proposal_id,
            "revision_number": self.revision_number,
            "status": self.status,
            "source_rect": _rect_dict(self.source_rect),
            "provenance": _json_safe(self.provenance),
        }


def record_review(
    project_path: str | Path,
    proposal_id: str,
    status: str | ProposalReviewStatus,
    source_rect: Rect | Sequence[int] | Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    *,
    rect: Rect | Sequence[int] | Mapping[str, Any] | None = None,
    geometry: Rect | Sequence[int] | Mapping[str, Any] | None = None,
    actor: str | None = None,
    reviewer: str | None = None,
    source_proposal_id: str | None = None,
) -> ProposalReviewRevision:
    """Append one proposal review revision without mutating annotations.

    ``accepted`` and ``rejected`` revisions retain the immutable proposal
    geometry.  ``corrected`` requires an explicit, different source rectangle;
    its provenance is augmented with the source proposal and manual action.
    """

    _identifier(proposal_id, "proposal_id")
    normalized_status = _status(status)
    chosen_rect = _coalesce_geometry(source_rect, rect, geometry)
    chosen_actor = _coalesce_text(actor, reviewer, "actor")
    provenance_value = _provenance(provenance)
    if chosen_actor is not None:
        provenance_value.setdefault("actor", chosen_actor)
    if source_proposal_id is not None:
        _identifier(source_proposal_id, "source_proposal_id")
        provenance_value["source_proposal_id"] = source_proposal_id

    info = open_project(project_path)
    if info.schema_version < _PROPOSAL_SCHEMA_VERSION:
        raise ProposalReviewError("Proposal review requires project schema 16 or newer")
    if info.schema_version > _PROPOSAL_REVIEW_SCHEMA_VERSION:
        raise ProposalReviewError(f"Unsupported project schema: {info.path / 'project.sqlite'}")

    database = info.path / "project.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        _ensure_schema(connection, info.project_id, database)
        row = connection.execute(
            "SELECT source_rect_json FROM defect_proposals WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        if row is None:
            raise ProposalReviewError(f"Unknown Defect Proposal: {proposal_id}")
        try:
            original_rect = _decode_rect(row[0])
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ProposalReviewError("Invalid Defect Proposal geometry") from error

        if normalized_status == "corrected":
            if chosen_rect is None:
                raise ProposalReviewError("Corrected review requires an explicit source_rect")
            normalized_rect = _rect(chosen_rect)
            if normalized_rect == original_rect:
                raise ProposalReviewError("Corrected review geometry must differ from the proposal")
            if not provenance_value:
                raise ProposalReviewError("Corrected review requires provenance")
            provenance_value.setdefault("source_proposal_id", proposal_id)
            provenance_value.setdefault("review_action", "manual_correction")
        else:
            normalized_rect = original_rect if chosen_rect is None else _rect(chosen_rect)
            if normalized_rect != original_rect:
                raise ProposalReviewError(
                    "Only corrected reviews may change Defect Proposal geometry"
                )
            provenance_value.setdefault("source_proposal_id", proposal_id)

        latest_row = connection.execute(
            "SELECT COALESCE(MAX(revision_number), 0) "
            "FROM proposal_review_revisions WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        revision_number = int(latest_row[0]) + 1
        connection.execute(
            "INSERT INTO proposal_review_revisions "
            "(proposal_id, revision_number, status, source_rect_json, provenance_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                proposal_id,
                revision_number,
                normalized_status,
                _encode_rect(normalized_rect),
                _encode_provenance(provenance_value),
            ),
        )
        connection.commit()
    except ProposalReviewError:
        connection.rollback()
        raise
    except sqlite3.IntegrityError as error:
        connection.rollback()
        raise ProposalReviewError("Proposal review identity is invalid") from error
    except sqlite3.Error as error:
        connection.rollback()
        raise ProposalReviewError(f"Invalid Proposal Review metadata: {database}") from error
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    return ProposalReviewRevision(
        proposal_id,
        revision_number,
        normalized_status,
        normalized_rect,
        provenance_value,
    )


def load_proposal_revisions(
    project_path: str | Path, proposal_id: str
) -> tuple[ProposalReviewRevision, ...]:
    """Load persisted proposal review revisions in ascending revision order."""

    _identifier(proposal_id, "proposal_id")
    proposal = _load_proposal(project_path, proposal_id)
    info = open_project(project_path)
    if info.schema_version < _PROPOSAL_REVIEW_SCHEMA_VERSION:
        return ()
    database = info.path / "project.sqlite"
    connection = _read_only(database)
    try:
        rows = connection.execute(
            "SELECT proposal_id, revision_number, status, source_rect_json, provenance_json "
            "FROM proposal_review_revisions WHERE proposal_id = ? ORDER BY revision_number",
            (proposal_id,),
        ).fetchall()
    except sqlite3.Error as error:
        raise ProposalReviewError(f"Invalid Proposal Review metadata: {database}") from error
    finally:
        connection.close()
    try:
        revisions = tuple(_decode_revision(row) for row in rows)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ProposalReviewError(f"Invalid Proposal Review metadata: {database}") from error
    for revision in revisions:
        if revision.proposal_id != proposal.proposal_id:
            raise ProposalReviewError(f"Invalid Proposal Review metadata: {database}")
    return revisions


def latest_proposal_review(
    project_path: str | Path, proposal_id: str
) -> ProposalReviewRevision:
    """Load the latest revision, or a virtual Unreviewed revision when absent."""

    proposal = _load_proposal(project_path, proposal_id)
    revisions = load_proposal_revisions(project_path, proposal_id)
    if revisions:
        return revisions[-1]
    return ProposalReviewRevision(
        proposal.proposal_id,
        0,
        "unreviewed",
        proposal.source_rect,
        {"source_proposal_id": proposal.proposal_id, "virtual": True},
    )


def load_proposal_revision(
    project_path: str | Path, proposal_id: str
) -> ProposalReviewRevision:
    """Singular/latest spelling for callers that need the current review."""

    return latest_proposal_review(project_path, proposal_id)


def _ensure_schema(connection: sqlite3.Connection, project_id: str, database: Path) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == _PROPOSAL_SCHEMA_VERSION:
        connection.execute(_PROPOSAL_REVIEW_REVISIONS_TABLE_SQL)
        _create_immutable_triggers(connection)
        updated = connection.execute(
            "UPDATE project_metadata SET schema_version = ? "
            "WHERE project_id = ? AND schema_version = ?",
            ( _PROPOSAL_REVIEW_SCHEMA_VERSION, project_id, _PROPOSAL_SCHEMA_VERSION),
        ).rowcount
        if updated != 1:
            raise ProposalReviewError(f"Invalid project metadata: {database}")
        connection.execute(f"PRAGMA user_version = {_PROPOSAL_REVIEW_SCHEMA_VERSION}")
    elif version == _PROPOSAL_REVIEW_SCHEMA_VERSION:
        connection.execute(_PROPOSAL_REVIEW_REVISIONS_TABLE_SQL)
        _create_immutable_triggers(connection)
    elif version < _PROPOSAL_SCHEMA_VERSION or version > _PROPOSAL_REVIEW_SCHEMA_VERSION:
        raise ProposalReviewError(f"Unsupported project schema: {database}")


def _create_immutable_triggers(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TRIGGER IF NOT EXISTS proposal_review_revisions_no_update "
        "BEFORE UPDATE ON proposal_review_revisions "
        "BEGIN SELECT RAISE(ABORT, 'Proposal Review revisions are immutable'); END"
    )
    connection.execute(
        "CREATE TRIGGER IF NOT EXISTS proposal_review_revisions_no_delete "
        "BEFORE DELETE ON proposal_review_revisions "
        "BEGIN SELECT RAISE(ABORT, 'Proposal Review revisions are immutable'); END"
    )


def _load_proposal(project_path: str | Path, proposal_id: str):
    try:
        return load_defect_proposal(project_path, proposal_id)
    except ProjectError as error:
        raise ProposalReviewError(str(error)) from error


def _decode_revision(row: Sequence[Any]) -> ProposalReviewRevision:
    status = _status(row[2])
    revision_number = row[1]
    if isinstance(revision_number, bool) or not isinstance(revision_number, int) or revision_number <= 0:
        raise ValueError("revision_number must be a positive integer")
    provenance = json.loads(row[4])
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be an object")
    return ProposalReviewRevision(
        _identifier(row[0], "proposal_id"),
        revision_number,
        status,
        _decode_rect(row[3]),
        _provenance(provenance),
    )


def _decode_rect(serialized: object) -> Rect:
    value = json.loads(serialized) if isinstance(serialized, str) else serialized
    return _rect(value)


def _rect(value: object) -> Rect:
    if isinstance(value, Rect):
        values = value
    elif isinstance(value, Mapping):
        values = (value.get("x"), value.get("y"), value.get("width"), value.get("height"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) == 4:
        values = value
    else:
        raise ValueError("source_rect must contain x, y, width, and height")
    return Rect(
        _coordinate(values[0], "source_rect.x", allow_zero=True),
        _coordinate(values[1], "source_rect.y", allow_zero=True),
        _coordinate(values[2], "source_rect.width"),
        _coordinate(values[3], "source_rect.height"),
    )


def _rect_dict(rect: Rect) -> dict[str, int]:
    return {"x": rect.x, "y": rect.y, "width": rect.width, "height": rect.height}


def _encode_rect(rect: Rect) -> str:
    return json.dumps(_rect_dict(rect), sort_keys=True, separators=(",", ":"))


def _provenance(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProposalReviewError("provenance must be a mapping")
    try:
        normalized = _json_safe(dict(value))
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ProposalReviewError("provenance must contain JSON-compatible values") from error
    return normalized


def _encode_provenance(value: Mapping[str, Any]) -> str:
    return json.dumps(_json_safe(dict(value)), sort_keys=True, separators=(",", ":"))


def _status(value: object) -> str:
    normalized = value.value if isinstance(value, ProposalReviewStatus) else value
    if not isinstance(normalized, str) or normalized not in _REVIEW_STATUSES:
        raise ProposalReviewError(f"unsupported Proposal Review status: {value!r}")
    return normalized


def _coordinate(value: object, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ProposalReviewError(f"{name} must be a {qualifier} integer")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProposalReviewError(f"{name} must be a non-empty string")
    if any(token in value for token in ("/", "\\", "..")):
        raise ProposalReviewError(f"{name} must be a path-safe identifier")
    return value


def _coalesce_geometry(
    first: Rect | Sequence[int] | Mapping[str, Any] | None,
    second: Rect | Sequence[int] | Mapping[str, Any] | None,
    third: Rect | Sequence[int] | Mapping[str, Any] | None,
) -> object | None:
    values = [value for value in (first, second, third) if value is not None]
    if len(values) > 1:
        normalized = tuple(_rect(value) for value in values)
        if any(candidate != normalized[0] for candidate in normalized[1:]):
            raise ProposalReviewError("source_rect aliases must identify the same geometry")
    return values[0] if values else None


def _coalesce_text(first: str | None, second: str | None, name: str) -> str | None:
    if first is not None and second is not None and first != second:
        raise ProposalReviewError(f"{name} aliases must identify the same value")
    if first is None and second is None:
        return None
    value = first if first is not None else second
    return _identifier(value, name)


def _read_only(database: Path) -> sqlite3.Connection:
    try:
        return sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise ProposalReviewError(f"Cannot open project database: {database}") from error


# Discoverable aliases follow existing project-service naming conventions.
record_proposal_review = record_review
review_proposal = record_review
proposal_revision = record_review
append_proposal_revision = record_review
load_latest_proposal_review = latest_proposal_review
latest_review = latest_proposal_review
list_proposal_revisions = load_proposal_revisions


__all__ = [
    "ProposalReviewError",
    "ProposalReviewStatus",
    "ProposalReviewRevision",
    "record_review",
    "record_proposal_review",
    "review_proposal",
    "proposal_revision",
    "append_proposal_revision",
    "load_proposal_revision",
    "load_proposal_revisions",
    "list_proposal_revisions",
    "latest_proposal_review",
    "load_latest_proposal_review",
    "latest_review",
]
