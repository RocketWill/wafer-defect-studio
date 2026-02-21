"""GUI-side persistence for immutable :class:`DefectProposal` rows."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .detection_run import load_detection_profile, load_detection_run
from .detection_windows import Rect
from .project import (
    _DEFECT_PROPOSALS_TABLE_SQL,
    _DEFECT_PROPOSAL_COLUMNS,
    _DETECTION_SCHEMA_VERSION,
    _PROPOSAL_SCHEMA_VERSION,
    _PROPOSAL_REVIEW_SCHEMA_VERSION,
    _PROPOSAL_CONVERSION_SCHEMA_VERSION,
    ProjectError,
    open_project,
)
from .proposal_generation import DefectProposal, _json_safe


class ProposalStoreError(ProjectError):
    """Raised when a Defect Proposal cannot be persisted or decoded."""


def save_defect_proposal(
    project_path: str | Path,
    proposal: DefectProposal,
    *,
    detection_run_id: str | None = None,
    profile_id: str | None = None,
    run_id: str | None = None,
) -> DefectProposal:
    """Persist one proposal and its immutable Detection Run/Profile identity.

    The writer is intentionally the project-side service.  A duplicate
    ``proposal_id`` is rejected; existing rows are never replaced.
    """

    value = _validate_proposal(proposal)
    chosen_run_id = _coalesce_id(detection_run_id, run_id, "detection_run_id")
    chosen_run_id = _association_id(value.provenance, chosen_run_id, "detection_run_id", "run_id")
    chosen_profile_id = _association_id(value.provenance, profile_id, "profile_id", "profile_id")
    _identifier(chosen_run_id, "detection_run_id")
    _identifier(chosen_profile_id, "profile_id")

    info = open_project(project_path)
    try:
        detection_run = load_detection_run(info.path, chosen_run_id)
        profile = load_detection_profile(info.path, chosen_profile_id)
    except (ProjectError, ValueError) as error:
        raise ProposalStoreError(str(error)) from error
    if detection_run.profile_id != profile.profile_id:
        raise ProposalStoreError("Defect Proposal Detection Run and Profile identities do not match")

    database = info.path / "project.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        _ensure_schema(connection, info.project_id, database)
        connection.execute(
            "INSERT INTO defect_proposals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                value.proposal_id,
                detection_run.run_id,
                profile.profile_id,
                value.class_name,
                _encode_rect(value.source_rect),
                value.area,
                value.peak_confidence,
                value.mean_confidence,
                _encode_provenance(value.provenance),
            ),
        )
        connection.commit()
    except sqlite3.IntegrityError as error:
        connection.rollback()
        raise ProposalStoreError("Defect Proposal identity is already used or invalid") from error
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return value


def load_defect_proposal(project_path: str | Path, proposal_id: str) -> DefectProposal:
    """Load one immutable Defect Proposal by identity."""

    _identifier(proposal_id, "proposal_id")
    info = open_project(project_path)
    if info.schema_version < _PROPOSAL_SCHEMA_VERSION:
        raise ProposalStoreError(f"Unknown Defect Proposal: {proposal_id}")
    connection = _read_only(info.path / "project.sqlite")
    try:
        row = connection.execute(
            "SELECT proposal_id, detection_run_id, profile_id, class_name, "
            "source_rect_json, area, peak_confidence, mean_confidence, provenance_json "
            "FROM defect_proposals WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ProposalStoreError(f"Unknown Defect Proposal: {proposal_id}")
    try:
        proposal = _decode_row(row)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ProposalStoreError("Invalid Defect Proposal metadata") from error
    _validate_association(proposal.provenance, row[1], row[2])
    return proposal


def load_defect_proposals(
    project_path: str | Path,
    *,
    detection_run_id: str | None = None,
    profile_id: str | None = None,
    run_id: str | None = None,
) -> tuple[DefectProposal, ...]:
    """Load proposals in deterministic identity order, optionally by run/profile."""

    chosen_run_id = _coalesce_id(detection_run_id, run_id, "detection_run_id")
    if chosen_run_id is not None:
        _identifier(chosen_run_id, "detection_run_id")
    if profile_id is not None:
        _identifier(profile_id, "profile_id")
    info = open_project(project_path)
    if info.schema_version < _PROPOSAL_SCHEMA_VERSION:
        return ()
    connection = _read_only(info.path / "project.sqlite")
    try:
        clauses: list[str] = []
        values: list[str] = []
        if chosen_run_id is not None:
            clauses.append("detection_run_id = ?")
            values.append(chosen_run_id)
        if profile_id is not None:
            clauses.append("profile_id = ?")
            values.append(profile_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = connection.execute(
            "SELECT proposal_id, detection_run_id, profile_id, class_name, "
            "source_rect_json, area, peak_confidence, mean_confidence, provenance_json "
            f"FROM defect_proposals{where} ORDER BY proposal_id",
            tuple(values),
        ).fetchall()
    finally:
        connection.close()
    try:
        proposals = tuple(_decode_row(row) for row in rows)
        for proposal, row in zip(proposals, rows):
            _validate_association(proposal.provenance, row[1], row[2])
        return proposals
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ProposalStoreError("Invalid Defect Proposal metadata") from error


def _ensure_schema(connection: sqlite3.Connection, project_id: str, database: Path) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == _DETECTION_SCHEMA_VERSION:
        connection.execute(_DEFECT_PROPOSALS_TABLE_SQL)
        connection.execute(
            "CREATE TRIGGER IF NOT EXISTS defect_proposals_no_update "
            "BEFORE UPDATE ON defect_proposals "
            "BEGIN SELECT RAISE(ABORT, 'Defect Proposals are immutable'); END"
        )
        connection.execute(
            "CREATE TRIGGER IF NOT EXISTS defect_proposals_no_delete "
            "BEFORE DELETE ON defect_proposals "
            "BEGIN SELECT RAISE(ABORT, 'Defect Proposals are immutable'); END"
        )
        updated = connection.execute(
            "UPDATE project_metadata SET schema_version = ? "
            "WHERE project_id = ? AND schema_version = ?",
            (_PROPOSAL_SCHEMA_VERSION, project_id, _DETECTION_SCHEMA_VERSION),
        ).rowcount
        if updated != 1:
            raise ProposalStoreError(f"Invalid project metadata: {database}")
        connection.execute(f"PRAGMA user_version = {_PROPOSAL_SCHEMA_VERSION}")
    elif version not in (
        _PROPOSAL_SCHEMA_VERSION,
        _PROPOSAL_REVIEW_SCHEMA_VERSION,
        _PROPOSAL_CONVERSION_SCHEMA_VERSION,
    ):
        raise ProposalStoreError("Defect Proposals require project schema 15 through 18")


def _decode_row(row: Sequence[Any]) -> DefectProposal:
    source_rect = json.loads(row[4])
    if not isinstance(source_rect, Mapping):
        raise ValueError("source_rect must be an object")
    rect = Rect(
        _positive_coordinate(source_rect.get("x"), "source_rect.x", allow_zero=True),
        _positive_coordinate(source_rect.get("y"), "source_rect.y", allow_zero=True),
        _positive_coordinate(source_rect.get("width"), "source_rect.width"),
        _positive_coordinate(source_rect.get("height"), "source_rect.height"),
    )
    provenance = json.loads(row[8])
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be an object")
    proposal = DefectProposal(
        proposal_id=row[0],
        class_name=row[3],
        source_rect=rect,
        area=_positive_coordinate(row[5], "area"),
        peak_confidence=_confidence(row[6], "peak_confidence"),
        mean_confidence=_confidence(row[7], "mean_confidence"),
        provenance=provenance,
    )
    return _validate_proposal(proposal)


def _validate_proposal(proposal: DefectProposal) -> DefectProposal:
    if not isinstance(proposal, DefectProposal):
        raise ValueError("proposal must be a DefectProposal")
    _identifier(proposal.proposal_id, "proposal_id")
    _text(proposal.class_name, "class_name")
    source_rect = _rect(proposal.source_rect)
    area = _positive_coordinate(proposal.area, "area")
    peak = _confidence(proposal.peak_confidence, "peak_confidence")
    mean = _confidence(proposal.mean_confidence, "mean_confidence")
    if not isinstance(proposal.provenance, Mapping):
        raise ValueError("provenance must be a mapping")
    try:
        provenance = _json_safe(dict(proposal.provenance))
        json.dumps(provenance, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("provenance must contain JSON-compatible values") from error
    return DefectProposal(
        proposal_id=proposal.proposal_id,
        class_name=proposal.class_name,
        source_rect=source_rect,
        area=area,
        peak_confidence=peak,
        mean_confidence=mean,
        provenance=provenance,
    )


def _rect(value: object) -> Rect:
    if isinstance(value, Rect):
        values = value
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) == 4:
        values = tuple(value)
    else:
        raise ValueError("source_rect must contain x, y, width, and height")
    return Rect(
        _positive_coordinate(values[0], "source_rect.x", allow_zero=True),
        _positive_coordinate(values[1], "source_rect.y", allow_zero=True),
        _positive_coordinate(values[2], "source_rect.width"),
        _positive_coordinate(values[3], "source_rect.height"),
    )


def _encode_rect(rect: Rect) -> str:
    return json.dumps(
        {"x": rect.x, "y": rect.y, "width": rect.width, "height": rect.height},
        sort_keys=True,
        separators=(",", ":"),
    )


def _encode_provenance(provenance: Mapping[str, Any]) -> str:
    return json.dumps(_json_safe(dict(provenance)), sort_keys=True, separators=(",", ":"))


def _validate_association(provenance: Mapping[str, Any], run_id: str, profile_id: str) -> None:
    _association_id(provenance, run_id, "detection_run_id", "run_id")
    _association_id(provenance, profile_id, "profile_id", "profile_id")


def _association_id(
    provenance: Mapping[str, Any],
    explicit: str | None,
    canonical_name: str,
    provenance_name: str,
) -> str | None:
    value = provenance.get(provenance_name)
    if value is None and canonical_name != provenance_name:
        value = provenance.get(canonical_name)
    if value is not None:
        _identifier(value, provenance_name)
        if explicit is not None and explicit != value:
            raise ProposalStoreError(f"Defect Proposal {canonical_name} does not match provenance")
        return str(value)
    return explicit


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


def _positive_coordinate(value: object, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _confidence(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite value from 0 to 1")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be a finite value from 0 to 1")
    return normalized


def _read_only(database: Path) -> sqlite3.Connection:
    try:
        return sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise ProposalStoreError(f"Cannot open project database: {database}") from error


# Discoverable aliases follow the existing project-service naming convention.
create_defect_proposal = save_defect_proposal
create_proposal = save_defect_proposal
save_proposal = save_defect_proposal
persist_proposal = save_defect_proposal
load_proposal = load_defect_proposal
get_defect_proposal = load_defect_proposal
get_proposal = load_defect_proposal
list_defect_proposals = load_defect_proposals
list_proposals = load_defect_proposals


__all__ = [
    "ProposalStoreError",
    "save_defect_proposal",
    "load_defect_proposal",
    "load_defect_proposals",
    "create_defect_proposal",
    "create_proposal",
    "save_proposal",
    "persist_proposal",
    "load_proposal",
    "get_defect_proposal",
    "get_proposal",
    "list_defect_proposals",
    "list_proposals",
]
