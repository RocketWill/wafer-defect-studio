"""GUI-side persistence and audit history for immutable Grid Evaluations."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .project import (
    _EVALUATION_DECISIONS_TABLE_SQL,
    _EVALUATION_RUNS_TABLE_SQL,
    _EVALUATION_SCHEMA_VERSION,
    _TRAINING_RUN_SCHEMA_VERSION,
    ProjectError,
    open_project,
)
from .training_run import TrainingRunError, load_training_run


_DECISION_STATUSES = ("candidate", "validated", "approved")
_STATUS_ORDER = {status: index for index, status in enumerate(_DECISION_STATUSES)}


class EvaluationRunError(ProjectError):
    """Raised when an Evaluation or its audit history is invalid."""


class EvaluationStatus(str, Enum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    APPROVED = "approved"


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """Immutable provenance, metrics, thresholds, and criteria for one evaluation."""

    evaluation_id: str
    training_run_id: str
    snapshot_id: str
    split_id: str
    created_at: str
    environment: dict[str, Any]
    criteria: dict[str, Any]
    metrics: dict[str, Any]
    thresholds: dict[str, Any]
    target_satisfied: bool
    notes: str

    @property
    def run_id(self) -> str:
        """Compatibility spelling for the immutable Training Run identity."""

        return self.training_run_id


@dataclass(frozen=True, slots=True)
class EvaluationDecision:
    """One append-only Candidate, Validated, or Approved decision."""

    decision_id: str
    evaluation_id: str
    status: str
    actor: str
    decided_at: str
    criteria: dict[str, Any]
    target_satisfied: bool
    notes: str

def create_evaluation(
    project_path: str | Path,
    training_run_id: str | None = None,
    metrics: object | None = None,
    thresholds: object | None = None,
    criteria: Mapping[str, Any] | None = None,
    *,
    run_id: str | None = None,
    environment: Mapping[str, Any] | None = None,
    notes: str = "",
    actor: str = "system",
    evaluation_id: str | None = None,
    created_at: str | None = None,
) -> EvaluationRecord:
    """Persist one immutable Evaluation and its initial Candidate decision."""

    chosen_run_id = training_run_id or run_id
    _identifier(chosen_run_id, "training_run_id")
    if training_run_id is not None and run_id is not None and training_run_id != run_id:
        raise ValueError("training_run_id and run_id must identify the same Training Run")
    if metrics is None or thresholds is None:
        raise ValueError("metrics and thresholds are required")
    criteria_value = _json_object(criteria or {}, "criteria")
    environment_value = None if environment is None else _json_object(environment, "environment")
    notes = _text(notes, "notes")
    actor = _identifier(actor, "actor")
    metric_value = _json_object(metrics, "metrics")
    threshold_value = _json_object(thresholds, "thresholds")
    target_satisfied = _target_satisfied(threshold_value, metric_value, criteria_value)
    chosen_evaluation_id = evaluation_id or str(uuid.uuid4())
    _identifier(chosen_evaluation_id, "evaluation_id")
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    _identifier(timestamp, "created_at")

    try:
        run = load_training_run(project_path, chosen_run_id)
    except TrainingRunError as error:
        raise EvaluationRunError(str(error)) from error
    if run.snapshot_id == "" or run.split_id == "":
        raise EvaluationRunError("Training Run provenance is incomplete")
    if environment_value is None:
        environment_value = dict(run.environment)

    info = open_project(project_path)
    database = info.path / "project.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        _ensure_schema(connection, info.project_id, database)
        connection.execute(
            "INSERT INTO evaluation_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                chosen_evaluation_id,
                chosen_run_id,
                run.snapshot_id,
                run.split_id,
                timestamp,
                _encode(environment_value),
                _encode(criteria_value),
                _encode(metric_value),
                _encode(threshold_value),
                int(target_satisfied),
                notes,
            ),
        )
        _insert_decision(
            connection,
            EvaluationDecision(
                str(uuid.uuid4()),
                chosen_evaluation_id,
                "candidate",
                actor,
                timestamp,
                criteria_value,
                target_satisfied,
                notes,
            ),
        )
        connection.commit()
    except sqlite3.IntegrityError as error:
        connection.rollback()
        raise EvaluationRunError("Evaluation identity already exists or is invalid") from error
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return EvaluationRecord(
        chosen_evaluation_id,
        chosen_run_id,
        run.snapshot_id,
        run.split_id,
        timestamp,
        environment_value,
        criteria_value,
        metric_value,
        threshold_value,
        target_satisfied,
        notes,
    )


def load_evaluation(project_path: str | Path, evaluation_id: str) -> EvaluationRecord:
    """Read one immutable Evaluation through SQLite's read-only URI."""

    _identifier(evaluation_id, "evaluation_id")
    info = open_project(project_path)
    if info.schema_version < _EVALUATION_SCHEMA_VERSION:
        raise EvaluationRunError(f"Unknown Evaluation: {evaluation_id}")
    connection = _read_only(info.path / "project.sqlite")
    try:
        row = connection.execute(
            "SELECT evaluation_id, training_run_id, snapshot_id, split_id, created_at, "
            "environment_json, criteria_json, metrics_json, thresholds_json, "
            "target_satisfied, notes FROM evaluation_runs WHERE evaluation_id = ?",
            (evaluation_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise EvaluationRunError(f"Unknown Evaluation: {evaluation_id}")
    try:
        return EvaluationRecord(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            _decode_object(row[5], "environment"),
            _decode_object(row[6], "criteria"),
            _decode_object(row[7], "metrics"),
            _decode_object(row[8], "thresholds"),
            bool(row[9]),
            row[10],
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise EvaluationRunError("Invalid Evaluation metadata") from error


def list_evaluations(
    project_path: str | Path, training_run_id: str | None = None
) -> tuple[EvaluationRecord, ...]:
    """Read all immutable Evaluations in creation order."""

    if training_run_id is not None:
        _identifier(training_run_id, "training_run_id")
    info = open_project(project_path)
    if info.schema_version < _EVALUATION_SCHEMA_VERSION:
        return ()
    connection = _read_only(info.path / "project.sqlite")
    try:
        if training_run_id is None:
            rows = connection.execute(
                "SELECT evaluation_id FROM evaluation_runs ORDER BY created_at, rowid"
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT evaluation_id FROM evaluation_runs WHERE training_run_id = ? "
                "ORDER BY created_at, rowid",
                (training_run_id,),
            ).fetchall()
    finally:
        connection.close()
    return tuple(load_evaluation(info.path, row[0]) for row in rows)


def record_evaluation_decision(
    project_path: str | Path,
    evaluation_id: str,
    status: EvaluationStatus | str,
    actor: str,
    *,
    criteria: Mapping[str, Any] | None = None,
    notes: str = "",
    decision_id: str | None = None,
    decided_at: str | None = None,
) -> EvaluationDecision:
    """Append one auditable status transition without rewriting prior rows."""

    _identifier(evaluation_id, "evaluation_id")
    chosen_status = _coerce_status(status)
    actor = _identifier(actor, "actor")
    notes = _text(notes, "notes")
    if chosen_status == "approved" and not notes:
        raise EvaluationRunError("Approved decisions require non-empty notes")
    evaluation = load_evaluation(project_path, evaluation_id)
    criteria_value = _json_object(
        evaluation.criteria if criteria is None else criteria,
        "criteria",
    )
    allow_unmet = _allows_unmet(criteria_value)
    if not evaluation.target_satisfied and chosen_status in {"validated", "approved"} and not allow_unmet:
        raise EvaluationRunError("minimum-recall target is unmet; explicit criteria override is required")

    info = open_project(project_path)
    database = info.path / "project.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        _ensure_schema(connection, info.project_id, database)
        previous = connection.execute(
            "SELECT status FROM evaluation_decisions WHERE evaluation_id = ? "
            "ORDER BY decided_at DESC, rowid DESC LIMIT 1",
            (evaluation_id,),
        ).fetchone()
        if previous is None:
            if chosen_status != "candidate":
                raise EvaluationRunError("an Evaluation must begin as Candidate")
        elif _STATUS_ORDER[chosen_status] <= _STATUS_ORDER[previous[0]]:
            raise EvaluationRunError("Evaluation decisions must move forward append-only")
        if chosen_status == "approved" and (previous is None or previous[0] != "validated"):
            raise EvaluationRunError("Approved requires a preceding Validated decision")
        timestamp = decided_at or datetime.now(timezone.utc).isoformat()
        _identifier(timestamp, "decided_at")
        decision = EvaluationDecision(
            decision_id or str(uuid.uuid4()),
            evaluation_id,
            chosen_status,
            actor,
            timestamp,
            criteria_value,
            evaluation.target_satisfied,
            notes,
        )
        _identifier(decision.decision_id, "decision_id")
        _insert_decision(connection, decision)
        connection.commit()
    except EvaluationRunError:
        connection.rollback()
        raise
    except sqlite3.IntegrityError as error:
        connection.rollback()
        raise EvaluationRunError("Evaluation decision identity is already used") from error
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return decision


def load_evaluation_decisions(
    project_path: str | Path, evaluation_id: str
) -> tuple[EvaluationDecision, ...]:
    """Read the complete append-only status history for one Evaluation."""

    _identifier(evaluation_id, "evaluation_id")
    info = open_project(project_path)
    if info.schema_version < _EVALUATION_SCHEMA_VERSION:
        raise EvaluationRunError(f"Unknown Evaluation: {evaluation_id}")
    connection = _read_only(info.path / "project.sqlite")
    try:
        rows = connection.execute(
            "SELECT decision_id, evaluation_id, status, actor, decided_at, criteria_json, "
            "target_satisfied, notes FROM evaluation_decisions "
            "WHERE evaluation_id = ? ORDER BY decided_at, rowid",
            (evaluation_id,),
        ).fetchall()
    finally:
        connection.close()
    try:
        return tuple(
            EvaluationDecision(
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                _decode_object(row[5], "criteria"),
                bool(row[6]),
                row[7],
            )
            for row in rows
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise EvaluationRunError("Invalid Evaluation decision metadata") from error


def _insert_decision(connection: sqlite3.Connection, decision: EvaluationDecision) -> None:
    connection.execute(
        "INSERT INTO evaluation_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            decision.decision_id,
            decision.evaluation_id,
            decision.status,
            decision.actor,
            decision.decided_at,
            _encode(decision.criteria),
            int(decision.target_satisfied),
            decision.notes,
        ),
    )


def _ensure_schema(connection: sqlite3.Connection, project_id: str, database: Path) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == _TRAINING_RUN_SCHEMA_VERSION:
        connection.execute(_EVALUATION_RUNS_TABLE_SQL)
        connection.execute(_EVALUATION_DECISIONS_TABLE_SQL)
        connection.execute(
            "CREATE TRIGGER evaluation_runs_no_update BEFORE UPDATE ON evaluation_runs "
            "BEGIN SELECT RAISE(ABORT, 'Evaluations are immutable'); END"
        )
        connection.execute(
            "CREATE TRIGGER evaluation_runs_no_delete BEFORE DELETE ON evaluation_runs "
            "BEGIN SELECT RAISE(ABORT, 'Evaluations are immutable'); END"
        )
        connection.execute(
            "CREATE TRIGGER evaluation_decisions_no_update BEFORE UPDATE ON evaluation_decisions "
            "BEGIN SELECT RAISE(ABORT, 'Evaluation decisions are append-only'); END"
        )
        connection.execute(
            "CREATE TRIGGER evaluation_decisions_no_delete BEFORE DELETE ON evaluation_decisions "
            "BEGIN SELECT RAISE(ABORT, 'Evaluation decisions are append-only'); END"
        )
        updated = connection.execute(
            "UPDATE project_metadata SET schema_version = ? "
            "WHERE project_id = ? AND schema_version = ?",
            (_EVALUATION_SCHEMA_VERSION, project_id, _TRAINING_RUN_SCHEMA_VERSION),
        ).rowcount
        if updated != 1:
            raise EvaluationRunError(f"Invalid project metadata: {database}")
        connection.execute(f"PRAGMA user_version = {_EVALUATION_SCHEMA_VERSION}")
    elif version != _EVALUATION_SCHEMA_VERSION:
        raise EvaluationRunError("Evaluations require project schema 13 or 14")


def _target_satisfied(
    thresholds: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    criteria: Mapping[str, Any] | None = None,
) -> bool:
    selections = thresholds.get("per_class", thresholds.get("selections", ()))
    if not isinstance(selections, (list, tuple)):
        return True
    flags: list[bool] = []
    for selection in selections:
        if not isinstance(selection, Mapping):
            raise ValueError("thresholds per_class entries must be objects")
        metric = selection.get("metrics")
        target = selection.get("target")
        policy = str(selection.get("policy", ""))
        metric_name = "recall" if policy in {"min_recall", "minimum_recall", "recall"} else "fpr"
        if target is not None and isinstance(metric, Mapping):
            actual = metric.get(metric_name)
            if isinstance(actual, (int, float)):
                flags.append(float(actual) >= float(target) if metric_name == "recall" else float(actual) <= float(target))
                continue
        if "target_satisfied" in selection:
            value = selection["target_satisfied"]
            if not isinstance(value, bool):
                raise ValueError("target_satisfied must be boolean")
            flags.append(value)
            continue
        if target is not None and isinstance(metrics, Mapping):
            class_name = selection.get("class_name")
            per_class = metrics.get("per_class")
            if isinstance(per_class, (list, tuple)):
                metric_rows = (
                    row for row in per_class
                    if isinstance(row, Mapping) and row.get("class_name") == class_name
                )
                row = next(metric_rows, None)
                if row is not None and isinstance(row.get(metric_name), (int, float)):
                    actual = float(row[metric_name])
                    flags.append(actual >= float(target) if metric_name == "recall" else actual <= float(target))
    if flags:
        return all(flags)
    explicit = thresholds.get("target_satisfied")
    if isinstance(explicit, bool):
        return explicit
    target = None if criteria is None else criteria.get("minimum_recall_target")
    if target is None and criteria is not None:
        target = criteria.get("min_recall_target")
    per_class = None if metrics is None else metrics.get("per_class")
    if isinstance(target, (int, float)) and isinstance(per_class, (list, tuple)):
        recalls = [
            float(row["recall"])
            for row in per_class
            if isinstance(row, Mapping) and isinstance(row.get("recall"), (int, float))
        ]
        if recalls:
            return all(recall >= float(target) for recall in recalls)
    return True


def _allows_unmet(criteria: Mapping[str, Any]) -> bool:
    return any(
        criteria.get(name) is True
        for name in (
            "allow_unmet_targets",
            "allow_unmet_target",
            "allow_unmet_minimum_recall",
            "allow_unmet_recall",
            "allow_unmet",
            "approval_override",
            "approval_exception",
            "override_minimum_recall",
        )
    )


def _json_object(value: object, name: str) -> dict[str, Any]:
    result = _to_json(value, name)
    if not isinstance(result, dict):
        raise ValueError(f"{name} must be a JSON object")
    return result


def _to_json(value: object, name: str) -> Any:
    if isinstance(value, Enum):
        return _to_json(value.value, name)
    if is_dataclass(value):
        return {field.name: _to_json(getattr(value, field.name), name) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _to_json(item, name) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_json(item, name) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"{name} must contain JSON-compatible values")


def _encode(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _decode_object(serialized: str, name: str) -> dict[str, Any]:
    value = json.loads(serialized)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _coerce_status(value: EvaluationStatus | str) -> str:
    try:
        status = value.value if isinstance(value, EvaluationStatus) else str(value).lower()
        if status not in _DECISION_STATUSES:
            raise ValueError
    except (AttributeError, TypeError, ValueError) as error:
        raise EvaluationRunError(f"unsupported Evaluation status: {value!r}") from error
    return status


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _read_only(database: Path) -> sqlite3.Connection:
    try:
        return sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise EvaluationRunError(f"Cannot open project database: {database}") from error


# Short aliases keep the project-service seam discoverable.
create_evaluation_run = create_evaluation
load_evaluation_run = load_evaluation
append_evaluation_decision = record_evaluation_decision
save_evaluation = create_evaluation
get_evaluation = load_evaluation
list_evaluation_decisions = load_evaluation_decisions
append_decision = record_evaluation_decision
record_decision = record_evaluation_decision
create_evaluation_record = create_evaluation
load_evaluation_record = load_evaluation
EvaluationRun = EvaluationRecord


__all__ = [
    "EvaluationDecision",
    "EvaluationRecord",
    "EvaluationRun",
    "EvaluationRunError",
    "EvaluationStatus",
    "append_evaluation_decision",
    "append_decision",
    "create_evaluation",
    "create_evaluation_run",
    "create_evaluation_record",
    "list_evaluations",
    "list_evaluation_decisions",
    "load_evaluation",
    "load_evaluation_decisions",
    "load_evaluation_run",
    "load_evaluation_record",
    "get_evaluation",
    "record_evaluation_decision",
    "record_decision",
    "save_evaluation",
]
