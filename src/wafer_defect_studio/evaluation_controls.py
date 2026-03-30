"""Small, non-blocking Widgets for Grid Evaluation review and approval."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .evaluation_run import EvaluationDecision, EvaluationRecord


DecisionService = Callable[..., EvaluationDecision]


class _DecisionSignals(QObject):
    completed = Signal(object)
    failed = Signal(str)


class _DecisionTask(QRunnable):
    """Call an injected immutable decision service away from the GUI thread."""

    def __init__(
        self,
        service: DecisionService,
        evaluation_id: str,
        status: str,
        actor: str,
        criteria: Mapping[str, Any],
        notes: str,
    ) -> None:
        super().__init__()
        self.service = service
        self.evaluation_id = evaluation_id
        self.status = status
        self.actor = actor
        self.criteria = dict(criteria)
        self.notes = notes
        self.signals = _DecisionSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = _call_decision_service(
                self.service,
                self.evaluation_id,
                self.status,
                self.actor,
                self.criteria,
                self.notes,
            )
            self.signals.completed.emit(result)
        except Exception as error:  # Display the service error without blocking the UI.
            self.signals.failed.emit(str(error))


class EvaluationControls(QWidget):
    """Render immutable Evaluation data and append decisions through a service seam.

    The controls receive value objects only.  They never open a project or write
    SQLite; persistence remains in the injected evaluation service.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._evaluation: EvaluationRecord | None = None
        self._decisions: list[EvaluationDecision] = []
        self._decision_service: DecisionService | None = None
        self._tasks: set[_DecisionTask] = set()

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("evaluationTabs")
        self.grid_tab = QWidget(self.tabs)
        self.grid_tab.setObjectName("gridEvaluationTab")
        self.wafer_tab = QWidget(self.tabs)
        self.wafer_tab.setObjectName("waferEvaluationTab")
        self.tabs.addTab(self.grid_tab, "Grid Evaluation")
        self.tabs.addTab(self.wafer_tab, "Wafer Evaluation")
        self._build_grid_tab()
        self._build_wafer_tab()

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

    def _build_grid_tab(self) -> None:
        self.evaluation_id_label = QLabel("Evaluation: —", self.grid_tab)
        self.evaluation_id_label.setObjectName("evaluationIdLabel")
        self.macro_f1_label = QLabel("Macro F1: —", self.grid_tab)
        self.macro_f1_label.setObjectName("evaluationMacroF1Label")
        self.target_status_label = QLabel("Minimum-recall target: —", self.grid_tab)
        self.target_status_label.setObjectName("evaluationTargetStatusLabel")
        self.target_status_label.setWordWrap(True)

        self.metrics_table = QTableWidget(0, 10, self.grid_tab)
        self.metrics_table.setObjectName("evaluationMetricsTable")
        self.metrics_table.setHorizontalHeaderLabels(
            (
                "Class",
                "Precision",
                "Recall",
                "F1",
                "Support",
                "FPR",
                "Threshold",
                "Policy",
                "Target",
                "Target State",
            )
        )
        self.metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.history_table = QTableWidget(0, 6, self.grid_tab)
        self.history_table.setObjectName("evaluationDecisionHistoryTable")
        self.history_table.setHorizontalHeaderLabels(
            ("Evaluation", "Decided At", "Status", "Actor", "Target", "Notes")
        )
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        approval_group = QGroupBox("Approval decision", self.grid_tab)
        approval_form = QFormLayout(approval_group)
        self.actor_edit = QLineEdit(approval_group)
        self.actor_edit.setObjectName("evaluationApprovalActorEdit")
        self.actor_edit.setPlaceholderText("Required")
        self.criteria_edit = QLineEdit(approval_group)
        self.criteria_edit.setObjectName("evaluationApprovalCriteriaEdit")
        self.criteria_edit.setPlaceholderText("Optional override, e.g. allow_unmet_targets=true")
        self.notes_edit = QPlainTextEdit(approval_group)
        self.notes_edit.setObjectName("evaluationApprovalNotesEdit")
        self.notes_edit.setPlaceholderText("Required for approval")
        self.notes_edit.setMaximumHeight(72)
        self.approve_button = QPushButton("Approve Evaluation", approval_group)
        self.approve_button.setObjectName("approveEvaluationButton")
        self.approve_button.setEnabled(False)
        self.validate_button = QPushButton("Validate Evaluation", approval_group)
        self.validate_button.setObjectName("validateEvaluationButton")
        self.validate_button.setEnabled(False)
        self.approval_status_label = QLabel("Approval requires an explicit actor and notes.", approval_group)
        self.approval_status_label.setObjectName("evaluationApprovalStatusLabel")
        self.approval_status_label.setWordWrap(True)
        approval_form.addRow("Actor", self.actor_edit)
        approval_form.addRow("Criteria", self.criteria_edit)
        approval_form.addRow("Notes", self.notes_edit)
        approval_form.addRow(self.validate_button)
        approval_form.addRow(self.approve_button)
        approval_form.addRow(self.approval_status_label)
        self.actor_edit.textChanged.connect(self._refresh_approval_enabled)
        self.notes_edit.textChanged.connect(self._refresh_approval_enabled)
        self.validate_button.clicked.connect(self.validate_evaluation)
        self.approve_button.clicked.connect(self.approve_evaluation)

        layout = QVBoxLayout(self.grid_tab)
        layout.addWidget(self.evaluation_id_label)
        layout.addWidget(self.macro_f1_label)
        layout.addWidget(self.target_status_label)
        layout.addWidget(QLabel("Per-class metrics and thresholds", self.grid_tab))
        layout.addWidget(self.metrics_table)
        layout.addWidget(QLabel("Append-only decision history", self.grid_tab))
        layout.addWidget(self.history_table)
        layout.addWidget(approval_group)

    def _build_wafer_tab(self) -> None:
        self.wafer_placeholder_label = QLabel(
            "Wafer Evaluation — basic structure only. Wafer-level quality metrics are not available in this release.",
            self.wafer_tab,
        )
        self.wafer_placeholder_label.setObjectName("waferEvaluationPlaceholderLabel")
        self.wafer_placeholder_label.setWordWrap(True)
        layout = QVBoxLayout(self.wafer_tab)
        layout.addWidget(self.wafer_placeholder_label)
        layout.addStretch(1)

    def configure(
        self,
        evaluation: EvaluationRecord | Sequence[EvaluationRecord],
        decisions: Sequence[EvaluationDecision] | Mapping[str, Sequence[EvaluationDecision]] = (),
        *,
        decision_service: DecisionService | None = None,
    ) -> None:
        """Bind an immutable Evaluation and existing append-only decision history."""

        if isinstance(evaluation, EvaluationRecord):
            current = evaluation
        else:
            values = tuple(evaluation)
            if not values or not all(isinstance(item, EvaluationRecord) for item in values):
                raise TypeError("evaluation must be an EvaluationRecord or non-empty sequence")
            current = values[-1]
        self._evaluation = current
        self._decisions = list(_decision_values(decisions, current.evaluation_id))
        self._decision_service = decision_service
        self._render_evaluation()
        self._refresh_approval_enabled()
        self.approval_status_label.setText("Approval requires an explicit actor and notes.")

    def set_decisions(self, decisions: Sequence[EvaluationDecision]) -> None:
        """Refresh history without changing the immutable Evaluation value."""

        self._decisions = list(decisions)
        self._render_history()

    def clear(self) -> None:
        """Remove the selected Evaluation detail without touching persistence."""

        self._evaluation = None
        self._decisions = []
        self._decision_service = None
        self.evaluation_id_label.setText("Evaluation: —")
        self.macro_f1_label.setText("Macro F1: —")
        self.target_status_label.setText("Minimum-recall target: —")
        self.metrics_table.setRowCount(0)
        self.history_table.setRowCount(0)
        self._refresh_approval_enabled()

    def validate_evaluation(self) -> None:
        """Request one append-only Validated decision through the injected service."""

        self._submit_decision("validated")

    def approve_evaluation(self) -> None:
        """Request one append-only Approved decision through the injected service."""

        self._submit_decision("approved")

    def _submit_decision(self, status: str) -> None:
        """Run a decision append without opening SQLite or blocking the GUI."""

        evaluation = self._evaluation
        service = self._decision_service
        if evaluation is None or service is None:
            self.approval_status_label.setText("Decision service is not configured.")
            return
        actor = self.actor_edit.text().strip()
        notes = self.notes_edit.toPlainText().strip()
        if not actor or not notes:
            self.approval_status_label.setText("Decision requires a non-empty actor and notes.")
            return
        criteria = _criteria_from_text(self.criteria_edit.text())
        if not _target_satisfied(evaluation) and not criteria.get("allow_unmet_targets", False):
            self.approval_status_label.setText(
                f"{status.title()} requires an explicit allow_unmet_targets=true criterion while the 95% recall target is unmet."
            )
            return
        self.validate_button.setEnabled(False)
        self.approve_button.setEnabled(False)
        self.approval_status_label.setText(f"Recording {status.title()} decision…")
        task = _DecisionTask(service, evaluation.evaluation_id, status, actor, criteria, notes)
        self._tasks.add(task)
        task.signals.completed.connect(lambda result, item=task: self._decision_recorded(item, result))
        task.signals.failed.connect(lambda message, item=task: self._decision_failed(item, message))
        QThreadPool.globalInstance().start(task)

    def _decision_recorded(self, task: _DecisionTask, result: object) -> None:
        self._tasks.discard(task)
        if not isinstance(result, EvaluationDecision):
            self._decision_failed(task, "decision service returned an invalid record")
            return
        self._decisions.append(result)
        self._render_history()
        self.approval_status_label.setText(
            f"Recorded {result.status.title()} decision by {result.actor} at {result.decided_at}."
        )
        self._refresh_approval_enabled()

    def _decision_failed(self, task: _DecisionTask, message: str) -> None:
        self._tasks.discard(task)
        self.approval_status_label.setText(f"Approval not recorded: {message or 'service failed.'}")
        self._refresh_approval_enabled()

    def _render_evaluation(self) -> None:
        evaluation = self._evaluation
        if evaluation is None:
            return
        self.evaluation_id_label.setText(
            f"Evaluation: {evaluation.evaluation_id} | Training Run: {evaluation.training_run_id}"
        )
        metrics = _mapping(evaluation.metrics)
        macro = metrics.get("macro_f1")
        self.macro_f1_label.setText(
            "Macro F1: —" if not isinstance(macro, (int, float)) else f"Macro F1: {float(macro):.4f}"
        )
        target = _target_value(evaluation)
        state = "SATISFIED" if _target_satisfied(evaluation) else "UNMET"
        self.target_status_label.setText(
            f"Minimum-recall target: {target:.0%} — {state} (optimization target, not a guarantee)"
        )
        rows = _mapping_rows(metrics.get("per_class"))
        thresholds = _threshold_rows(evaluation.thresholds)
        self.metrics_table.setRowCount(len(rows))
        for row_index, metric in enumerate(rows):
            name = str(metric.get("class_name", ""))
            threshold = thresholds.get(name, {})
            values = (
                name,
                _decimal(metric.get("precision")),
                _decimal(metric.get("recall")),
                _decimal(metric.get("f1")),
                str(metric.get("support", "—")),
                _decimal(metric.get("fpr")),
                _decimal(threshold.get("threshold")),
                str(threshold.get("policy", "—")),
                _percent(threshold.get("target")),
                "SATISFIED" if _row_target_satisfied(metric, threshold) else "UNMET",
            )
            for column, value in enumerate(values):
                self.metrics_table.setItem(row_index, column, QTableWidgetItem(value))
        self._render_history()

    def _render_history(self) -> None:
        self.history_table.setRowCount(len(self._decisions))
        for row_index, decision in enumerate(self._decisions):
            values = (
                decision.evaluation_id,
                decision.decided_at,
                _status_text(decision.status),
                decision.actor,
                "SATISFIED" if decision.target_satisfied else "UNMET",
                decision.notes,
            )
            for column, value in enumerate(values):
                self.history_table.setItem(row_index, column, QTableWidgetItem(value))

    def _refresh_approval_enabled(self) -> None:
        enabled = (
            self._evaluation is not None
            and self._decision_service is not None
            and bool(self.actor_edit.text().strip())
            and bool(self.notes_edit.toPlainText().strip())
        )
        self.validate_button.setEnabled(enabled)
        self.approve_button.setEnabled(enabled)

    def closeEvent(self, event) -> None:  # pragma: no cover - exercised by Qt teardown
        self._tasks.clear()
        super().closeEvent(event)


def _call_decision_service(
    service: DecisionService,
    evaluation_id: str,
    status: str,
    actor: str,
    criteria: Mapping[str, Any],
    notes: str,
) -> EvaluationDecision:
    method = getattr(service, "append_decision", None)
    if not callable(method):
        method = getattr(service, "record_decision", None)
    if not callable(method):
        method = getattr(service, "append_evaluation_decision", None)
    if not callable(method):
        method = getattr(service, "record_evaluation_decision", None)
    if not callable(method):
        method = service
    result = method(
        evaluation_id,
        status,
        actor,
        criteria=dict(criteria),
        notes=notes,
    )
    if not isinstance(result, EvaluationDecision):
        raise TypeError("decision service must return an EvaluationDecision")
    return result


def _decision_values(
    decisions: Sequence[EvaluationDecision] | Mapping[str, Sequence[EvaluationDecision]],
    evaluation_id: str,
) -> tuple[EvaluationDecision, ...]:
    if isinstance(decisions, Mapping):
        # Keep prior re-evaluation decisions visible; mappings are commonly
        # keyed by Evaluation id and preserve insertion order in Python.
        values = tuple(item for group in decisions.values() for item in group)
    else:
        values = decisions
    return tuple(item for item in values if isinstance(item, EvaluationDecision))


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        result = asdict(value)
        return result if isinstance(result, Mapping) else {}
    return {}


def _mapping_rows(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (tuple, list)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _threshold_rows(value: object) -> dict[str, Mapping[str, Any]]:
    mapping = _mapping(value)
    rows = mapping.get("per_class", mapping.get("selections", ()))
    return {
        str(row.get("class_name")): row
        for row in _mapping_rows(rows)
        if row.get("class_name") is not None
    }


def _target_value(evaluation: EvaluationRecord) -> float:
    thresholds = _mapping(evaluation.thresholds)
    criteria = _mapping(evaluation.criteria)
    value = thresholds.get("min_recall_target", thresholds.get("minimum_recall_target"))
    if value is None:
        value = criteria.get("min_recall_target", criteria.get("minimum_recall_target", 0.95))
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.95


def _target_satisfied(evaluation: EvaluationRecord) -> bool:
    if not bool(evaluation.target_satisfied):
        return False
    metrics = {str(row.get("class_name")): row for row in _mapping_rows(_mapping(evaluation.metrics).get("per_class"))}
    for name, threshold in _threshold_rows(evaluation.thresholds).items():
        if threshold.get("target_satisfied") is False:
            return False
        target = threshold.get("target")
        recall = metrics.get(name, {}).get("recall")
        if target is not None:
            if not isinstance(recall, (int, float)) or float(recall) < float(target):
                return False
    return True


def _row_target_satisfied(metric: Mapping[str, Any], threshold: Mapping[str, Any]) -> bool:
    if threshold.get("target_satisfied") is False:
        return False
    target = threshold.get("target")
    recall = metric.get("recall")
    if target is not None:
        return isinstance(recall, (int, float)) and float(recall) >= float(target)
    return bool(threshold.get("target_satisfied", True))


def _criteria_from_text(value: str) -> dict[str, Any]:
    text = value.strip()
    if not text:
        return {}
    criteria: dict[str, Any] = {}
    for part in text.split(","):
        if "=" not in part:
            continue
        key, raw = (item.strip() for item in part.split("=", 1))
        if key:
            criteria[key] = raw.lower() == "true" if raw.lower() in {"true", "false"} else raw
    return criteria


def _status_text(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw).title()


def _decimal(value: object) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{float(value):.4f}"


def _percent(value: object) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{float(value):.0%}"


__all__ = ["DecisionService", "EvaluationControls"]
