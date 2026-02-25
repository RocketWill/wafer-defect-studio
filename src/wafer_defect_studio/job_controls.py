"""Nonmodal, value-only Widgets for inspecting and acting on background jobs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .job_actions import RecoveryAction, available_recovery_actions
from .job_state import JobSnapshot, JobStatus


JobsActionCallback = Callable[[JobSnapshot, RecoveryAction | str], Any]
CANCEL_ACTION = "cancel"


class JobsControls(QWidget):
    """Render immutable snapshots and route explicit actions to a callback."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._jobs: tuple[JobSnapshot, ...] = ()
        self._action_callback: JobsActionCallback | None = None
        self._selected_job_id: str | None = None

        self.jobs_table = QTableWidget(0, 8, self)
        self.jobs_table.setObjectName("jobsTable")
        self.jobs_table.setHorizontalHeaderLabels(
            ("Job id", "Kind", "Status", "Phase", "Progress", "ETA", "Message", "Attempt")
        )
        self.jobs_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.jobs_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.jobs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.jobs_table.currentCellChanged.connect(self._selection_changed)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setObjectName("cancelJobButton")
        self.restart_button = QPushButton("Restart", self)
        self.restart_button.setObjectName("restartJobButton")
        self.retry_button = QPushButton("Retry", self)
        self.retry_button.setObjectName("retryJobButton")
        self.clone_button = QPushButton("Clone Adjusted", self)
        self.clone_button.setObjectName("cloneJobButton")
        self.inspect_button = QPushButton("Inspect Log", self)
        self.inspect_button.setObjectName("inspectJobLogButton")
        actions = QHBoxLayout()
        for button in (
            self.cancel_button,
            self.restart_button,
            self.retry_button,
            self.clone_button,
            self.inspect_button,
        ):
            actions.addWidget(button)

        self.status_label = QLabel("No jobs configured.", self)
        self.status_label.setObjectName("jobsStatusLabel")
        self.status_label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.jobs_table)
        layout.addLayout(actions)
        layout.addWidget(self.status_label)

        self.cancel_button.clicked.connect(lambda: self._submit(CANCEL_ACTION))
        self.restart_button.clicked.connect(lambda: self._submit(RecoveryAction.RESTART))
        self.retry_button.clicked.connect(lambda: self._submit(RecoveryAction.RETRY))
        self.clone_button.clicked.connect(lambda: self._submit(RecoveryAction.CLONE_ADJUSTED))
        self.inspect_button.clicked.connect(lambda: self._submit(RecoveryAction.INSPECT_LOG))
        self._set_action_enabled(None)

    @property
    def jobs(self) -> tuple[JobSnapshot, ...]:
        """Return the currently displayed immutable values."""

        return self._jobs

    def configure(
        self,
        jobs: Iterable[JobSnapshot],
        action_callback: JobsActionCallback | None = None,
    ) -> None:
        """Bind values and an injected action callback; no persistence is performed."""

        if action_callback is not None and not callable(action_callback):
            raise TypeError("action_callback must be callable or None")
        values = tuple(jobs)
        if any(not isinstance(job, JobSnapshot) for job in values):
            raise TypeError("jobs must contain only JobSnapshot values")
        self._jobs = tuple(sorted(values, key=lambda job: (job.job_id, job.attempt)))
        self._action_callback = action_callback
        self._selected_job_id = None
        self._render()
        self.status_label.setText(f"{len(self._jobs)} job(s) ready.")

    def _render(self) -> None:
        self.jobs_table.blockSignals(True)
        try:
            self.jobs_table.setRowCount(len(self._jobs))
            for row, job in enumerate(self._jobs):
                values = (
                    job.job_id,
                    job.kind.value,
                    job.status.value.title(),
                    job.phase,
                    _format_progress(job),
                    _format_eta(job.eta_seconds),
                    job.message,
                    str(job.attempt),
                )
                for column, value in enumerate(values):
                    cell = QTableWidgetItem(value)
                    if column == 0:
                        cell.setData(Qt.ItemDataRole.UserRole, job.job_id)
                    self.jobs_table.setItem(row, column, cell)
            target_row = 0 if self._jobs else -1
            self.jobs_table.setCurrentCell(target_row, 0)
        finally:
            self.jobs_table.blockSignals(False)
        if self._jobs:
            self._selection_changed(self.jobs_table.currentRow(), 0, -1, -1)
        else:
            self._selected_job_id = None
            self._set_action_enabled(None)

    def _selection_changed(
        self,
        row: int,
        _column: int,
        _previous_row: int,
        _previous_column: int,
    ) -> None:
        if row < 0 or row >= self.jobs_table.rowCount():
            self._selected_job_id = None
            self._set_action_enabled(None)
            return
        cell = self.jobs_table.item(row, 0)
        job_id = cell.data(Qt.ItemDataRole.UserRole) if cell is not None else None
        if not isinstance(job_id, str):
            self._selected_job_id = None
            self._set_action_enabled(None)
            return
        self._selected_job_id = job_id
        self._set_action_enabled(self._selected_job())

    def _selected_job(self) -> JobSnapshot | None:
        if self._selected_job_id is None:
            return None
        return next((job for job in self._jobs if job.job_id == self._selected_job_id), None)

    def _set_action_enabled(self, job: JobSnapshot | None) -> None:
        ready = job is not None and self._action_callback is not None
        is_active = job is not None and job.status in (JobStatus.RUNNING, JobStatus.CANCELLING)
        actions = set(available_recovery_actions(job)) if job is not None else set()
        self.cancel_button.setEnabled(bool(ready and is_active))
        self.restart_button.setEnabled(bool(ready and RecoveryAction.RESTART in actions))
        self.retry_button.setEnabled(bool(ready and RecoveryAction.RETRY in actions))
        self.clone_button.setEnabled(bool(ready and RecoveryAction.CLONE_ADJUSTED in actions))
        self.inspect_button.setEnabled(bool(ready and RecoveryAction.INSPECT_LOG in actions))

    def _submit(self, action: RecoveryAction | str) -> None:
        job = self._selected_job()
        callback = self._action_callback
        if job is None or callback is None:
            self.status_label.setText("Select a job and configure an action callback first.")
            return
        try:
            result = callback(job, action)
        except Exception as error:  # callback owns process/store policy
            self.status_label.setText(f"Action failed: {error}")
            return
        if result is False:
            self.status_label.setText(f"Action failed for {job.job_id}: callback rejected {action}.")
            return
        label = "Cancel" if action == CANCEL_ACTION else action.value.replace("_", " ").title()
        self.status_label.setText(f"{label} requested for {job.job_id}.")


def _format_progress(job: JobSnapshot) -> str:
    if job.total:
        return f"{job.completed}/{job.total}"
    return str(job.completed)


def _format_eta(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}s"


JobsPanel = JobsControls


__all__ = ["CANCEL_ACTION", "JobsActionCallback", "JobsControls", "JobsPanel"]
