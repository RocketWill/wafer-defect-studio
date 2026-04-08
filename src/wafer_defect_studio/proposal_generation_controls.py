"""Non-blocking control for project-owned Defect Proposal generation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from .proposal_generation import DefectProposal


ProposalGenerationCallback = Callable[[], Iterable[DefectProposal]]


class _GenerationSignals(QObject):
    completed = Signal(object)
    failed = Signal(str)


class _GenerationTask(QRunnable):
    def __init__(self, callback: ProposalGenerationCallback) -> None:
        super().__init__()
        self.callback = callback
        self.signals = _GenerationSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = tuple(self.callback())
            if any(not isinstance(item, DefectProposal) for item in result):
                raise TypeError("proposal generation callback returned invalid values")
            self.signals.completed.emit(result)
        except Exception as error:  # The project service owns persistence/error policy.
            self.signals.failed.emit(str(error))


class ProposalGenerationControls(QWidget):
    """Request one asynchronous generation pass for the selected Detection Run."""

    generated = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.generate_button = QPushButton("Generate Proposals", self)
        self.generate_button.setObjectName("generateProposalsButton")
        self.status_label = QLabel("Select a completed Detection Run first.", self)
        self.status_label.setObjectName("proposalGenerationStatusLabel")
        self.status_label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.status_label)

        self._callback: ProposalGenerationCallback | None = None
        self._tasks: set[_GenerationTask] = set()
        self.generate_button.clicked.connect(self.generate)
        self._refresh_enabled()

    def configure(
        self,
        callback: ProposalGenerationCallback | None,
        *,
        status: str | None = None,
    ) -> None:
        """Bind the current project/run generation seam."""

        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or None")
        self._callback = callback
        if status is not None:
            self.status_label.setText(status)
        self._refresh_enabled()

    def generate(self) -> None:
        callback = self._callback
        if callback is None:
            self.status_label.setText("Select a completed Detection Run first.")
            self._refresh_enabled()
            return
        self.generate_button.setEnabled(False)
        self.status_label.setText("Generating Proposals…")
        task = _GenerationTask(callback)
        self._tasks.add(task)
        task.signals.completed.connect(lambda result, item=task: self._completed(item, result))
        task.signals.failed.connect(lambda message, item=task: self._failed(item, message))
        QThreadPool.globalInstance().start(task)

    def _completed(self, task: _GenerationTask, result: object) -> None:
        self._tasks.discard(task)
        values = tuple(result) if isinstance(result, tuple) else ()
        self.status_label.setText(f"Generated {len(values)} proposal(s).")
        self._refresh_enabled()
        self.generated.emit(values)

    def _failed(self, task: _GenerationTask, message: str) -> None:
        self._tasks.discard(task)
        self.status_label.setText(f"Proposal generation failed: {message}")
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        self.generate_button.setEnabled(self._callback is not None and not self._tasks)

    def closeEvent(self, event) -> None:  # pragma: no cover - Qt teardown
        self._tasks.clear()
        super().closeEvent(event)


__all__ = ["ProposalGenerationCallback", "ProposalGenerationControls"]
