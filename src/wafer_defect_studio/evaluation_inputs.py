"""Read-only Evaluation inventory for the Evaluate workspace."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QPushButton, QWidget

from .evaluation_run import load_evaluation
from .project import _EVALUATION_SCHEMA_VERSION, open_project
from .training_run import TrainingRunError, load_training_run


@dataclass(frozen=True)
class EvaluationInputOption:
    identifier: str
    label: str
    available: bool = True


@dataclass(frozen=True)
class TrainingRunOption:
    identifier: str
    label: str
    available: bool = True


@dataclass(frozen=True)
class EvaluationInputInventory:
    options: tuple[EvaluationInputOption, ...]
    status: str
    training_runs: tuple[TrainingRunOption, ...] = ()


def load_evaluation_input_inventory(project_path: str | Path) -> EvaluationInputInventory:
    """Load persisted Evaluation rows without changing the project."""

    info = open_project(project_path)
    if info.schema_version < _EVALUATION_SCHEMA_VERSION:
        return EvaluationInputInventory(
            (), "No Evaluations available (project schema 14 is required)."
        )
    database = info.path / "project.sqlite"
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        ids = tuple(
            row[0]
            for row in connection.execute(
                "SELECT evaluation_id FROM evaluation_runs ORDER BY evaluation_id"
            )
        )
    finally:
        connection.close()

    options: list[EvaluationInputOption] = []
    for evaluation_id in ids:
        try:
            evaluation = load_evaluation(info.path, evaluation_id)
        except Exception:
            options.append(
                EvaluationInputOption(
                    evaluation_id,
                    f"Unavailable Evaluation: {evaluation_id}",
                    False,
                )
            )
        else:
            options.append(
                EvaluationInputOption(
                    evaluation_id,
                    f"Evaluation {evaluation.evaluation_id} — Training Run {evaluation.training_run_id}",
                )
            )
    options.sort(key=lambda option: (not option.available, option.identifier))
    run_options: list[TrainingRunOption] = []
    if info.schema_version >= 12:
        database = info.path / "project.sqlite"
        connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            run_ids = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT run_id FROM training_runs ORDER BY created_at, run_id"
                )
            )
        finally:
            connection.close()
        for run_id in run_ids:
            try:
                run = load_training_run(info.path, run_id)
            except TrainingRunError:
                run_options.append(TrainingRunOption(run_id, f"Unavailable Training Run: {run_id}", False))
            else:
                available = run.status == "completed" and run.artifact_path is not None
                label = f"Training Run {run.run_id} ({run.status})"
                if not available:
                    label = f"Unavailable {label}"
                run_options.append(TrainingRunOption(run_id, label, available))
    if not options:
        status = "No Evaluations available for this project."
    elif any(not option.available for option in options):
        status = "Some Evaluations are unavailable."
    else:
        status = f"{len(options)} Evaluation(s) available."
    return EvaluationInputInventory(tuple(options), status, tuple(run_options))


class EvaluationInputControls(QWidget):
    """Display immutable Evaluation choices without editing them."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.combo = QComboBox(self)
        self.combo.setObjectName("evaluationInventoryComboBox")
        self.combo.setAccessibleName("Evaluation inventory")
        self.training_run_combo = QComboBox(self)
        self.training_run_combo.setObjectName("evaluationTrainingRunComboBox")
        self.training_run_combo.setAccessibleName("Training Run for Evaluation")
        self.start_button = QPushButton("Evaluate Test Split", self)
        self.start_button.setObjectName("startEvaluationButton")
        self.start_button.setEnabled(False)
        self.status_label = QLabel("No Evaluations available for this project.", self)
        self.status_label.setObjectName("evaluationInventoryStatusLabel")
        self.status_label.setWordWrap(True)
        layout = QFormLayout(self)
        layout.addRow("Evaluation", self.combo)
        layout.addRow("Completed Training Run", self.training_run_combo)
        layout.addRow(self.start_button)
        layout.addRow(self.status_label)

    def set_inventory(self, inventory: EvaluationInputInventory) -> None:
        self.combo.clear()
        for option in inventory.options:
            self.combo.addItem(option.label, option.identifier)
            item = self.combo.model().item(self.combo.count() - 1)
            if item is not None:
                item.setEnabled(option.available)
        self.training_run_combo.clear()
        for option in inventory.training_runs:
            self.training_run_combo.addItem(option.label, option.identifier)
            item = self.training_run_combo.model().item(self.training_run_combo.count() - 1)
            if item is not None:
                item.setEnabled(option.available)
        self.start_button.setEnabled(any(option.available for option in inventory.training_runs))
        self.status_label.setText(inventory.status)


__all__ = [
    "EvaluationInputControls",
    "EvaluationInputInventory",
    "EvaluationInputOption",
    "TrainingRunOption",
    "load_evaluation_input_inventory",
]
