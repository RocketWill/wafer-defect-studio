"""Read-only Evaluation inventory for the Evaluate workspace."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QWidget

from .evaluation_run import load_evaluation
from .project import _EVALUATION_SCHEMA_VERSION, open_project


@dataclass(frozen=True)
class EvaluationInputOption:
    identifier: str
    label: str
    available: bool = True


@dataclass(frozen=True)
class EvaluationInputInventory:
    options: tuple[EvaluationInputOption, ...]
    status: str


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
    if not options:
        status = "No Evaluations available for this project."
    elif any(not option.available for option in options):
        status = "Some Evaluations are unavailable."
    else:
        status = f"{len(options)} Evaluation(s) available."
    return EvaluationInputInventory(tuple(options), status)


class EvaluationInputControls(QWidget):
    """Display immutable Evaluation choices without editing them."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.combo = QComboBox(self)
        self.combo.setObjectName("evaluationInventoryComboBox")
        self.combo.setAccessibleName("Evaluation inventory")
        self.status_label = QLabel("No Evaluations available for this project.", self)
        self.status_label.setObjectName("evaluationInventoryStatusLabel")
        self.status_label.setWordWrap(True)
        layout = QFormLayout(self)
        layout.addRow("Evaluation", self.combo)
        layout.addRow(self.status_label)

    def set_inventory(self, inventory: EvaluationInputInventory) -> None:
        self.combo.clear()
        for option in inventory.options:
            self.combo.addItem(option.label, option.identifier)
            item = self.combo.model().item(self.combo.count() - 1)
            if item is not None:
                item.setEnabled(option.available)
        self.status_label.setText(inventory.status)


__all__ = [
    "EvaluationInputControls",
    "EvaluationInputInventory",
    "EvaluationInputOption",
    "load_evaluation_input_inventory",
]
