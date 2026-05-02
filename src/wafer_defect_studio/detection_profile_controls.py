"""Project-bound controls for creating immutable Detection Profiles."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from .evaluation_run import load_evaluation, load_evaluation_decisions
from .project import _EVALUATION_SCHEMA_VERSION, open_project
from .training_run import load_training_run


@dataclass(frozen=True, slots=True)
class DetectionProfileEvaluationOption:
    """One Evaluation choice for Detection Profile creation."""

    identifier: str
    label: str
    available: bool = True
    required_window_size: tuple[int, int] | None = None
    required_stride: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class DetectionProfileEvaluationInventory:
    """Read-only Approved Evaluation choices for one project."""

    options: tuple[DetectionProfileEvaluationOption, ...]
    status: str


@dataclass(frozen=True, slots=True)
class DetectionProfileDraft:
    """The editable values needed to create one Detection Profile."""

    evaluation_id: str
    window_size: tuple[int, int]
    stride: tuple[int, int]
    reflect_padding: bool


ProfileCreateCallback = Callable[[DetectionProfileDraft], Any]


def load_detection_profile_evaluation_inventory(
    project_path: str | Path,
) -> DetectionProfileEvaluationInventory:
    """Load only Approved Evaluations without changing the project."""

    info = open_project(project_path)
    if info.schema_version < _EVALUATION_SCHEMA_VERSION:
        return DetectionProfileEvaluationInventory(
            (), "No Approved Evaluations available (project schema 14 is required)."
        )
    database = info.path / "project.sqlite"
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        evaluation_ids = tuple(
            row[0]
            for row in connection.execute(
                "SELECT evaluation_id FROM evaluation_runs ORDER BY evaluation_id"
            )
        )
    finally:
        connection.close()

    options: list[DetectionProfileEvaluationOption] = []
    for evaluation_id in evaluation_ids:
        try:
            evaluation = load_evaluation(info.path, evaluation_id)
            decisions = load_evaluation_decisions(info.path, evaluation_id)
            if not decisions or decisions[-1].status != "approved":
                raise ValueError("Evaluation is not Approved")
        except Exception as error:
            options.append(
                DetectionProfileEvaluationOption(
                    evaluation_id,
                    f"Unavailable Evaluation: {evaluation_id} ({error})",
                    False,
                )
            )
        else:
            training = load_training_run(info.path, evaluation.training_run_id)
            required_window = None
            required_stride = None
            if training.config.patch_size is not None:
                required_window = (
                    training.config.patch_size,
                    training.config.patch_size,
                )
                required_stride = (
                    training.config.patch_stride,
                    training.config.patch_stride,
                )
            options.append(
                DetectionProfileEvaluationOption(
                    evaluation_id,
                    f"Approved Evaluation {evaluation.evaluation_id} — Training Run {evaluation.training_run_id}",
                    True,
                    required_window,
                    required_stride,
                )
            )
    options.sort(key=lambda option: (not option.available, option.identifier))
    if not options:
        status = "No Approved Evaluations available for this project."
    elif any(not option.available for option in options):
        status = "Some Evaluations are unavailable or not Approved."
    else:
        status = f"{len(options)} Approved Evaluation(s) available."
    return DetectionProfileEvaluationInventory(tuple(options), status)


class DetectionProfileControls(QWidget):
    """Edit basic profile values and request one immutable project record."""

    _SPINBOX_MAX = 2_000_000_000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.evaluation_combo = QComboBox(self)
        self.evaluation_combo.setObjectName("detectionProfileEvaluationComboBox")
        self.evaluation_combo.setAccessibleName("Approved Evaluation for Detection Profile")

        self.window_width_spin = self._spin("detectionProfileWindowWidthSpinBox", 256)
        self.window_height_spin = self._spin("detectionProfileWindowHeightSpinBox", 256)
        self.stride_x_spin = self._spin("detectionProfileStrideXSpinBox", 256)
        self.stride_y_spin = self._spin("detectionProfileStrideYSpinBox", 256)
        self.reflect_padding_checkbox = QCheckBox("Reflect padding", self)
        self.reflect_padding_checkbox.setObjectName("detectionProfileReflectPaddingCheckBox")

        self.create_button = QPushButton("Create Detection Profile", self)
        self.create_button.setObjectName("createDetectionProfileButton")
        self.status_label = QLabel("No Approved Evaluations available.", self)
        self.status_label.setObjectName("detectionProfileCreationStatusLabel")
        self.status_label.setWordWrap(True)

        layout = QFormLayout(self)
        layout.addRow("Approved Evaluation", self.evaluation_combo)
        layout.addRow("Window width", self.window_width_spin)
        layout.addRow("Window height", self.window_height_spin)
        layout.addRow("Stride x", self.stride_x_spin)
        layout.addRow("Stride y", self.stride_y_spin)
        layout.addRow(self.reflect_padding_checkbox)
        layout.addRow(self.create_button)
        layout.addRow(self.status_label)

        self._create_callback: ProfileCreateCallback | None = None
        self._inventory_status = "No Approved Evaluations available."
        self.evaluation_combo.currentIndexChanged.connect(self._evaluation_changed)
        for spin in (
            self.window_width_spin,
            self.window_height_spin,
            self.stride_x_spin,
            self.stride_y_spin,
        ):
            spin.valueChanged.connect(self._refresh_gate)
        self.create_button.clicked.connect(self._create)
        self._refresh_gate()

    @staticmethod
    def _spin(object_name: str, value: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setObjectName(object_name)
        widget.setRange(1, DetectionProfileControls._SPINBOX_MAX)
        widget.setValue(value)
        return widget

    def configure(self, callback: ProfileCreateCallback | None) -> None:
        """Set the project service seam used by the Create button."""

        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or None")
        self._create_callback = callback
        self._refresh_gate()

    def set_inventory(self, inventory: DetectionProfileEvaluationInventory) -> None:
        """Display deterministic Approved/Unavailable Evaluation choices."""

        self._inventory_status = inventory.status
        self.evaluation_combo.clear()
        for option in inventory.options:
            self.evaluation_combo.addItem(option.label, option.identifier)
            self.evaluation_combo.setItemData(
                self.evaluation_combo.count() - 1,
                option,
                Qt.UserRole + 1,
            )
            item = self.evaluation_combo.model().item(self.evaluation_combo.count() - 1)
            if item is not None:
                item.setEnabled(option.available)
        self._evaluation_changed()

    def _evaluation_changed(self, *_args: object) -> None:
        option = self.evaluation_combo.itemData(
            self.evaluation_combo.currentIndex(), Qt.UserRole + 1
        )
        if (
            isinstance(option, DetectionProfileEvaluationOption)
            and option.required_window_size is not None
            and option.required_stride is not None
        ):
            self.window_width_spin.setValue(option.required_window_size[0])
            self.window_height_spin.setValue(option.required_window_size[1])
            self.stride_x_spin.setValue(option.required_stride[0])
            self.stride_y_spin.setValue(option.required_stride[1])
        self._refresh_gate()

    def _refresh_gate(self, *_args: object) -> None:
        index = self.evaluation_combo.currentIndex()
        item = self.evaluation_combo.model().item(index) if index >= 0 else None
        option = self.evaluation_combo.itemData(index, Qt.UserRole + 1) if index >= 0 else None
        geometry_matches = True
        if (
            isinstance(option, DetectionProfileEvaluationOption)
            and option.required_window_size is not None
            and option.required_stride is not None
        ):
            geometry_matches = (
                self.window_width_spin.value(),
                self.window_height_spin.value(),
            ) == option.required_window_size and (
                self.stride_x_spin.value(),
                self.stride_y_spin.value(),
            ) == option.required_stride
        self.status_label.setText(
            self._inventory_status
            if geometry_matches
            else "Detection Profile geometry must match Patch Classification checkpoint settings."
        )
        self.create_button.setEnabled(
            self._create_callback is not None
            and bool(self.evaluation_combo.currentData())
            and item is not None
            and item.isEnabled()
            and geometry_matches
        )

    def _create(self) -> None:
        callback = self._create_callback
        evaluation_id = self.evaluation_combo.currentData()
        if callback is None or not isinstance(evaluation_id, str) or not evaluation_id:
            self.status_label.setText("Detection Profile requires an Approved Evaluation.")
            self._refresh_gate()
            return
        draft = DetectionProfileDraft(
            evaluation_id=evaluation_id,
            window_size=(self.window_width_spin.value(), self.window_height_spin.value()),
            stride=(self.stride_x_spin.value(), self.stride_y_spin.value()),
            reflect_padding=self.reflect_padding_checkbox.isChecked(),
        )
        try:
            result = callback(draft)
        except Exception as error:  # The project service owns persistence/error policy.
            self.status_label.setText(f"Detection Profile not created: {error}")
            return
        profile_id = getattr(result, "profile_id", "")
        self.status_label.setText(
            f"Created Detection Profile {profile_id or '(unknown)'}."
        )


__all__ = [
    "DetectionProfileControls",
    "DetectionProfileDraft",
    "DetectionProfileEvaluationInventory",
    "DetectionProfileEvaluationOption",
    "load_detection_profile_evaluation_inventory",
]
