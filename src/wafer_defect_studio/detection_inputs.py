"""Read-only Detection Profile and Detection Run inventory."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QWidget

from .detection_run import load_detection_profile, load_detection_run
from .project import _DETECTION_SCHEMA_VERSION, open_project


@dataclass(frozen=True)
class DetectionInputOption:
    identifier: str
    label: str
    available: bool = True


@dataclass(frozen=True)
class DetectionInputInventory:
    profiles: tuple[DetectionInputOption, ...]
    runs: tuple[DetectionInputOption, ...]
    status: str


def load_detection_input_inventory(project_path: str | Path) -> DetectionInputInventory:
    """Load Detection Profile/Run rows without changing the project."""

    info = open_project(project_path)
    if info.schema_version < _DETECTION_SCHEMA_VERSION:
        return DetectionInputInventory(
            (), (), "No Detection Profiles or Runs available (project schema 15 is required)."
        )
    database = info.path / "project.sqlite"
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        profile_ids = tuple(
            row[0]
            for row in connection.execute(
                "SELECT profile_id FROM detection_profiles ORDER BY profile_id"
            )
        )
        run_ids = tuple(
            row[0]
            for row in connection.execute(
                "SELECT detection_run_id FROM detection_runs ORDER BY detection_run_id"
            )
        )
    finally:
        connection.close()

    profiles: list[DetectionInputOption] = []
    for profile_id in profile_ids:
        try:
            profile = load_detection_profile(info.path, profile_id)
        except Exception:
            profiles.append(
                DetectionInputOption(
                    profile_id,
                    f"Unavailable Detection Profile: {profile_id}",
                    False,
                )
            )
        else:
            profiles.append(
                DetectionInputOption(
                    profile_id,
                    f"Detection Profile {profile.profile_id} — Evaluation {profile.evaluation_id}",
                )
            )

    runs: list[DetectionInputOption] = []
    for run_id in run_ids:
        try:
            run = load_detection_run(info.path, run_id)
        except Exception:
            runs.append(
                DetectionInputOption(
                    run_id,
                    f"Unavailable Detection Run: {run_id}",
                    False,
                )
            )
        else:
            runs.append(
                DetectionInputOption(
                    run_id,
                    f"Detection Run {run.run_id} — Profile {run.profile_id}",
                )
            )
    profiles.sort(key=lambda option: (not option.available, option.identifier))
    runs.sort(key=lambda option: (not option.available, option.identifier))
    if not profiles and not runs:
        status = "No Detection Profiles or Runs available for this project."
    elif any(not option.available for option in (*profiles, *runs)):
        status = "Some Detection Profiles or Runs are unavailable."
    else:
        status = f"{len(profiles)} Profile(s), {len(runs)} Run(s) available."
    return DetectionInputInventory(tuple(profiles), tuple(runs), status)


class DetectionInputControls(QWidget):
    """Display immutable Detection Profile/Run choices."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.profile_combo = QComboBox(self)
        self.profile_combo.setObjectName("detectionProfileComboBox")
        self.profile_combo.setAccessibleName("Detection Profile")
        self.run_combo = QComboBox(self)
        self.run_combo.setObjectName("detectionRunComboBox")
        self.run_combo.setAccessibleName("Detection Run")
        self.status_label = QLabel("No Detection Profiles or Runs available.", self)
        self.status_label.setObjectName("detectionInventoryStatusLabel")
        self.status_label.setWordWrap(True)
        layout = QFormLayout(self)
        layout.addRow("Detection Profile", self.profile_combo)
        layout.addRow("Detection Run", self.run_combo)
        layout.addRow(self.status_label)

    def set_inventory(self, inventory: DetectionInputInventory) -> None:
        self._set_options(self.profile_combo, inventory.profiles)
        self._set_options(self.run_combo, inventory.runs)
        self.status_label.setText(inventory.status)

    @staticmethod
    def _set_options(
        combo: QComboBox, options: tuple[DetectionInputOption, ...]
    ) -> None:
        combo.clear()
        for option in options:
            combo.addItem(option.label, option.identifier)
            item = combo.model().item(combo.count() - 1)
            if item is not None:
                item.setEnabled(option.available)


__all__ = [
    "DetectionInputControls",
    "DetectionInputInventory",
    "DetectionInputOption",
    "load_detection_input_inventory",
]
