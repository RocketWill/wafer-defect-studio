"""Read-only Dataset Snapshot and Dataset Split choices for Train."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QLabel, QFormLayout, QWidget

from .dataset_snapshot import load_dataset_snapshot
from .dataset_split import load_dataset_split
from .project import open_project


@dataclass(frozen=True)
class TrainingInputOption:
    identifier: str
    label: str
    available: bool = True
    class_count: int | None = None
    snapshot_id: str | None = None


@dataclass(frozen=True)
class TrainingInputInventory:
    snapshots: tuple[TrainingInputOption, ...]
    splits: tuple[TrainingInputOption, ...]


def load_training_input_inventory(project_path: str | Path) -> TrainingInputInventory:
    """Load existing Snapshot/Split rows without creating or changing records."""

    info = open_project(project_path)
    if info.schema_version < 11:
        return TrainingInputInventory((), ())
    connection = sqlite3.connect((info.path / "project.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    try:
        snapshot_ids = tuple(
            row[0]
            for row in connection.execute(
                "SELECT snapshot_id FROM dataset_snapshots ORDER BY snapshot_id"
            )
        )
        split_ids = tuple(
            row[0]
            for row in connection.execute(
                "SELECT split_id FROM dataset_splits ORDER BY split_id"
            )
        ) if info.schema_version >= 12 else ()
    finally:
        connection.close()

    snapshots = []
    valid_snapshot_ids: set[str] = set()
    for snapshot_id in snapshot_ids:
        try:
            snapshot = load_dataset_snapshot(info.path, snapshot_id)
        except Exception:
            snapshots.append(
                TrainingInputOption(
                    snapshot_id,
                    f"Unavailable Dataset Snapshot: {snapshot_id}",
                    False,
                )
            )
        else:
            valid_snapshot_ids.add(snapshot_id)
            snapshots.append(
                TrainingInputOption(
                    snapshot_id,
                    f"Snapshot {snapshot_id} — {snapshot.created_at}",
                    True,
                    len(snapshot.classes),
                    snapshot_id,
                )
            )

    splits = []
    for split_id in split_ids:
        try:
            split = load_dataset_split(info.path, split_id)
            if split.snapshot_id not in valid_snapshot_ids:
                raise ValueError("Dataset Snapshot is unavailable")
        except Exception:
            splits.append(
                TrainingInputOption(
                    split_id,
                    f"Unavailable Dataset Split: {split_id}",
                    False,
                )
            )
        else:
            splits.append(
                TrainingInputOption(
                    split_id,
                    f"Split {split_id} — seed {split.seed}",
                    True,
                    None,
                    split.snapshot_id,
                )
            )
    snapshots.sort(key=lambda option: (not option.available, option.identifier))
    splits.sort(key=lambda option: (not option.available, option.identifier))
    return TrainingInputInventory(tuple(snapshots), tuple(splits))


class TrainingInputControls(QWidget):
    """Display persisted Dataset Snapshot/Split inputs without editing them."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.snapshot_combo = QComboBox(self)
        self.snapshot_combo.setObjectName("trainingSnapshotComboBox")
        self.snapshot_combo.setAccessibleName("Dataset Snapshot")
        self.split_combo = QComboBox(self)
        self.split_combo.setObjectName("trainingSplitComboBox")
        self.split_combo.setAccessibleName("Dataset Split")
        self.empty_state = QLabel("No Dataset Snapshot or Split available.", self)
        self.empty_state.setObjectName("trainingInputsEmptyState")
        self.empty_state.setWordWrap(True)
        layout = QFormLayout(self)
        layout.addRow("Dataset Snapshot", self.snapshot_combo)
        layout.addRow("Dataset Split", self.split_combo)
        layout.addRow(self.empty_state)

    def set_inventory(self, inventory: TrainingInputInventory) -> None:
        self._set_options(self.snapshot_combo, inventory.snapshots)
        self._set_options(self.split_combo, inventory.splits)
        self.empty_state.setVisible(not inventory.snapshots or not inventory.splits)

    @staticmethod
    def _set_options(combo: QComboBox, options: tuple[TrainingInputOption, ...]) -> None:
        combo.clear()
        for option in options:
            combo.addItem(option.label, option.identifier)
            item = combo.model().item(combo.count() - 1)
            if item is not None:
                item.setEnabled(option.available)
            combo.setItemData(combo.count() - 1, option, Qt.UserRole + 1)


__all__ = [
    "TrainingInputControls",
    "TrainingInputInventory",
    "TrainingInputOption",
    "load_training_input_inventory",
]
