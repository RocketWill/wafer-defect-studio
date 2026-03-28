"""Small asynchronous Training Scope and Dataset Snapshot controls."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import QCheckBox, QLabel, QPushButton, QVBoxLayout, QWidget

from .dataset_diagnostics import DatasetPreview
from .defect_class import DefectClass
from .training_scope import DataGroup


SnapshotCreator = Callable[
    [tuple[str, ...], tuple[str, ...]], tuple[str, DatasetPreview]
]


class _TaskSignals(QObject):
    completed = Signal(object)
    failed = Signal(str)


class _CreateTask(QRunnable):
    def __init__(self, creator: SnapshotCreator, groups: tuple[str, ...], classes: tuple[str, ...]):
        super().__init__()
        self.signals = _TaskSignals()
        self._creator = creator
        self._groups = groups
        self._classes = classes

    @Slot()
    def run(self) -> None:
        try:
            self.signals.completed.emit(self._creator(self._groups, self._classes))
        except Exception as error:  # Display the application service's actionable error.
            self.signals.failed.emit(str(error))


class TrainingScopeControls(QWidget):
    """Select snapshot inputs and run creation outside the GUI thread."""

    selectionChanged = Signal()
    snapshotCreated = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._group_layout = QVBoxLayout()
        self._class_layout = QVBoxLayout()
        self._group_boxes: dict[str, QCheckBox] = {}
        self._class_boxes: dict[str, QCheckBox] = {}
        self._creator: SnapshotCreator | None = None
        self._project_bound_options = False
        self._tasks: set[_CreateTask] = set()
        self.warnings_label = QLabel("No preview available.", self)
        self.warnings_label.setObjectName("datasetWarningsLabel")
        self.warnings_label.setWordWrap(True)
        self.status_label = QLabel("Select a Training Scope.", self)
        self.status_label.setObjectName("datasetSnapshotStatusLabel")
        self.status_label.setWordWrap(True)
        self.create_button = QPushButton("Create Dataset Snapshot", self)
        self.create_button.setObjectName("createDatasetSnapshotButton")
        self.create_button.clicked.connect(self._create)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Data Groups", self))
        layout.addLayout(self._group_layout)
        self.group_empty_state = QLabel("No Data Groups available.", self)
        self.group_empty_state.setObjectName("datasetGroupsEmptyState")
        self.group_empty_state.setWordWrap(True)
        layout.addWidget(self.group_empty_state)
        layout.addWidget(QLabel("Defect Classes", self))
        layout.addLayout(self._class_layout)
        self.class_empty_state = QLabel("No enabled Defect Classes available.", self)
        self.class_empty_state.setObjectName("datasetClassesEmptyState")
        self.class_empty_state.setWordWrap(True)
        layout.addWidget(self.class_empty_state)
        layout.addWidget(QLabel("Warnings", self))
        layout.addWidget(self.warnings_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.create_button)
        self._update_empty_states((), ())

    def set_available_options(
        self,
        groups: tuple[DataGroup, ...],
        classes: tuple[DefectClass, ...],
    ) -> None:
        """Bind persisted project options without injecting a snapshot creator."""

        enabled_classes = tuple(item for item in classes if item.enabled)
        self._project_bound_options = True
        self._replace_options(self._group_layout, self._group_boxes, groups, "dataGroup")
        self._replace_options(
            self._class_layout,
            self._class_boxes,
            enabled_classes,
            "snapshotClass",
        )
        self._creator = None
        self._update_empty_states(groups, enabled_classes)
        self.warnings_label.setText("Select Data Groups and Defect Classes to preview eligibility.")
        self.status_label.setText("Dataset options loaded. Select a Training Scope.")
        self.create_button.setEnabled(False)

    def show_preview(self, preview: DatasetPreview) -> None:
        """Render a project-bound preview while keeping creation disabled."""

        self._show_warnings(preview)
        image_distribution = ", ".join(
            f"{name}: {count}" for name, count in preview.image_distribution
        )
        group_distribution = ", ".join(
            f"{name}: {count}" for name, count in preview.group_distribution
        )
        class_distribution = ", ".join(
            f"{name}: {count}" for name, count in preview.class_distribution
        )
        details = [f"Images — {image_distribution}"]
        if group_distribution:
            details.append(f"Groups — {group_distribution}")
        if class_distribution:
            details.append(f"Classes — {class_distribution}")
        self.status_label.setText("Preview — " + " · ".join(details))
        self.create_button.setEnabled(
            self._creator is not None
            and bool(self.selected_data_groups())
            and bool(self.selected_classes())
        )

    def show_preview_empty(self, message: str) -> None:
        """Show an actionable state when a preview cannot be calculated."""

        self.warnings_label.setText(message)
        self.status_label.setText("Select a Training Scope.")
        self.create_button.setEnabled(False)

    def set_snapshot_creator(self, creator: SnapshotCreator) -> None:
        """Attach the active project's asynchronous snapshot creator."""

        self._creator = creator
        self.create_button.setEnabled(False)

    def configure(
        self,
        groups: tuple[DataGroup, ...],
        classes: tuple[DefectClass, ...],
        preview: DatasetPreview,
        creator: SnapshotCreator,
    ) -> None:
        self._replace_options(self._group_layout, self._group_boxes, groups, "dataGroup")
        enabled_classes = tuple(item for item in classes if item.enabled)
        self._project_bound_options = False
        self._replace_options(self._class_layout, self._class_boxes, enabled_classes, "snapshotClass")
        self._update_empty_states(groups, enabled_classes)
        self._creator = creator
        self._show_warnings(preview)
        self.status_label.setText("Select a Training Scope.")
        self.create_button.setEnabled(True)

    @property
    def project_bound_options(self) -> bool:
        """Whether options came from the active project binding."""

        return self._project_bound_options

    def set_selected_data_groups(self, identifiers: tuple[str, ...]) -> None:
        self._set_selected(self._group_boxes, identifiers)

    def set_selected_classes(self, identifiers: tuple[str, ...]) -> None:
        self._set_selected(self._class_boxes, identifiers)

    def selected_data_groups(self) -> tuple[str, ...]:
        return tuple(key for key, box in self._group_boxes.items() if box.isChecked())

    def selected_classes(self) -> tuple[str, ...]:
        return tuple(key for key, box in self._class_boxes.items() if box.isChecked())

    def _create(self) -> None:
        groups, classes = self.selected_data_groups(), self.selected_classes()
        if not groups or not classes:
            self.status_label.setText("Select at least one Data Group and Defect Class.")
            return
        if self._creator is None:
            self.status_label.setText("Configure a snapshot creator before creating.")
            return
        self.create_button.setEnabled(False)
        self.status_label.setText("Creating snapshot…")
        task = _CreateTask(self._creator, groups, classes)
        self._tasks.add(task)
        task.signals.completed.connect(lambda result, item=task: self._created(item, result))
        task.signals.failed.connect(lambda message, item=task: self._failed(item, message))
        QThreadPool.globalInstance().start(task)

    def _created(self, task: _CreateTask, result: object) -> None:
        self._tasks.discard(task)
        snapshot_id, preview = result
        self._show_warnings(preview)
        distribution = ", ".join(f"{name}: {count}" for name, count in preview.image_distribution)
        self.status_label.setText(f"Created {snapshot_id} — {distribution}")
        self.create_button.setEnabled(True)
        self.snapshotCreated.emit(str(snapshot_id))

    def _failed(self, task: _CreateTask, message: str) -> None:
        self._tasks.discard(task)
        self.status_label.setText(f"Snapshot not created: {message or 'Check the Training Scope.'}")
        self.create_button.setEnabled(True)

    def _show_warnings(self, preview: DatasetPreview) -> None:
        self.warnings_label.setText(
            "\n".join(warning.message for warning in preview.warnings) or "No warnings."
        )

    def _update_empty_states(self, groups, classes) -> None:
        self.group_empty_state.setVisible(not groups)
        self.class_empty_state.setVisible(not classes)

    def _replace_options(
        self, layout, boxes: dict[str, QCheckBox], values, prefix: str
    ) -> None:
        while layout.count():
            widget = layout.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        boxes.clear()
        for value in values:
            identifier = value.data_group_id if isinstance(value, DataGroup) else value.code
            box = QCheckBox(value.name)
            box.setObjectName(f"{prefix}_{identifier}CheckBox")
            box.toggled.connect(lambda _checked: self.selectionChanged.emit())
            boxes[identifier] = box
            layout.addWidget(box)

    @staticmethod
    def _set_selected(boxes: dict[str, QCheckBox], identifiers: tuple[str, ...]) -> None:
        selected = set(identifiers)
        for key, box in boxes.items():
            box.setChecked(key in selected)
