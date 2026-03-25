"""Main window for the initial application shell."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QPoint, QSettings, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QInputDialog,
    QPushButton,
    QToolBar,
    QTableWidget,
    QTableWidgetItem,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .image_asset import ImageAsset, ReopenedWaferImage, SourceHealth, _source_health
from .annotation import GridAnnotation, save_grid_annotation
from .annotation_tools import AnnotationMode
from .autosave import AutosaveGuard, SaveFailureState
from .defect_class import DefectClass, load_defect_classes
from .effective_area import (
    EffectiveWaferArea,
    confirm_effective_wafer_area,
    load_effective_wafer_area,
    participating_annotation_grids,
)
from .grid_controls import GridProfileControls
from .grid_geometry import annotation_grids
from .grid_profile import (
    GridProfile,
    GridProfileConflictError,
    load_grid_profiles,
    save_grid_profile,
)
from .image_grid_placement import load_image_grid_placement, set_image_grid_origin
from .review import ReviewError, load_review_state, mark_image_reviewed, reopen_image
from .review_counts import ReviewCounts, load_review_counts
from .training_scope_controls import SnapshotCreator, TrainingScopeControls
from .training_scope import (
    DataGroup,
    ensure_data_group_schema,
    load_data_groups,
    load_image_data_group_assignments,
    save_data_groups,
)
from .dataset_diagnostics import DatasetPreview
from .evaluation_controls import DecisionService, EvaluationControls
from .detection_controls import DetectionControls, DetectionLauncher, DetectionRequestSource
from .proposal_controls import ProposalReviewControls, ReviewCallback
from .conversion_controls import ProposalConversionControls, ConversionCallback
from .export_controls import ExportCallback, ResultExportControls
from .job_controls import JobsActionCallback, JobsControls
from .job_recovery import recover_stale_jobs
from .job_store import list_jobs
from .accessibility_audit import ensure_accessible_labels
from . import image_asset
from . import project
from .ui_theme import ThemeMode, apply_theme
from .training_controls import CloneCallback, TrainingControls, TrainingLauncher, TrainingRequestSource
from .wafer_loader import WaferLoader
from .wafer_view import LoadedWaferImage, WaferView, _decode_wafer_image


class _GridOriginControls(QWidget):
    draftChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.origin_x_spin = QSpinBox(self)
        self.origin_x_spin.setObjectName("gridOriginXSpinBox")
        self.origin_y_spin = QSpinBox(self)
        self.origin_y_spin.setObjectName("gridOriginYSpinBox")
        self.apply_button = QPushButton("Apply Origin", self)
        self.apply_button.setObjectName("applyGridOriginButton")
        self.apply_button.setEnabled(False)
        self.participating_count_label = QLabel("Participating: 0", self)
        self.participating_count_label.setObjectName("participatingGridCountLabel")
        self.confirmation_label = QLabel("Unconfirmed", self)
        self.confirmation_label.setObjectName("effectiveAreaConfirmationLabel")
        self.confirm_button = QPushButton("Confirm Area", self)
        self.confirm_button.setObjectName("confirmEffectiveWaferAreaButton")
        self.confirm_button.setEnabled(False)
        form = QFormLayout()
        form.addRow("Origin x (px)", self.origin_x_spin)
        form.addRow("Origin y (px)", self.origin_y_spin)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.apply_button)
        layout.addWidget(self.participating_count_label)
        layout.addWidget(self.confirmation_label)
        layout.addWidget(self.confirm_button)
        self.origin_x_spin.valueChanged.connect(lambda _value: self.draftChanged.emit())
        self.origin_y_spin.valueChanged.connect(lambda _value: self.draftChanged.emit())

    def bind(self, origin_x: int, origin_y: int, cell_width: int, cell_height: int) -> None:
        widgets = (self.origin_x_spin, self.origin_y_spin)
        previous = [widget.blockSignals(True) for widget in widgets]
        try:
            self.origin_x_spin.setRange(0, cell_width - 1)
            self.origin_y_spin.setRange(0, cell_height - 1)
            self.origin_x_spin.setValue(origin_x)
            self.origin_y_spin.setValue(origin_y)
            self.apply_button.setEnabled(False)
        finally:
            for widget, was_blocked in zip(widgets, previous):
                widget.blockSignals(was_blocked)

    def draft_origin(self) -> tuple[int, int]:
        return self.origin_x_spin.value(), self.origin_y_spin.value()

    def set_participating_count(self, count: int) -> None:
        self.participating_count_label.setText(f"Participating: {count}")

    def set_confirmation_state(self, has_area: bool, confirmed: bool) -> None:
        self.confirmation_label.setText("Confirmed" if confirmed else "Unconfirmed")
        self.confirm_button.setEnabled(has_area and not confirmed)


class _AnnotationToolControls(QWidget):
    """Visible tool buttons and a compact multi-select Defect Class list."""

    modeChanged = Signal(object)
    classSelectionChanged = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.mode_label = QLabel("Mode: Pan", self)
        self.mode_label.setObjectName("annotationModeLabel")
        self._mode_buttons: dict[AnnotationMode, QPushButton] = {}
        mode_row = QHBoxLayout()
        for mode, name in (
            (AnnotationMode.PAN, "Pan"),
            (AnnotationMode.ANNOTATE, "Annotate"),
            (AnnotationMode.PAINT, "Paint"),
            (AnnotationMode.ERASE, "Erase"),
        ):
            button = QPushButton(name, self)
            button.setCheckable(True)
            button.setObjectName(f"{name.lower()}ToolButton")
            button.clicked.connect(lambda _checked, selected=mode: self._choose_mode(selected))
            mode_row.addWidget(button)
            self._mode_buttons[mode] = button
        self._mode_buttons[AnnotationMode.PAN].setChecked(True)
        self._class_layout = QVBoxLayout()
        self._class_boxes: dict[str, QCheckBox] = {}
        layout = QVBoxLayout(self)
        layout.addWidget(self.mode_label)
        layout.addLayout(mode_row)
        layout.addWidget(QLabel("Defect Classes", self))
        layout.addLayout(self._class_layout)

    def _choose_mode(self, mode: AnnotationMode) -> None:
        self.set_mode(mode)
        self.modeChanged.emit(mode)

    def set_mode(self, mode: AnnotationMode | str) -> None:
        selected = mode if isinstance(mode, AnnotationMode) else AnnotationMode(mode)
        for candidate, button in self._mode_buttons.items():
            button.setChecked(candidate is selected)
        self.mode_label.setText(f"Mode: {selected.value}")

    def set_classes(self, classes: tuple[DefectClass, ...]) -> None:
        while self._class_layout.count():
            item = self._class_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._class_boxes.clear()
        for defect_class in classes:
            box = QCheckBox(f"{defect_class.code} — {defect_class.name}", self)
            box.setObjectName(f"defectClass_{defect_class.code}CheckBox")
            box.setEnabled(defect_class.enabled)
            box.toggled.connect(self._emit_selected_classes)
            self._class_layout.addWidget(box)
            self._class_boxes[defect_class.code] = box

    def set_selected_classes(self, codes) -> None:
        selected = set(codes)
        for code, box in self._class_boxes.items():
            blocked = box.blockSignals(True)
            box.setChecked(code in selected)
            box.blockSignals(blocked)

    def selected_classes(self) -> tuple[str, ...]:
        return tuple(code for code, box in self._class_boxes.items() if box.isChecked())

    def _emit_selected_classes(self, _checked: bool) -> None:
        self.classSelectionChanged.emit(self.selected_classes())


class _ReviewControls(QWidget):
    """Compact counts and image-level review actions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.labeled_count_label = QLabel("Labeled: —", self)
        self.labeled_count_label.setObjectName("labeledGridCountLabel")
        self.unreviewed_count_label = QLabel("Unreviewed: —", self)
        self.unreviewed_count_label.setObjectName("unreviewedGridCountLabel")
        self.derived_normal_count_label = QLabel("Derived Normal: —", self)
        self.derived_normal_count_label.setObjectName("derivedNormalGridCountLabel")
        self.excluded_count_label = QLabel("Excluded: —", self)
        self.excluded_count_label.setObjectName("excludedGridCountLabel")
        self.mark_button = QPushButton("Mark Reviewed", self)
        self.mark_button.setObjectName("markImageReviewedButton")
        self.mark_button.setEnabled(False)
        self.reopen_button = QPushButton("Reopen Image", self)
        self.reopen_button.setObjectName("reopenImageButton")
        self.reopen_button.setEnabled(False)
        layout = QVBoxLayout(self)
        layout.addWidget(self.labeled_count_label)
        layout.addWidget(self.unreviewed_count_label)
        layout.addWidget(self.derived_normal_count_label)
        layout.addWidget(self.excluded_count_label)
        layout.addWidget(self.mark_button)
        layout.addWidget(self.reopen_button)

    def set_counts(self, counts: ReviewCounts | None) -> None:
        if counts is None:
            self.labeled_count_label.setText("Labeled: —")
            self.unreviewed_count_label.setText("Unreviewed: —")
            self.derived_normal_count_label.setText("Derived Normal: —")
            self.excluded_count_label.setText("Excluded: —")
            return
        self.labeled_count_label.setText(f"Labeled: {counts.labeled}")
        self.unreviewed_count_label.setText(f"Unreviewed: {counts.unreviewed}")
        self.derived_normal_count_label.setText(f"Derived Normal: {counts.derived_normal}")
        self.excluded_count_label.setText(f"Excluded: {counts.excluded}")

    def set_review_state(self, reviewed: bool, available: bool) -> None:
        self.mark_button.setEnabled(available and not reviewed)
        self.reopen_button.setEnabled(available and reviewed)


class _AutosaveFailureControls(QWidget):
    """Actions shown while one annotation edit is waiting for recovery."""

    retryRequested = Signal()
    saveAsRequested = Signal()
    cancelRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.message_label = QLabel("Save failed", self)
        self.message_label.setObjectName("autosaveFailureMessageLabel")
        self.retry_button = QPushButton("Retry", self)
        self.retry_button.setObjectName("retryAutosaveButton")
        self.save_as_button = QPushButton("Save As", self)
        self.save_as_button.setObjectName("saveAsAutosaveButton")
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setObjectName("cancelAutosaveButton")
        self.retry_button.clicked.connect(self.retryRequested)
        self.save_as_button.clicked.connect(self.saveAsRequested)
        self.cancel_button.clicked.connect(self.cancelRequested)
        layout = QVBoxLayout(self)
        layout.addWidget(self.message_label)
        layout.addWidget(self.retry_button)
        layout.addWidget(self.save_as_button)
        layout.addWidget(self.cancel_button)

    def set_failure(self, failure: SaveFailureState) -> None:
        self.message_label.setText(f"Save failed: {failure.error_message}")



class MainWindow(QMainWindow):
    """Top-level window for Wafer Defect Studio."""

    _RECENT_PROJECTS_KEY = "recentProjects"
    _ACTIVE_PROJECT_KEY = "activeProjectPath"
    _CURRENT_WORKSPACE_KEY = "currentWorkspace"
    _CURRENT_IMAGE_ASSET_KEY = "currentImageAssetId"
    WORKSPACES = (
        "Data",
        "Annotate",
        "Dataset",
        "Train",
        "Evaluate",
        "Detect",
        "Review",
    )
    workspaceChanged = Signal(str)

    def __init__(
        self,
        parent: QMainWindow | None = None,
        *,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Wafer Defect Studio")
        self._settings = (
            settings
            if settings is not None
            else QSettings("WaferDefectStudio", "WaferDefectStudio")
        )
        self._active_project_path: Path | None = None
        self._current_workspace = self.WORKSPACES[0]
        self.workspace_toolbar = QToolBar("Workspace Navigation", self)
        self.workspace_toolbar.setObjectName("workspaceToolbar")
        self.workspace_toolbar.setAccessibleName("Workspace navigation")
        self.workspace_toolbar.setMovable(False)
        self.workspace_toolbar.setFloatable(False)
        self.workspace_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.workspace_toolbar.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._workspace_action_group = QActionGroup(self)
        self._workspace_action_group.setExclusive(True)
        self.workspace_actions: dict[str, QAction] = {}
        for workspace in self.WORKSPACES:
            action = QAction(workspace, self)
            action.setObjectName(f"{workspace.lower()}WorkspaceAction")
            action.setCheckable(True)
            action.setActionGroup(self._workspace_action_group)
            action.triggered.connect(
                lambda _checked, selected=workspace: self._set_workspace(selected)
            )
            self.workspace_toolbar.addAction(action)
            button = self.workspace_toolbar.widgetForAction(action)
            if button is not None:
                button.setObjectName(f"{workspace.lower()}WorkspaceButton")
                button.setAccessibleName(workspace)
                button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.workspace_actions[workspace] = action
        self.workspace_actions[self._current_workspace].setChecked(True)
        self._set_workspace_actions_enabled(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.workspace_toolbar)
        file_menu = self.menuBar().addMenu("File")
        self.create_project_action = QAction("Create Project…", self)
        self.create_project_action.setObjectName("createProjectAction")
        self.create_project_action.triggered.connect(self._create_project)
        file_menu.addAction(self.create_project_action)
        self.open_project_action = QAction("Open Project…", self)
        self.open_project_action.setObjectName("openProjectAction")
        self.open_project_action.triggered.connect(self._open_project)
        file_menu.addAction(self.open_project_action)
        self.import_wafer_image_action = QAction("Import Wafer Image…", self)
        self.import_wafer_image_action.setObjectName("importWaferImageAction")
        self.import_wafer_image_action.setEnabled(False)
        self.import_wafer_image_action.triggered.connect(self._import_wafer_image)
        file_menu.addAction(self.import_wafer_image_action)
        self._loaded_wafer_image: LoadedWaferImage | None = None
        self._image_view = WaferView()
        self.setCentralWidget(self._image_view)
        self._data_workspace = QWidget(self)
        self._data_workspace.setObjectName("dataWorkspace")
        self._data_workspace.setAccessibleName("Data workspace")
        data_layout = QVBoxLayout(self._data_workspace)
        data_title = QLabel("Data", self._data_workspace)
        data_title.setObjectName("dataWorkspaceTitle")
        data_layout.addWidget(data_title)
        self._data_workspace_context_label = QLabel(
            "No active project. Create or open a project to view registered Wafer Images.",
            self._data_workspace,
        )
        self._data_workspace_context_label.setObjectName("dataWorkspaceContextLabel")
        self._data_workspace_context_label.setWordWrap(True)
        data_layout.addWidget(self._data_workspace_context_label)
        self._create_data_group_button = QPushButton("Create Data Group…", self._data_workspace)
        self._create_data_group_button.setObjectName("createDataGroupButton")
        self._create_data_group_button.setAccessibleName("Create Data Group")
        self._create_data_group_button.setEnabled(False)
        self._create_data_group_button.clicked.connect(self._create_data_group)
        data_layout.addWidget(self._create_data_group_button)
        self._image_inventory_table = QTableWidget(0, 5, self._data_workspace)
        self._image_inventory_table.setObjectName("imageInventoryTable")
        self._image_inventory_table.setAccessibleName("Wafer Image inventory")
        self._image_inventory_table.setHorizontalHeaderLabels(
            ("Wafer Image", "Dimensions", "Format", "Source Health", "Data Group")
        )
        self._image_inventory_table.horizontalHeader().setAccessibleName(
            "Wafer Image inventory columns"
        )
        self._image_inventory_table.verticalHeader().setAccessibleName(
            "Wafer Image inventory rows"
        )
        self._image_inventory_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._image_inventory_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._image_inventory_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._image_inventory_table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._image_inventory_table.itemSelectionChanged.connect(
            self._on_image_inventory_selection_changed
        )
        data_layout.addWidget(self._image_inventory_table)
        self._data_group_inventory_table = QTableWidget(0, 2, self._data_workspace)
        self._data_group_inventory_table.setObjectName("dataGroupInventoryTable")
        self._data_group_inventory_table.setAccessibleName("Data Group inventory")
        self._data_group_inventory_table.setHorizontalHeaderLabels(
            ("Data Group", "Assigned Images")
        )
        self._data_group_inventory_table.horizontalHeader().setAccessibleName(
            "Data Group inventory columns"
        )
        self._data_group_inventory_table.verticalHeader().setAccessibleName(
            "Data Group inventory rows"
        )
        self._data_group_inventory_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._data_group_inventory_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._data_group_inventory_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._data_group_inventory_table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        data_layout.addWidget(self._data_group_inventory_table)
        self._data_groups_empty_state = QLabel(
            "No Data Groups configured.", self._data_workspace
        )
        self._data_groups_empty_state.setObjectName("dataGroupsEmptyState")
        self._data_groups_empty_state.setWordWrap(True)
        data_layout.addWidget(self._data_groups_empty_state)
        self._image_inspector_empty_state = QLabel(
            "No active project. Select a Wafer Image to inspect its metadata.",
            self._data_workspace,
        )
        self._image_inspector_empty_state.setObjectName("imageInspectorEmptyState")
        self._image_inspector_empty_state.setWordWrap(True)
        data_layout.addWidget(self._image_inspector_empty_state)
        self._data_workspace_dock = QDockWidget("Data", self)
        self._data_workspace_dock.setObjectName("dataWorkspaceDock")
        self._data_workspace_dock.setAccessibleName("Data workspace")
        self._data_workspace_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._data_workspace_dock.setWidget(self._data_workspace)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._data_workspace_dock)
        self._data_workspace_dock.show()
        self.project_hub_list = QListWidget(self)
        self.project_hub_list.setObjectName("projectHubList")
        self.project_hub_list.itemActivated.connect(self._open_recent_project)
        self._project_hub_dock = QDockWidget("Project Hub", self)
        self._project_hub_dock.setObjectName("projectHubDock")
        self._project_hub_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._project_hub_dock.setWidget(self.project_hub_list)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._project_hub_dock)
        self._project_hub_dock.show()
        self._refresh_project_hub()
        self._annotation_controls = _AnnotationToolControls()
        self._annotation_controls.modeChanged.connect(self._image_view.set_tool_mode)
        self._annotation_controls.classSelectionChanged.connect(
            self._on_annotation_class_selection_changed
        )
        self._image_view.modeChanged.connect(self._annotation_controls.set_mode)
        self._image_view.classSelectionChanged.connect(
            self._annotation_controls.set_selected_classes
        )
        self._image_view.annotationChanged.connect(self._save_annotation_change)
        self._annotation_tool_dock = QDockWidget("Annotation Tools", self)
        self._annotation_tool_dock.setObjectName("annotationToolsDock")
        self._annotation_tool_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._annotation_tool_dock.setWidget(self._annotation_controls)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._annotation_tool_dock)
        # Keep the canvas usable at the default window size until a project
        # provides at least one Defect Class to annotate.  The tool pane has
        # four side-by-side buttons and therefore a wide minimum size; leaving
        # it open before classes are available can consume the entire canvas
        # (and force the window wider than the requested size).
        self._annotation_tool_dock.setEnabled(False)
        self._annotation_tool_dock.hide()
        self._review_controls = _ReviewControls()
        self._review_controls.mark_button.clicked.connect(self._mark_current_image_reviewed)
        self._review_controls.reopen_button.clicked.connect(self._reopen_current_image)
        self._review_dock = QDockWidget("Review", self)
        self._review_dock.setObjectName("reviewControlsDock")
        self._review_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._review_dock.setWidget(self._review_controls)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._review_dock)
        self._review_dock.setEnabled(False)
        self._review_dock.hide()
        self._wafer_loader = WaferLoader(self)
        self._wafer_loader.loaded.connect(self._on_load_ready)
        self._wafer_loader.failed.connect(self._on_load_error)
        self._latest_load_token = 0
        self._latest_lossy_source = False
        self._load_threads = self._wafer_loader._threads
        self._pending_image_assets: dict[int, ImageAsset] = {}
        self._current_image_asset: ImageAsset | None = None
        self._grid_project_path: Path | None = None
        self._grid_profile: GridProfile | None = None
        self._grid_origin = (0, 0)
        self._effective_wafer_area: EffectiveWaferArea | None = None
        self._autosave_guard = AutosaveGuard(
            self._persist_annotation,
            restore_callback=self._restore_annotation,
            apply_callback=self._apply_annotation,
            save_as_callback=self._persist_annotation_as,
        )
        self._autosave_failure_controls = _AutosaveFailureControls()
        self._autosave_failure_controls.retryRequested.connect(self._retry_autosave)
        self._autosave_failure_controls.saveAsRequested.connect(self._save_as_autosave)
        self._autosave_failure_controls.cancelRequested.connect(self._cancel_autosave)
        self._autosave_failure_dock = QDockWidget("Autosave", self)
        self._autosave_failure_dock.setObjectName("autosaveFailureDock")
        self._autosave_failure_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._autosave_failure_dock.setWidget(self._autosave_failure_controls)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._autosave_failure_dock)
        self._autosave_failure_dock.setEnabled(False)
        self._autosave_failure_dock.hide()
        self._grid_controls = GridProfileControls()
        self._grid_controls.draftChanged.connect(self._on_grid_draft_changed)
        self._grid_controls.apply_button.clicked.connect(self._apply_grid_profile)
        self._grid_profile_dock = QDockWidget("Grid Profile", self)
        self._grid_profile_dock.setObjectName("gridProfileDock")
        self._grid_profile_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._grid_profile_dock.setWidget(self._grid_controls)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._grid_profile_dock)
        self._grid_profile_dock.setEnabled(False)
        self._grid_profile_dock.hide()
        self._grid_origin_controls = _GridOriginControls()
        self._grid_origin_controls.draftChanged.connect(self._on_grid_origin_draft_changed)
        self._grid_origin_controls.apply_button.clicked.connect(self._apply_grid_origin)
        self._grid_origin_controls.confirm_button.clicked.connect(self._confirm_effective_wafer_area)
        self._grid_origin_dock = QDockWidget("Grid Origin", self)
        self._grid_origin_dock.setObjectName("gridOriginDock")
        self._grid_origin_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._grid_origin_dock.setWidget(self._grid_origin_controls)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._grid_origin_dock)
        self._grid_origin_dock.setEnabled(False)
        self._grid_origin_dock.hide()
        self._training_scope_controls = TrainingScopeControls()
        self._training_scope_dock = QDockWidget("Dataset Snapshot", self)
        self._training_scope_dock.setObjectName("datasetSnapshotDock")
        self._training_scope_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._training_scope_dock.setWidget(self._training_scope_controls)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._training_scope_dock)
        self._training_scope_dock.hide()
        self._training_controls = TrainingControls()
        self._training_dock = QDockWidget("Training", self)
        self._training_dock.setObjectName("trainingDock")
        self._training_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._training_dock.setWidget(self._training_controls)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._training_dock)
        self._training_dock.hide()
        self._evaluation_controls = EvaluationControls()
        self._evaluation_dock = QDockWidget("Evaluation", self)
        self._evaluation_dock.setObjectName("evaluationDock")
        self._evaluation_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._evaluation_dock.setWidget(self._evaluation_controls)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._evaluation_dock)
        self._evaluation_dock.hide()
        self._detection_controls = DetectionControls()
        self._detection_dock = QDockWidget("Detection", self)
        self._detection_dock.setObjectName("detectionControlsDock")
        self._detection_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._detection_dock.setWidget(self._detection_controls)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._detection_dock)
        self._detection_dock.hide()
        self._proposal_review_controls = ProposalReviewControls()
        self._proposal_review_dock = QDockWidget("Proposal Review", self)
        self._proposal_review_dock.setObjectName("proposalReviewDock")
        self._proposal_review_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._proposal_review_dock.setWidget(self._proposal_review_controls)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._proposal_review_dock)
        self._proposal_review_dock.setEnabled(False)
        self._proposal_review_dock.hide()
        self._proposal_conversion_controls = ProposalConversionControls()
        self._proposal_conversion_dock = QDockWidget("Proposal Conversion", self)
        self._proposal_conversion_dock.setObjectName("proposalConversionDock")
        self._proposal_conversion_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._proposal_conversion_dock.setWidget(self._proposal_conversion_controls)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._proposal_conversion_dock)
        self._proposal_conversion_dock.setEnabled(False)
        self._proposal_conversion_dock.hide()
        self._result_export_controls = ResultExportControls()
        self._result_export_dock = QDockWidget("Result Export", self)
        self._result_export_dock.setObjectName("resultExportDock")
        self._result_export_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._result_export_dock.setWidget(self._result_export_controls)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._result_export_dock)
        self._result_export_dock.setEnabled(False)
        self._result_export_dock.hide()
        self._jobs_controls = JobsControls()
        self._jobs_dock = QDockWidget("Jobs", self)
        self._jobs_dock.setObjectName("jobsDock")
        self._jobs_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._jobs_dock.setWidget(self._jobs_controls)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._jobs_dock)
        self._jobs_dock.setEnabled(False)
        self._jobs_dock.hide()
        ensure_accessible_labels(self)
        apply_theme(self, ThemeMode.SYSTEM)
        self.statusBar().showMessage(
            f"No active project — Workspace: {self._current_workspace}"
        )
        self._restore_workspace_context()

    @property
    def current_workspace(self) -> str:
        """Return the currently selected product workspace."""

        return self._current_workspace

    def _set_workspace(self, workspace: str) -> None:
        if workspace == self._current_workspace:
            return
        if workspace not in self.workspace_actions:
            raise ValueError(f"Unknown workspace: {workspace}")
        self._current_workspace = workspace
        self.workspace_actions[workspace].setChecked(True)
        self._settings.setValue(self._CURRENT_WORKSPACE_KEY, workspace)
        self._settings.sync()
        self.statusBar().showMessage(f"Workspace: {workspace}")
        self._data_workspace_dock.setVisible(workspace == self.WORKSPACES[0])
        self.workspaceChanged.emit(workspace)

    def _set_workspace_actions_enabled(self, has_project: bool) -> None:
        """Keep Data available while project-dependent workspaces are gated."""

        for workspace, action in self.workspace_actions.items():
            action.setEnabled(has_project or workspace == self.WORKSPACES[0])

    @property
    def active_project_path(self) -> Path | None:
        """Return the currently active project path, if one is open."""

        return self._active_project_path

    @property
    def current_image_asset(self) -> ImageAsset | None:
        """Return the currently displayed registered image asset, if any."""

        return self._current_image_asset

    @property
    def loaded_wafer_image(self) -> LoadedWaferImage | None:
        """Return the currently displayed native image metadata, if any."""

        return self._loaded_wafer_image

    def _recent_project_paths(self) -> list[str]:
        value = self._settings.value(self._RECENT_PROJECTS_KEY, [])
        if isinstance(value, str):
            value = [value]
        if not value:
            return []
        return [str(path) for path in value if str(path)][:10]

    def _refresh_project_hub(self) -> None:
        self.project_hub_list.clear()
        for project_path in self._recent_project_paths():
            status = self._project_source_health(project_path)
            item = QListWidgetItem(f"{project_path} — {status}", self.project_hub_list)
            item.setData(Qt.ItemDataRole.UserRole, project_path)

    @staticmethod
    def _project_source_health(project_path: str | Path) -> str:
        try:
            assets = image_asset.load_image_assets(project_path)
        except Exception:
            return "Project Unavailable"
        if not assets:
            return "No Wafer Images"
        if any(reopened.source_health is SourceHealth.MISSING for reopened in assets):
            return "Missing Source"
        if any(reopened.source_health is SourceHealth.CHANGED for reopened in assets):
            return "Changed Source"
        return "Available"

    def _remember_recent_project(self, project_path: str | Path) -> None:
        resolved_path = str(Path(project_path).expanduser().resolve())
        recent_paths = [
            candidate
            for candidate in self._recent_project_paths()
            if candidate != resolved_path
        ]
        recent_paths.insert(0, resolved_path)
        self._settings.setValue(self._RECENT_PROJECTS_KEY, recent_paths[:10])
        self._settings.sync()
        self._refresh_project_hub()

    def _activate_project(self, project_info, status: str) -> None:
        self._active_project_path = project_info.path
        self._settings.setValue(self._ACTIVE_PROJECT_KEY, str(project_info.path))
        self._settings.sync()
        self._set_workspace_actions_enabled(True)
        self.import_wafer_image_action.setEnabled(True)
        self._create_data_group_button.setEnabled(True)
        self._data_workspace_context_label.setText(
            f"Project: {project_info.path}. Registered Wafer Images will appear here."
        )
        self._image_inspector_empty_state.setText(
            "Select a Wafer Image to inspect its metadata."
        )
        self.setWindowTitle(f"Wafer Defect Studio — {project_info.path.name}")
        self.statusBar().showMessage(f"{status}: {project_info.path}")
        self._refresh_image_inventory(project_info.path)
        self._refresh_data_group_inventory(project_info.path)
        self._remember_recent_project(project_info.path)

    def _refresh_data_group_inventory(
        self, project_path: str | Path | None = None
    ) -> None:
        """Render persisted Data Groups and assignment counts read-only."""

        self._data_group_inventory_table.clearContents()
        self._data_group_inventory_table.setRowCount(0)
        resolved_path = (
            Path(project_path).expanduser().resolve()
            if project_path is not None
            else self._active_project_path
        )
        if resolved_path is None:
            self._data_groups_empty_state.setVisible(True)
            return
        try:
            groups = load_data_groups(resolved_path)
            assignments = load_image_data_group_assignments(resolved_path)
        except Exception as error:
            self._data_groups_empty_state.setVisible(True)
            self.statusBar().showMessage(f"Data Group inventory unavailable: {error}")
            return

        counts = {group.data_group_id: 0 for group in groups}
        for assignment in assignments:
            if assignment.data_group_id in counts:
                counts[assignment.data_group_id] += 1
        self._data_group_inventory_table.setRowCount(len(groups))
        for row, group in enumerate(groups):
            group_item = QTableWidgetItem(f"{group.data_group_id}\n{group.name}")
            group_item.setData(Qt.ItemDataRole.UserRole, group.data_group_id)
            count_item = QTableWidgetItem(str(counts[group.data_group_id]))
            self._data_group_inventory_table.setItem(row, 0, group_item)
            self._data_group_inventory_table.setItem(row, 1, count_item)
        self._data_groups_empty_state.setVisible(not groups)

    def _create_data_group(self) -> None:
        """Prompt for and persist one new Data Group without other mutations."""

        if self._active_project_path is None:
            self.statusBar().showMessage("Create Data Group unavailable: no active project.")
            return
        data_group_id, accepted = QInputDialog.getText(
            self,
            "Create Data Group",
            "Data Group ID:",
        )
        if not accepted:
            self.statusBar().showMessage("Create Data Group cancelled.")
            return
        normalized_id = data_group_id.strip()
        if not normalized_id:
            self.statusBar().showMessage("Data Group ID must not be empty.")
            return
        data_group_name, accepted = QInputDialog.getText(
            self,
            "Create Data Group",
            "Data Group name:",
        )
        if not accepted:
            self.statusBar().showMessage("Create Data Group cancelled.")
            return
        normalized_name = data_group_name.strip()
        if not normalized_name:
            self.statusBar().showMessage("Data Group name must not be empty.")
            return

        try:
            existing = load_data_groups(self._active_project_path)
        except Exception as error:
            self.statusBar().showMessage(f"Unable to create Data Group: {error}")
            return
        if any(group.data_group_id == normalized_id for group in existing):
            self.statusBar().showMessage(
                f"Data Group already exists: {normalized_id}."
            )
            return

        next_order = max((group.order for group in existing), default=-1) + 1
        candidate = DataGroup(normalized_id, normalized_name, next_order)
        try:
            ensure_data_group_schema(self._active_project_path)
            save_data_groups(self._active_project_path, (candidate,))
        except Exception as error:
            self.statusBar().showMessage(f"Unable to create Data Group: {error}")
            return
        self._refresh_data_group_inventory(self._active_project_path)
        self.statusBar().showMessage(f"Data Group created: {normalized_id}.")

    def _refresh_image_inventory(self, project_path: str | Path | None = None) -> None:
        """Render persisted image metadata without changing any source or project data."""

        self._image_inventory_table.clearContents()
        self._image_inventory_table.setRowCount(0)
        resolved_path = (
            Path(project_path).expanduser().resolve()
            if project_path is not None
            else self._active_project_path
        )
        if resolved_path is None:
            return
        try:
            reopened_assets = image_asset.load_image_assets(resolved_path)
        except Exception as error:
            self.statusBar().showMessage(f"Image inventory unavailable: {error}")
            return

        self._image_inventory_table.setRowCount(len(reopened_assets))
        health_labels = {
            SourceHealth.AVAILABLE: "Available",
            SourceHealth.MISSING: "Missing Source",
            SourceHealth.CHANGED: "Changed Source",
        }
        for row, reopened in enumerate(reopened_assets):
            asset = reopened.asset
            image_item = QTableWidgetItem(f"{asset.path.name}\n{asset.path}")
            image_item.setData(Qt.ItemDataRole.UserRole, asset.image_asset_id)
            values = (
                image_item,
                QTableWidgetItem(f"{asset.width} × {asset.height} px"),
                QTableWidgetItem(
                    f"{asset.dtype} · {asset.format}"
                    + (" — Lossy source" if asset.lossy_source else "")
                ),
                QTableWidgetItem(health_labels[reopened.source_health]),
                QTableWidgetItem(""),
            )
            for column, item in enumerate(values):
                self._image_inventory_table.setItem(row, column, item)

    def _on_image_inventory_selection_changed(self) -> None:
        """Preview the selected source only after the persisted health check."""

        if self._active_project_path is None:
            return
        row = self._image_inventory_table.currentRow()
        if row < 0:
            return
        item = self._image_inventory_table.item(row, 0)
        image_asset_id = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if not image_asset_id:
            return
        try:
            reopened_assets = image_asset.load_image_assets(self._active_project_path)
        except Exception as error:
            self.statusBar().showMessage(f"Image inventory unavailable: {error}")
            return
        reopened = next(
            (
                candidate
                for candidate in reopened_assets
                if candidate.asset.image_asset_id == str(image_asset_id)
            ),
            None,
        )
        if reopened is None:
            self.statusBar().showMessage(f"Wafer Image unavailable: {image_asset_id}")
            return

        asset = reopened.asset
        health_labels = {
            SourceHealth.AVAILABLE: "Available",
            SourceHealth.MISSING: "Missing Source",
            SourceHealth.CHANGED: "Changed Source",
        }
        lossy = " — Lossy source" if asset.lossy_source else ""
        self._image_inspector_empty_state.setText(
            f"Filename: {asset.path.name}\n"
            f"Path: {asset.path}\n"
            f"Native dimensions: {asset.width} × {asset.height} px\n"
            f"Dtype/format: {asset.dtype} · {asset.format}{lossy}\n"
            f"Source health: {health_labels[reopened.source_health]}"
        )
        if reopened.source_health is SourceHealth.MISSING:
            self.statusBar().showMessage(f"Missing Source: {asset.path}")
            return
        if reopened.source_health is SourceHealth.CHANGED:
            self.statusBar().showMessage(f"Changed Source: {asset.path}")
            return
        self.load_wafer_image(reopened.asset)

    def _restore_workspace_context(self) -> None:
        """Restore persisted shell state while keeping invalid sources visible."""

        saved_project = self._settings.value(self._ACTIVE_PROJECT_KEY)
        if not saved_project:
            return
        try:
            project_info = project.open_project(str(saved_project))
        except Exception as error:
            for key in (
                self._ACTIVE_PROJECT_KEY,
                self._CURRENT_WORKSPACE_KEY,
                self._CURRENT_IMAGE_ASSET_KEY,
            ):
                self._settings.remove(key)
            self._settings.sync()
            self.statusBar().showMessage(f"No active project — Saved project unavailable: {error}")
            return

        self._activate_project(project_info, "Project restored")
        saved_workspace = self._settings.value(self._CURRENT_WORKSPACE_KEY)
        if saved_workspace in self.workspace_actions and saved_workspace != self._current_workspace:
            self._set_workspace(str(saved_workspace))

        if project_info.schema_version >= project._JOB_SCHEMA_VERSION:
            self.configure_jobs_for_project(
                project_info.path,
                now=datetime.now(timezone.utc),
            )

        saved_image_id = self._settings.value(self._CURRENT_IMAGE_ASSET_KEY)
        if not saved_image_id:
            return
        try:
            assets = image_asset.load_image_assets(project_info.path)
        except Exception as error:
            self.statusBar().showMessage(f"Saved image unavailable: {error}")
            return
        reopened = next(
            (
                candidate
                for candidate in assets
                if candidate.asset.image_asset_id == str(saved_image_id)
            ),
            None,
        )
        if reopened is None:
            self.statusBar().showMessage(f"Saved image unavailable: {saved_image_id}")
            return
        if reopened.source_health is SourceHealth.MISSING:
            self.statusBar().showMessage(f"Missing Source: {reopened.asset.path}")
            return
        if reopened.source_health is SourceHealth.CHANGED:
            self.statusBar().showMessage(f"Changed Source: {reopened.asset.path}")
            return
        self.load_wafer_image(reopened.asset)

    def _create_project(self) -> None:
        selected_path = QFileDialog.getExistingDirectory(self, "Create Project")
        if not selected_path:
            return
        try:
            project_info = project.create_project(selected_path)
        except Exception as error:
            self.statusBar().showMessage(f"Create Project failed: {error}")
            return
        self._activate_project(project_info, "Project created")

    def _open_project(self) -> None:
        selected_path = QFileDialog.getExistingDirectory(self, "Open Project")
        if not selected_path:
            return
        try:
            project_info = project.open_project(selected_path)
        except Exception as error:
            self.statusBar().showMessage(f"Open Project failed: {error}")
            return
        self._activate_project(project_info, "Project opened")

    def _open_recent_project(self, item: QListWidgetItem) -> None:
        selected_path = item.data(Qt.ItemDataRole.UserRole) or item.text()
        try:
            project_info = project.open_project(selected_path)
        except Exception as error:
            self.statusBar().showMessage(f"Open Project failed: {error}")
            return
        self._activate_project(project_info, "Project opened")

    def _import_wafer_image(self) -> None:
        if self._active_project_path is None:
            return
        selected_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Import Wafer Image",
        )
        if not selected_path:
            return
        try:
            asset = image_asset.register_wafer_image(
                self._active_project_path,
                selected_path,
            )
        except Exception as error:
            self.statusBar().showMessage(f"Import Wafer Image failed: {error}")
            return
        self.load_wafer_image(asset)
        self._refresh_image_inventory(self._active_project_path)
        self._refresh_project_hub()

    def show_wafer_image(self, asset: ImageAsset) -> LoadedWaferImage:
        """Decode *asset*, retain native pixels, and show one fitted pixmap."""

        self._latest_load_token += 1
        loaded, image = _decode_wafer_image(asset.path)
        self._project_hub_dock.hide()
        self._loaded_wafer_image = loaded
        self._image_view._set_loaded_image(loaded, image)
        self._set_current_image_asset(asset)
        QTimer.singleShot(0, self._image_view._fit_image)
        return loaded

    def set_grid_profile(self, project_path: str | Path, profile: GridProfile) -> None:
        """Bind the controls and overlay to the persisted latest profile."""

        if not isinstance(profile, GridProfile):
            raise GridProfileConflictError("profile is not a persisted GridProfile")
        resolved_path = Path(project_path).expanduser().resolve()
        persisted = [
            candidate
            for candidate in load_grid_profiles(resolved_path)
            if candidate.grid_profile_id == profile.grid_profile_id
        ]
        if not persisted or persisted[-1] != profile:
            raise GridProfileConflictError("profile is not the persisted latest version")
        self._grid_project_path = resolved_path
        self._grid_profile = profile
        defect_classes = load_defect_classes(resolved_path)
        self._annotation_controls.set_classes(defect_classes)
        self._annotation_tool_dock.setEnabled(bool(defect_classes))
        self._annotation_tool_dock.setVisible(bool(defect_classes))
        self._grid_controls.bind(profile)
        self._grid_profile_dock.setEnabled(True)
        self._grid_profile_dock.show()
        self._bind_grid_for_current_image()

    def set_annotation_mode(self, mode: AnnotationMode | str) -> None:
        """Select the visible canvas tool."""

        self._image_view.set_tool_mode(mode)

    def set_selected_defect_classes(self, codes) -> tuple[str, ...]:
        """Select the complete Defect Class set used by Annotate/Paint/Erase."""

        self._annotation_controls.set_selected_classes(codes)
        return self._image_view.set_selected_class_codes(codes)

    def set_class_key(self, code: str, key: str) -> None:
        """Configure a keyboard key for toggling one Defect Class."""

        self._image_view.set_class_key(code, key)

    def set_defect_classes(self, classes) -> None:
        """Bind Defect Class checkboxes without requiring a project reopen."""

        self._annotation_controls.set_classes(tuple(classes))

    def configure_dataset_snapshot(
        self,
        groups: tuple[DataGroup, ...],
        classes: tuple[DefectClass, ...],
        preview: DatasetPreview,
        creator: SnapshotCreator,
    ) -> None:
        """Expose a prepared Training Scope preview and asynchronous creator."""

        self._training_scope_controls.configure(groups, classes, preview, creator)
        self._training_scope_dock.show()

    def configure_training(
        self,
        request: TrainingRequestSource,
        *,
        launcher: TrainingLauncher | None = None,
        clone_callback: CloneCallback | None = None,
    ) -> None:
        """Show background Training controls bound to a value-only worker request."""

        self._training_controls.configure(
            request,
            launcher=launcher,
            clone_callback=clone_callback,
        )
        self._training_dock.setEnabled(True)
        self._training_dock.show()

    def configure_evaluation(
        self,
        evaluation,
        decisions=(),
        *,
        decision_service: DecisionService | None = None,
        approval_callback: DecisionService | None = None,
    ) -> None:
        """Show Grid Evaluation review controls using value objects and a service seam."""

        if decision_service is not None and approval_callback is not None:
            raise ValueError("pass either decision_service or approval_callback, not both")
        self._evaluation_controls.configure(
            evaluation,
            decisions,
            decision_service=decision_service or approval_callback,
        )
        self._evaluation_dock.setEnabled(True)
        self._evaluation_dock.show()

    def configure_detection(
        self,
        artifact=None,
        request: DetectionRequestSource | None = None,
        *,
        launcher: DetectionLauncher | None = None,
        request_source: DetectionRequestSource | None = None,
    ) -> None:
        """Show native-coordinate Detection layers using value-only services."""

        self._detection_controls.configure(
            artifact,
            request,
            launcher=launcher,
            request_source=request_source,
        )
        self._detection_dock.setEnabled(True)
        self._detection_dock.show()

    def set_detection_artifact(self, artifact) -> None:
        """Refresh the displayed Detection map without touching project state."""

        self._detection_controls.set_artifact(artifact)
        self._detection_dock.setEnabled(artifact is not None)
        self._detection_dock.setVisible(artifact is not None)

    def configure_proposal_review(self, items, review_callback: ReviewCallback | None) -> None:
        """Show pure Review Queue values with an injected append-only callback."""

        self._proposal_review_controls.configure(items, review_callback)
        self._proposal_review_dock.setEnabled(True)
        self._proposal_review_dock.show()

    def configure_proposal_conversion(
        self,
        preview,
        confirmation_callback: ConversionCallback | None,
    ) -> None:
        """Show a conversion preview with an explicit callback confirmation gate."""

        self._proposal_conversion_controls.configure(preview, confirmation_callback)
        self._proposal_conversion_dock.setEnabled(True)
        self._proposal_conversion_dock.show()

    def configure_result_export(
        self,
        source_image,
        rows,
        selected_class: str,
        *,
        confidence_map=None,
        grid_rects=(),
        export_callback: ExportCallback | None = None,
    ) -> None:
        """Show explicit CSV/JSON/PNG destinations with an injected callback."""

        self._result_export_controls.configure(
            source_image,
            rows,
            selected_class,
            confidence_map=confidence_map,
            grid_rects=grid_rects,
            export_callback=export_callback,
        )
        self._result_export_dock.setEnabled(True)
        self._result_export_dock.show()

    def configure_jobs(self, jobs, action_callback: JobsActionCallback | None = None) -> None:
        """Show immutable Job snapshots with an injected nonmodal action seam."""

        self._jobs_controls.configure(jobs, action_callback=action_callback)
        self._jobs_dock.setEnabled(True)
        self._jobs_dock.show()

    def configure_jobs_for_project(
        self,
        project_path,
        *,
        now,
        stale_after_seconds: int | float = 60,
        process_is_alive=None,
        action_callback: JobsActionCallback | None = None,
    ) -> None:
        """Recover stale persisted jobs before presenting the nonmodal panel."""

        try:
            recover_stale_jobs(
                project_path,
                now=now,
                stale_after_seconds=stale_after_seconds,
                process_is_alive=process_is_alive,
            )
            jobs = list_jobs(project_path)
        except Exception as error:
            self._jobs_controls.configure((), action_callback=action_callback)
            self._jobs_controls.status_label.setText(f"Jobs recovery failed: {error}")
            self._jobs_dock.setEnabled(True)
            self._jobs_dock.show()
            return
        self.configure_jobs(jobs, action_callback=action_callback)

    def start_detection(self) -> None:
        """Start the configured Detection worker without blocking the GUI."""

        self._detection_controls.start_detection()

    def cancel_detection(self) -> None:
        """Request cancellation; terminal status still comes from the worker."""

        self._detection_controls.cancel_detection()

    def start_training(self) -> None:
        """Start the configured training request without blocking the GUI."""

        self._training_controls.start_training()

    def set_effective_wafer_area(self, area: EffectiveWaferArea | None) -> None:
        """Show the area for the current image and refresh derived participation."""

        if area is not None and not isinstance(area, EffectiveWaferArea):
            raise ValueError("area must be an EffectiveWaferArea or None")
        if (
            area is not None
            and self._current_image_asset is not None
            and area.image_asset_id != self._current_image_asset.image_asset_id
        ):
            area = None
        self._effective_wafer_area = area
        self._image_view.set_effective_wafer_area(area)
        self._refresh_participating_grid_count()
        self._refresh_effective_area_confirmation()
        self._refresh_review_controls()

    def _on_grid_draft_changed(self) -> None:
        profile = self._grid_profile
        if profile is None:
            return
        self._grid_controls.apply_button.setEnabled(
            self._grid_controls.draft_dimensions() != (profile.cell_width, profile.cell_height)
        )

    def _on_annotation_class_selection_changed(self, codes) -> None:
        self._image_view.set_selected_class_codes(codes)

    def _save_annotation_change(self, row: int, column: int, class_codes) -> None:
        if self._grid_project_path is None or self._current_image_asset is None:
            return
        annotation = GridAnnotation(
            self._current_image_asset.image_asset_id,
            row,
            column,
            tuple(class_codes),
        )
        previous = self._image_view.annotations.get((row, column), ())
        if not self._autosave_guard.autosave(annotation, previous):
            self._show_autosave_failure()
            return
        self._refresh_review_controls()

    def _persist_annotation(self, annotation: GridAnnotation) -> None:
        if self._grid_project_path is None:
            raise RuntimeError("No project is open")
        save_grid_annotation(self._grid_project_path, annotation)

    def _persist_annotation_as(self, target_path: Path, annotation: GridAnnotation) -> None:
        save_grid_annotation(target_path, annotation)

    def _restore_annotation(
        self, annotation: GridAnnotation, previous_class_codes: tuple[str, ...]
    ) -> None:
        annotations = self._image_view.annotations
        key = (annotation.row, annotation.column)
        if previous_class_codes:
            annotations[key] = previous_class_codes
        else:
            annotations.pop(key, None)
        self._image_view.set_annotations(annotations)

    def _apply_annotation(self, annotation: GridAnnotation) -> None:
        annotations = self._image_view.annotations
        annotations[(annotation.row, annotation.column)] = annotation.class_codes
        self._image_view.set_annotations(annotations)

    def _show_autosave_failure(self) -> None:
        failure = self._autosave_guard.pending
        if failure is None:
            return
        self._autosave_failure_controls.set_failure(failure)
        self._autosave_failure_dock.setEnabled(True)
        self._autosave_failure_dock.show()
        self.statusBar().showMessage(
            "Save failed: choose Retry, Save As, or Cancel. "
            f"{failure.error_message}"
        )

    def _clear_autosave_failure(self, message: str) -> None:
        self._autosave_failure_dock.setEnabled(False)
        self._autosave_failure_dock.hide()
        self.statusBar().showMessage(message)

    def _retry_autosave(self) -> None:
        if self._autosave_guard.retry():
            self._clear_autosave_failure("Annotation saved")
            self._refresh_review_controls()
            return
        self._show_autosave_failure()

    def _save_as_autosave(self) -> None:
        if self._grid_project_path is None:
            self.statusBar().showMessage("Save As unavailable: no project is open")
            return
        if self._autosave_guard.save_as(self._grid_project_path):
            self._clear_autosave_failure("Annotation saved")
            self._refresh_review_controls()
            return
        self._show_autosave_failure()

    def _cancel_autosave(self) -> None:
        if self._autosave_guard.cancel():
            self._clear_autosave_failure("Annotation save cancelled")

    def _apply_grid_profile(self) -> None:
        if self._grid_project_path is None or self._grid_profile is None:
            return
        width, height = self._grid_controls.draft_dimensions()
        profile = save_grid_profile(
            self._grid_project_path,
            width,
            height,
            previous=self._grid_profile,
        )
        self.set_grid_profile(self._grid_project_path, profile)

    def _set_current_image_asset(self, asset: ImageAsset) -> None:
        self._current_image_asset = asset
        self._settings.setValue(self._CURRENT_IMAGE_ASSET_KEY, asset.image_asset_id)
        self._settings.sync()
        self._bind_grid_for_current_image()

    def _bind_grid_for_current_image(self) -> None:
        profile = self._grid_profile
        if profile is None:
            self._grid_origin_dock.setEnabled(False)
            self._grid_origin_dock.hide()
            self._bind_effective_area_for_current_image()
            return
        origin_x = 0
        origin_y = 0
        asset = self._current_image_asset
        if asset is not None and self._grid_project_path is not None:
            placement = load_image_grid_placement(
                self._grid_project_path,
                asset.image_asset_id,
            )
            if (
                placement is not None
                and placement.grid_profile_id == profile.grid_profile_id
                and placement.grid_profile_version == profile.version
            ):
                origin_x = placement.origin_x
                origin_y = placement.origin_y
        self._image_view.set_annotation_grid(profile, QPoint(origin_x, origin_y))
        if asset is None:
            self._grid_origin_dock.setEnabled(False)
            self._grid_origin_dock.hide()
            self._bind_effective_area_for_current_image()
            return
        self._grid_origin = (origin_x, origin_y)
        self._grid_origin_controls.bind(origin_x, origin_y, profile.cell_width, profile.cell_height)
        self._grid_origin_dock.setEnabled(True)
        self._grid_origin_dock.show()
        self._bind_effective_area_for_current_image()

    def _bind_effective_area_for_current_image(self) -> None:
        asset = self._current_image_asset
        area = None
        if asset is not None and self._grid_project_path is not None:
            area = load_effective_wafer_area(self._grid_project_path, asset.image_asset_id)
        elif asset is not None:
            view_area = self._image_view._effective_wafer_area
            if view_area is not None and view_area.image_asset_id == asset.image_asset_id:
                area = view_area
        if area is not None and asset is not None and area.image_asset_id != asset.image_asset_id:
            area = None
        self._effective_wafer_area = area
        self._image_view.set_effective_wafer_area(area)
        self._refresh_participating_grid_count()
        self._refresh_effective_area_confirmation()
        self._refresh_review_controls()

    def _refresh_participating_grid_count(self) -> None:
        loaded = self._loaded_wafer_image
        profile = self._grid_profile
        area = self._effective_wafer_area
        if loaded is None or profile is None or area is None:
            self._grid_origin_controls.set_participating_count(0)
            return
        grids = annotation_grids(
            loaded.width,
            loaded.height,
            profile.cell_width,
            profile.cell_height,
            self._grid_origin[0],
            self._grid_origin[1],
        )
        count = len(participating_annotation_grids(grids, area))
        self._grid_origin_controls.set_participating_count(count)

    def _refresh_effective_area_confirmation(self) -> None:
        area = self._effective_wafer_area
        self._grid_origin_controls.set_confirmation_state(
            area is not None,
            area.confirmed if area is not None else False,
        )

    def _refresh_review_controls(self) -> None:
        asset = self._current_image_asset
        if self._grid_project_path is None or asset is None or self._grid_profile is None:
            self._review_controls.set_counts(None)
            self._review_controls.set_review_state(False, False)
            self._review_dock.setEnabled(False)
            self._review_dock.hide()
            return

        self._review_dock.setEnabled(True)
        self._review_dock.show()
        try:
            state = load_review_state(self._grid_project_path, asset.image_asset_id)
        except ReviewError as error:
            self._review_controls.set_counts(None)
            self._review_controls.set_review_state(False, False)
            self.statusBar().showMessage(f"Review error: {error}")
            return
        self._review_controls.set_review_state(state.reviewed, True)
        try:
            counts = load_review_counts(self._grid_project_path, asset.image_asset_id)
        except ReviewError as error:
            self._review_controls.set_counts(None)
            self.statusBar().showMessage(f"Review error: {error}")
            return
        self._review_controls.set_counts(counts)

    def _mark_current_image_reviewed(self) -> None:
        if self._grid_project_path is None or self._current_image_asset is None:
            return
        try:
            mark_image_reviewed(
                self._grid_project_path,
                self._current_image_asset.image_asset_id,
            )
        except ReviewError as error:
            self.statusBar().showMessage(f"Review error: {error}")
            self._refresh_review_controls()
            return
        self.statusBar().showMessage("Image Reviewed")
        self._refresh_review_controls()

    def _reopen_current_image(self) -> None:
        if self._grid_project_path is None or self._current_image_asset is None:
            return
        try:
            reopen_image(
                self._grid_project_path,
                self._current_image_asset.image_asset_id,
            )
        except ReviewError as error:
            self.statusBar().showMessage(f"Review error: {error}")
            self._refresh_review_controls()
            return
        self.statusBar().showMessage("Image Reopened")
        self._refresh_review_controls()

    def _confirm_effective_wafer_area(self) -> None:
        if (
            self._grid_project_path is None
            or self._current_image_asset is None
            or self._effective_wafer_area is None
            or self._effective_wafer_area.confirmed
        ):
            return
        area = confirm_effective_wafer_area(
            self._grid_project_path,
            self._current_image_asset.image_asset_id,
        )
        self.set_effective_wafer_area(area)

    def _on_grid_origin_draft_changed(self) -> None:
        if self._grid_profile is None or self._current_image_asset is None:
            return
        self._grid_origin_controls.apply_button.setEnabled(
            self._grid_origin_controls.draft_origin() != self._grid_origin
        )

    def _apply_grid_origin(self) -> None:
        if (
            self._grid_project_path is None
            or self._grid_profile is None
            or self._current_image_asset is None
        ):
            return
        origin_x, origin_y = self._grid_origin_controls.draft_origin()
        set_image_grid_origin(
            self._grid_project_path,
            self._current_image_asset.image_asset_id,
            self._grid_profile.grid_profile_id,
            origin_x,
            origin_y,
        )
        self._bind_grid_for_current_image()

    def load_wafer_image(self, selection: ImageAsset | ReopenedWaferImage) -> None:
        """Start decoding *asset* without blocking the GUI thread."""

        asset = selection.asset if isinstance(selection, ReopenedWaferImage) else selection
        if not self._autosave_guard.request_image_switch(asset.image_asset_id):
            self._show_autosave_failure()
            return
        self._latest_load_token += 1
        health = _source_health(asset)
        if health is SourceHealth.MISSING:
            self._latest_lossy_source = False
            self._latest_load_token = -1
            self.statusBar().showMessage("Missing Source")
            return
        if health is SourceHealth.CHANGED:
            self._latest_lossy_source = False
            self._latest_load_token = -1
            self.statusBar().showMessage("Changed Source")
            return
        self._latest_lossy_source = asset.lossy_source
        loading_status = "Loading - Lossy JPEG Source" if self._latest_lossy_source else "Loading"
        self.statusBar().showMessage(loading_status)
        self._latest_load_token = self._wafer_loader.request(asset.path)
        self._pending_image_assets[self._latest_load_token] = asset

    def _on_load_ready(self, token: int, loaded: LoadedWaferImage, image) -> None:
        asset = self._pending_image_assets.pop(token, None)
        if token != self._latest_load_token:
            return
        self._project_hub_dock.hide()
        self._loaded_wafer_image = loaded
        self._image_view._set_loaded_image(loaded, image)
        if asset is not None:
            self._set_current_image_asset(asset)
        QTimer.singleShot(0, self._image_view._fit_image)
        ready_status = "Ready - Lossy JPEG Source" if self._latest_lossy_source else "Ready"
        self.statusBar().showMessage(ready_status)

    def _on_load_error(self, token: int, message: str) -> None:
        self._pending_image_assets.pop(token, None)
        if token != self._latest_load_token:
            return
        self.statusBar().showMessage(f"Error: {message}")
