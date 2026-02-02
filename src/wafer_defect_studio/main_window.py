"""Main window for the initial application shell."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .image_asset import ImageAsset, ReopenedWaferImage, SourceHealth, _source_health
from .annotation import GridAnnotation, save_grid_annotation
from .annotation_tools import AnnotationMode
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


class MainWindow(QMainWindow):
    """Top-level window for Wafer Defect Studio."""

    def __init__(self, parent: QMainWindow | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Wafer Defect Studio")
        self._loaded_wafer_image: LoadedWaferImage | None = None
        self._image_view = WaferView()
        self.setCentralWidget(self._image_view)
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

    def show_wafer_image(self, asset: ImageAsset) -> LoadedWaferImage:
        """Decode *asset*, retain native pixels, and show one fitted pixmap."""

        self._latest_load_token += 1
        loaded, image = _decode_wafer_image(asset.path)
        self._loaded_wafer_image = loaded
        self._image_view._set_loaded_image(loaded, image)
        self._set_current_image_asset(asset)
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
        self._annotation_controls.set_classes(load_defect_classes(resolved_path))
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
        save_grid_annotation(
            self._grid_project_path,
            GridAnnotation(
                self._current_image_asset.image_asset_id,
                row,
                column,
                tuple(class_codes),
            ),
        )

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
        self._loaded_wafer_image = loaded
        self._image_view._set_loaded_image(loaded, image)
        if asset is not None:
            self._set_current_image_asset(asset)
        ready_status = "Ready - Lossy JPEG Source" if self._latest_lossy_source else "Ready"
        self.statusBar().showMessage(ready_status)

    def _on_load_error(self, token: int, message: str) -> None:
        self._pending_image_assets.pop(token, None)
        if token != self._latest_load_token:
            return
        self.statusBar().showMessage(f"Error: {message}")
