"""Main window for the initial application shell."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QDockWidget,
    QFormLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .image_asset import ImageAsset, ReopenedWaferImage, SourceHealth, _source_health
from .effective_area import (
    EffectiveWaferArea,
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
        form = QFormLayout()
        form.addRow("Origin x (px)", self.origin_x_spin)
        form.addRow("Origin y (px)", self.origin_y_spin)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.apply_button)
        layout.addWidget(self.participating_count_label)
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


class MainWindow(QMainWindow):
    """Top-level window for Wafer Defect Studio."""

    def __init__(self, parent: QMainWindow | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Wafer Defect Studio")
        self._loaded_wafer_image: LoadedWaferImage | None = None
        self._image_view = WaferView()
        self.setCentralWidget(self._image_view)
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
        self._grid_controls.bind(profile)
        self._grid_profile_dock.setEnabled(True)
        self._grid_profile_dock.show()
        self._bind_grid_for_current_image()

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

    def _on_grid_draft_changed(self) -> None:
        profile = self._grid_profile
        if profile is None:
            return
        self._grid_controls.apply_button.setEnabled(
            self._grid_controls.draft_dimensions() != (profile.cell_width, profile.cell_height)
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
