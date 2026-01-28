"""Main window for the initial application shell."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QDockWidget, QMainWindow

from .image_asset import ImageAsset, ReopenedWaferImage, SourceHealth, _source_health
from .grid_controls import GridProfileControls
from .grid_profile import (
    GridProfile,
    GridProfileConflictError,
    load_grid_profiles,
    save_grid_profile,
)
from .wafer_loader import WaferLoader
from .wafer_view import LoadedWaferImage, WaferView, _decode_wafer_image


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
        self._grid_project_path: Path | None = None
        self._grid_profile: GridProfile | None = None
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

    def show_wafer_image(self, asset: ImageAsset) -> LoadedWaferImage:
        """Decode *asset*, retain native pixels, and show one fitted pixmap."""

        loaded, image = _decode_wafer_image(asset.path)
        self._loaded_wafer_image = loaded
        self._image_view._set_loaded_image(loaded, image)
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
        self._image_view.set_annotation_grid(profile, QPoint(0, 0))

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

    def _on_load_ready(self, token: int, loaded: LoadedWaferImage, image) -> None:
        if token != self._latest_load_token:
            return
        self._loaded_wafer_image = loaded
        self._image_view._set_loaded_image(loaded, image)
        ready_status = "Ready - Lossy JPEG Source" if self._latest_lossy_source else "Ready"
        self.statusBar().showMessage(ready_status)

    def _on_load_error(self, token: int, message: str) -> None:
        if token != self._latest_load_token:
            return
        self.statusBar().showMessage(f"Error: {message}")
