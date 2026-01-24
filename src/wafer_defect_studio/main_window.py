"""Main window for the initial application shell."""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow

from .image_asset import ImageAsset, ReopenedWaferImage, SourceHealth, _source_health
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
        self._load_threads = self._wafer_loader._threads

    def show_wafer_image(self, asset: ImageAsset) -> LoadedWaferImage:
        """Decode *asset*, retain native pixels, and show one fitted pixmap."""

        loaded, image = _decode_wafer_image(asset.path)
        self._loaded_wafer_image = loaded
        self._image_view._set_loaded_image(loaded, image)
        return loaded

    def load_wafer_image(self, selection: ImageAsset | ReopenedWaferImage) -> None:
        """Start decoding *asset* without blocking the GUI thread."""

        asset = selection.asset if isinstance(selection, ReopenedWaferImage) else selection
        self._latest_load_token += 1
        health = _source_health(asset)
        if health is SourceHealth.MISSING:
            self.statusBar().showMessage("Missing Source")
            return
        if health is SourceHealth.CHANGED:
            self.statusBar().showMessage("Changed Source")
            return
        self.statusBar().showMessage("Loading")
        self._latest_load_token = self._wafer_loader.request(asset.path)

    def _on_load_ready(self, token: int, loaded: LoadedWaferImage, image) -> None:
        if token != self._latest_load_token:
            return
        self._loaded_wafer_image = loaded
        self._image_view._set_loaded_image(loaded, image)
        self.statusBar().showMessage("Ready")

    def _on_load_error(self, token: int, message: str) -> None:
        if token != self._latest_load_token:
            return
        self.statusBar().showMessage(f"Error: {message}")
