"""Main window for the initial application shell."""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow

from .image_asset import ImageAsset
from .wafer_view import LoadedWaferImage, WaferView, _decode_wafer_image


class MainWindow(QMainWindow):
    """Top-level window for Wafer Defect Studio."""

    def __init__(self, parent: QMainWindow | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Wafer Defect Studio")
        self._loaded_wafer_image: LoadedWaferImage | None = None
        self._image_view = WaferView()
        self.setCentralWidget(self._image_view)

    def show_wafer_image(self, asset: ImageAsset) -> LoadedWaferImage:
        """Decode *asset*, retain native pixels, and show one fitted pixmap."""

        loaded, image = _decode_wafer_image(asset.path)
        self._loaded_wafer_image = loaded
        self._image_view._set_loaded_image(loaded, image)
        return loaded
