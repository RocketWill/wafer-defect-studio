"""Main window for the initial application shell."""

from __future__ import annotations

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QMainWindow

from .image_asset import ImageAsset
from .wafer_view import LoadedWaferImage, _decode_wafer_image


class MainWindow(QMainWindow):
    """Top-level window for Wafer Defect Studio."""

    def __init__(self, parent: QMainWindow | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Wafer Defect Studio")
        self._loaded_wafer_image: LoadedWaferImage | None = None
        self._image_view = QGraphicsView()
        self._image_scene = QGraphicsScene(self)
        self._image_view.setScene(self._image_scene)
        self.setCentralWidget(self._image_view)

    def show_wafer_image(self, asset: ImageAsset) -> LoadedWaferImage:
        """Decode *asset*, retain native pixels, and show one fitted pixmap."""

        loaded, image = _decode_wafer_image(asset.path)
        self._loaded_wafer_image = loaded

        pixmap = QPixmap.fromImage(image)
        target = self._image_view.viewport().size()
        if target.width() <= 0 or target.height() <= 0:
            target = QSize(800, 600)
        target = QSize(min(target.width(), loaded.width), min(target.height(), loaded.height))
        fitted = pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_scene.clear()
        item = self._image_scene.addPixmap(fitted)
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._image_scene.setSceneRect(item.boundingRect())
        self._image_view.fitInView(item, Qt.AspectRatioMode.KeepAspectRatio)
        return loaded
