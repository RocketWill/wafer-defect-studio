"""Decode a registered wafer image while preserving its native pixels."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import floor
from pathlib import Path
import sys

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QImage, QImageReader, QPixmap, QTransform
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


@dataclass(frozen=True)
class LoadedWaferImage:
    """Native grayscale pixels decoded for display and coordinate lookup."""

    width: int
    height: int
    dtype: str
    pixels: array


class WaferView(QGraphicsView):
    """Graphics view that maps viewport points to native wafer pixels."""

    _ZOOM_FACTOR = 1.25

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._loaded_wafer_image: LoadedWaferImage | None = None
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._space_pressed = False
        self._drag_mode_before_space = QGraphicsView.DragMode.NoDrag
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

    def source_pixel_at(self, viewport_position: QPoint) -> tuple[int, int, int] | None:
        """Return the native pixel under a viewport point, if it is in bounds."""

        loaded = self._loaded_wafer_image
        if loaded is None:
            return None
        scene_position = self.mapToScene(viewport_position)
        x = floor(scene_position.x())
        y = floor(scene_position.y())
        if x < 0 or y < 0 or x >= loaded.width or y >= loaded.height:
            return None
        return x, y, int(loaded.pixels[y * loaded.width + x])

    def _set_loaded_image(self, loaded: LoadedWaferImage, image: QImage) -> None:
        self._loaded_wafer_image = loaded
        pixmap = QPixmap.fromImage(image)
        target = self.viewport().size()
        if target.width() <= 0 or target.height() <= 0:
            target = QSize(800, 600)
        target = QSize(min(target.width(), loaded.width), min(target.height(), loaded.height))
        fitted = pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self._scene.clear()
        item = self._scene.addPixmap(fitted)
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        item.setTransform(
            QTransform.fromScale(loaded.width / fitted.width(), loaded.height / fitted.height())
        )
        self._pixmap_item = item
        self._scene.setSceneRect(QRectF(0, 0, loaded.width, loaded.height))
        self._fit_image()

    def _fit_image(self) -> None:
        if self._pixmap_item is None:
            return
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        if event.angleDelta().y() == 0 or self._pixmap_item is None:
            super().wheelEvent(event)
            return

        factor = self._ZOOM_FACTOR if event.angleDelta().y() > 0 else 1 / self._ZOOM_FACTOR
        viewport_position = event.position().toPoint()
        scene_position_before = self.mapToScene(viewport_position)
        anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.scale(factor, factor)
        scene_position_after = self.mapToScene(viewport_position)
        scene_delta = scene_position_after - scene_position_before
        self.translate(scene_delta.x(), scene_delta.y())
        self.setTransformationAnchor(anchor)
        event.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if not self._space_pressed:
                self._drag_mode_before_space = self.dragMode()
                self._space_pressed = True
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            self._restore_drag_mode()
            event.accept()
            return
        if event.key() == Qt.Key.Key_F:
            self._fit_image()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._restore_drag_mode()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _restore_drag_mode(self) -> None:
        if self._space_pressed:
            self._space_pressed = False
            self.setDragMode(self._drag_mode_before_space)


def _decode_wafer_image(path: Path) -> tuple[LoadedWaferImage, QImage]:
    reader = QImageReader(str(path))
    image = reader.read()
    if image.isNull():
        raise ValueError(f"Unable to decode wafer image: {path}")
    if image.format() != QImage.Format_Grayscale16:
        raise ValueError(f"Wafer image must be grayscale16: {path}")

    width = image.width()
    height = image.height()
    row_bytes = width * 2
    stride = image.bytesPerLine()
    source = image.constBits()
    pixels = array("H")
    for row_index in range(height):
        row = array("H")
        row.frombytes(bytes(source[row_index * stride : row_index * stride + row_bytes]))
        if sys.byteorder == "big":
            row.byteswap()
        pixels.extend(row)

    return LoadedWaferImage(width, height, "uint16", pixels), image
