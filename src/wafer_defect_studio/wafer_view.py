"""Decode a registered wafer image while preserving its native pixels."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import floor
from pathlib import Path
import sys

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QImage, QImageReader, QPixmap, QPolygonF, QTransform, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPolygonItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from .effective_area import EffectiveWaferArea, PolygonGeometry
from .grid_overlay import _GridOverlayItem
from .grid_profile import GridProfile


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
        self._annotation_grid_profile: GridProfile | None = None
        self._annotation_grid_origin = QPoint(0, 0)
        self._grid_overlay_item: _GridOverlayItem | None = None
        self._effective_wafer_area: EffectiveWaferArea | None = None
        self._effective_area_item: QGraphicsItem | None = None
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

    def set_annotation_grid(self, profile: GridProfile, origin: QPoint = QPoint(0, 0)) -> None:
        """Set the source-aligned annotation grid shown over the wafer image."""

        if not isinstance(profile, GridProfile):
            raise ValueError("profile must be a GridProfile")
        if not (0 <= origin.x() < profile.cell_width and 0 <= origin.y() < profile.cell_height):
            raise ValueError("origin must be canonical for the grid profile")
        self._annotation_grid_profile = profile
        self._annotation_grid_origin = QPoint(origin)
        self._rebuild_grid_overlay()

    def set_effective_wafer_area(self, area: EffectiveWaferArea | None) -> None:
        """Set or clear the source-aligned effective-area outline."""

        if area is not None and not isinstance(area, EffectiveWaferArea):
            raise ValueError("area must be an EffectiveWaferArea or None")
        if area is not None and area.shape not in ("ellipse", "polygon"):
            raise ValueError("unsupported effective wafer area shape")
        self._effective_wafer_area = area
        self._rebuild_effective_area_overlay()

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

        self._grid_overlay_item = None
        self._effective_area_item = None
        self._scene.clear()
        item = self._scene.addPixmap(fitted)
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        item.setTransform(
            QTransform.fromScale(loaded.width / fitted.width(), loaded.height / fitted.height())
        )
        self._pixmap_item = item
        self._scene.setSceneRect(QRectF(0, 0, loaded.width, loaded.height))
        self._rebuild_grid_overlay()
        self._rebuild_effective_area_overlay()
        self._fit_image()

    def _rebuild_grid_overlay(self) -> None:
        if self._grid_overlay_item is not None:
            self._scene.removeItem(self._grid_overlay_item)
            self._grid_overlay_item = None
        loaded = self._loaded_wafer_image
        profile = self._annotation_grid_profile
        if loaded is None or profile is None:
            return
        self._grid_overlay_item = _GridOverlayItem(
            loaded.width,
            loaded.height,
            profile,
            self._annotation_grid_origin.x(),
            self._annotation_grid_origin.y(),
        )
        self._scene.addItem(self._grid_overlay_item)
        self._scene.invalidate()
        self.viewport().update()

    def _rebuild_effective_area_overlay(self) -> None:
        if self._effective_area_item is not None:
            self._scene.removeItem(self._effective_area_item)
            self._effective_area_item = None
        area = self._effective_wafer_area
        if area is None:
            return
        geometry = area.geometry
        if area.shape == "ellipse":
            item: QGraphicsItem = QGraphicsEllipseItem(
                QRectF(
                    geometry.center_x - geometry.radius_x,
                    geometry.center_y - geometry.radius_y,
                    2 * geometry.radius_x,
                    2 * geometry.radius_y,
                )
            )
        elif area.shape == "polygon" and isinstance(geometry, PolygonGeometry):
            item = QGraphicsPolygonItem(
                QPolygonF([QPointF(point.x, point.y) for point in geometry.vertices])
            )
        else:
            raise ValueError("effective wafer area geometry does not match its shape")
        pen = QPen(QColor("#ffff00"))
        pen.setCosmetic(True)
        pen.setWidth(2)
        item.setPen(pen)
        item.setBrush(Qt.BrushStyle.NoBrush)
        item.setZValue(2)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        item.setAcceptHoverEvents(False)
        self._scene.addItem(item)
        self._effective_area_item = item
        self._scene.invalidate()
        self.viewport().update()

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
    if image.format() not in (
        QImage.Format_Grayscale8,
        QImage.Format_Grayscale16,
        QImage.Format_Indexed8,
    ):
        raise ValueError(f"Wafer image must be grayscale8 or grayscale16: {path}")

    display_image = image
    if image.format() == QImage.Format_Indexed8:
        display_image = image.convertToFormat(QImage.Format_Grayscale8)

    width = display_image.width()
    height = display_image.height()
    sixteen_bit = image.format() == QImage.Format_Grayscale16
    row_bytes = width * (2 if sixteen_bit else 1)
    stride = display_image.bytesPerLine()
    source = display_image.constBits()
    pixels = array("H" if sixteen_bit else "B")
    for row_index in range(height):
        row = array("H" if sixteen_bit else "B")
        row.frombytes(bytes(source[row_index * stride : row_index * stride + row_bytes]))
        if sixteen_bit and sys.byteorder == "big":
            row.byteswap()
        pixels.extend(row)

    return LoadedWaferImage(width, height, "uint16" if sixteen_bit else "uint8", pixels), display_image
