"""Source-aligned, zoom-aware annotation-grid drawing."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem

from .grid_geometry import annotation_grids
from .grid_profile import GridProfile


class _GridOverlayItem(QGraphicsItem):
    def __init__(self, width: int, height: int, profile: GridProfile, origin_x: int, origin_y: int) -> None:
        super().__init__()
        self._rect = QRectF(0, 0, width, height)
        self._profile = profile
        self._origin_x = origin_x
        self._origin_y = origin_y
        self.setZValue(1)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setAcceptHoverEvents(False)

    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        pen = QPen(QColor("#ff00ff"))
        pen.setCosmetic(True)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        width = self._rect.width()
        height = self._rect.height()
        painter.drawLine(QPointF(0, 0), QPointF(width, 0))
        painter.drawLine(QPointF(0, height), QPointF(width, height))
        painter.drawLine(QPointF(0, 0), QPointF(0, height))
        painter.drawLine(QPointF(width, 0), QPointF(width, height))

        level_of_detail = QStyleOptionGraphicsItem.levelOfDetailFromTransform(
            painter.worldTransform()
        )
        minimum_cell_pixels = min(self._profile.cell_width, self._profile.cell_height) * level_of_detail
        if minimum_cell_pixels < 4:
            painter.restore()
            return
        stride = 4 if minimum_cell_pixels < 16 else 1
        grids = annotation_grids(
            int(width),
            int(height),
            self._profile.cell_width,
            self._profile.cell_height,
            self._origin_x,
            self._origin_y,
        )
        columns = {
            (grid.column, grid.x)
            for grid in grids
            if 0 < grid.x < width and (stride == 1 or grid.column % stride == 0)
        }
        rows = {
            (grid.row, grid.y)
            for grid in grids
            if 0 < grid.y < height and (stride == 1 or grid.row % stride == 0)
        }
        for _, x in sorted(columns):
            painter.drawLine(QPointF(x, 0), QPointF(x, height))
        for _, y in sorted(rows):
            painter.drawLine(QPointF(0, y), QPointF(width, y))
        painter.restore()
