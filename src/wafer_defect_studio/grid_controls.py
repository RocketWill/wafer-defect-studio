"""Widgets for editing a draft Grid Profile."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFormLayout, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget

from .grid_profile import GridProfile


class GridProfileControls(QWidget):
    """Exact pixel fields and a square-size convenience slider."""

    draftChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.width_spin = QSpinBox(self)
        self.width_spin.setObjectName("gridWidthSpinBox")
        self.height_spin = QSpinBox(self)
        self.height_spin.setObjectName("gridHeightSpinBox")
        for spin in (self.width_spin, self.height_spin):
            spin.setRange(1, 100000)

        self.size_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.size_slider.setObjectName("gridSizeSlider")
        self.size_slider.setRange(1, 100000)
        self.size_slider.setSingleStep(1)
        self.size_slider.setPageStep(16)

        self.apply_button = QPushButton("Apply", self)
        self.apply_button.setObjectName("applyGridProfileButton")
        self.apply_button.setEnabled(False)

        form = QFormLayout()
        form.addRow("Cell width (px)", self.width_spin)
        form.addRow("Cell height (px)", self.height_spin)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.size_slider)
        layout.addWidget(self.apply_button)

        self.size_slider.valueChanged.connect(self._set_square_size)
        self.width_spin.valueChanged.connect(lambda _value: self.draftChanged.emit())
        self.height_spin.valueChanged.connect(lambda _value: self.draftChanged.emit())

    def bind(self, profile: GridProfile) -> None:
        widgets = (self.width_spin, self.height_spin, self.size_slider)
        previous = [widget.blockSignals(True) for widget in widgets]
        try:
            self.width_spin.setValue(profile.cell_width)
            self.height_spin.setValue(profile.cell_height)
            self.size_slider.setValue(profile.cell_width)
            self.apply_button.setEnabled(False)
        finally:
            for widget, was_blocked in zip(widgets, previous):
                widget.blockSignals(was_blocked)

    def draft_dimensions(self) -> tuple[int, int]:
        return self.width_spin.value(), self.height_spin.value()

    def _set_square_size(self, value: int) -> None:
        width_blocked = self.width_spin.blockSignals(True)
        height_blocked = self.height_spin.blockSignals(True)
        try:
            self.width_spin.setValue(value)
            self.height_spin.setValue(value)
        finally:
            self.width_spin.blockSignals(width_blocked)
            self.height_spin.blockSignals(height_blocked)
        self.draftChanged.emit()
