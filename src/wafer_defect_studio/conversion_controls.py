"""Small Widgets seam for explicitly confirming a proposal conversion."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtWidgets import QCheckBox, QLabel, QPushButton, QVBoxLayout, QWidget

from .proposal_conversion import ConversionPreview


ConversionCallback = Callable[[ConversionPreview], Any]


class ProposalConversionControls(QWidget):
    """Render a value-only conversion preview and gate an injected callback."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("proposalConversionControls")
        self._preview: ConversionPreview | None = None
        self._confirmation_callback: ConversionCallback | None = None

        self.preview_label = QLabel("No conversion preview configured.", self)
        self.preview_label.setObjectName("proposalConversionPreviewLabel")
        self.preview_label.setWordWrap(True)
        self.confirmation_checkbox = QCheckBox(
            "I confirm these proposals should become Grid Annotations", self
        )
        self.confirmation_checkbox.setObjectName("proposalConversionConfirmCheckBox")
        self.convert_button = QPushButton("Convert to Grid Annotations", self)
        self.convert_button.setObjectName("convertToGridAnnotationsButton")
        self.status_label = QLabel("Conversion not confirmed.", self)
        self.status_label.setObjectName("proposalConversionStatusLabel")
        self.status_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.preview_label)
        layout.addWidget(self.confirmation_checkbox)
        layout.addWidget(self.convert_button)
        layout.addWidget(self.status_label)

        self.confirmation_checkbox.toggled.connect(self._refresh_gate)
        self.convert_button.clicked.connect(self._submit)
        self._refresh_gate(False)

    @property
    def preview(self) -> ConversionPreview | None:
        """Return the currently displayed immutable preview."""

        return self._preview

    def configure(
        self,
        preview: ConversionPreview,
        confirmation_callback: ConversionCallback | None,
    ) -> None:
        """Bind a preview and a callback without opening SQLite in the widget."""

        if not isinstance(preview, ConversionPreview):
            raise TypeError("preview must be a ConversionPreview")
        if confirmation_callback is not None and not callable(confirmation_callback):
            raise TypeError("confirmation_callback must be callable or None")
        self._preview = preview
        self._confirmation_callback = confirmation_callback
        self.confirmation_checkbox.setChecked(False)
        self.preview_label.setText(_format_preview(preview))
        self.status_label.setText("Conversion not confirmed.")
        self._refresh_gate(False)

    def _refresh_gate(self, checked: bool) -> None:
        self.convert_button.setEnabled(
            bool(checked and self._preview is not None and self._confirmation_callback is not None)
        )

    def _submit(self) -> None:
        preview = self._preview
        callback = self._confirmation_callback
        if preview is None or callback is None or not self.confirmation_checkbox.isChecked():
            self.status_label.setText("Check the confirmation box and configure a callback first.")
            self._refresh_gate(self.confirmation_checkbox.isChecked())
            return
        try:
            callback(preview)
        except Exception as error:  # UI seam reports callback failures without owning storage.
            self.status_label.setText(f"Conversion failed: {error}")
            return
        self.status_label.setText("Converted to Grid Annotations.")


def _format_preview(preview: ConversionPreview) -> str:
    lines = [f"Affected cells: {len(preview.cells)}"]
    for cell in preview.cells:
        lines.append(
            f"({cell.row}, {cell.column}) — classes: {', '.join(cell.class_codes)}; "
            f"proposals: {', '.join(cell.proposal_ids)}"
        )
    lines.append(f"Proposal IDs: {', '.join(preview.source_proposal_ids)}")
    marker = preview.provenance.get("source_coordinate_system", "unknown")
    lines.append(f"Coordinates: {marker}")
    return "\n".join(lines)


# A compact generic name is useful to callers that do not need the review
# prefix, while the explicit class remains the documented API.
ConversionControls = ProposalConversionControls


__all__ = ["ConversionCallback", "ConversionControls", "ProposalConversionControls"]
