"""Widgets for explicitly selecting and publishing detection export files."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QFileDialog,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .detection_windows import Rect
from .result_export import ExportBundleResult, ReviewedProposalRow, export_result_bundle


ExportCallback = Callable[..., ExportBundleResult]


class ResultExportControls(QWidget):
    """Render destination controls and report an injected bundle callback."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("resultExportControls")
        self._source_image: QImage | None = None
        self._rows: tuple[ReviewedProposalRow, ...] = ()
        self._confidence_map: Any = None
        self._grid_rects: tuple[Any, ...] = ()
        self._export_callback: ExportCallback = export_result_bundle

        self.csv_path_edit = QLineEdit(self)
        self.csv_path_edit.setObjectName("resultExportCsvPathEdit")
        self.json_path_edit = QLineEdit(self)
        self.json_path_edit.setObjectName("resultExportJsonPathEdit")
        self.png_path_edit = QLineEdit(self)
        self.png_path_edit.setObjectName("resultExportPngPathEdit")
        self._path_edits = (
            self.csv_path_edit,
            self.json_path_edit,
            self.png_path_edit,
        )

        form = QFormLayout()
        form.addRow("CSV destination", self._path_row(self.csv_path_edit, "CSV"))
        form.addRow("JSON destination", self._path_row(self.json_path_edit, "JSON"))
        form.addRow("PNG destination", self._path_row(self.png_path_edit, "PNG"))

        self.selected_class_edit = QLineEdit(self)
        self.selected_class_edit.setObjectName("resultExportSelectedClassEdit")
        form.addRow("Selected class", self.selected_class_edit)

        self.overwrite_checkbox = QCheckBox(
            "I confirm overwriting existing export files", self
        )
        self.overwrite_checkbox.setObjectName("resultExportOverwriteCheckBox")
        self.export_button = QPushButton("Export Results", self)
        self.export_button.setObjectName("exportResultButton")
        self.status_label = QLabel("Choose CSV, JSON, and PNG destinations.", self)
        self.status_label.setObjectName("resultExportStatusLabel")
        self.status_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.overwrite_checkbox)
        layout.addWidget(self.export_button)
        layout.addWidget(self.status_label)

        for edit in (*self._path_edits, self.selected_class_edit):
            edit.textChanged.connect(self._refresh_gate)
        self.export_button.clicked.connect(self._submit)
        self._refresh_gate()

    def _path_row(self, edit: QLineEdit, kind: str) -> QWidget:
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse…", row)
        browse.setObjectName(f"browseResultExport{kind}Button")
        browse.clicked.connect(lambda _checked=False, target=edit, label=kind: self._browse(target, label))
        layout.addWidget(edit)
        layout.addWidget(browse)
        return row

    def configure(
        self,
        source_image: QImage,
        rows: Iterable[ReviewedProposalRow],
        selected_class: str,
        *,
        confidence_map: Any = None,
        grid_rects: Iterable[Rect | Sequence[int] | Mapping[str, Any]] = (),
        export_callback: ExportCallback | None = None,
    ) -> None:
        """Bind immutable render values and an optional injected export seam."""

        if not isinstance(source_image, QImage) or source_image.isNull():
            raise ValueError("source_image must be a non-empty QImage")
        if not isinstance(selected_class, str) or not selected_class.strip():
            raise ValueError("selected_class must be a non-empty string")
        if export_callback is not None and not callable(export_callback):
            raise TypeError("export_callback must be callable or None")
        values = tuple(rows)
        if any(not isinstance(row, ReviewedProposalRow) for row in values):
            raise TypeError("rows must contain ReviewedProposalRow values")
        self._source_image = source_image.copy()
        self._rows = values
        self._confidence_map = confidence_map
        self._grid_rects = tuple(grid_rects)
        self._export_callback = export_callback or export_result_bundle
        self.selected_class_edit.setText(selected_class)
        self.overwrite_checkbox.setChecked(False)
        self.status_label.setText("Choose CSV, JSON, and PNG destinations.")
        self._refresh_gate()

    def _browse(self, edit: QLineEdit, kind: str) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            self,
            f"Select {kind} destination",
            edit.text(),
            f"{kind} files (*.{kind.lower()});;All files (*)",
        )
        if path:
            edit.setText(path)

    def _refresh_gate(self, *_args: object) -> None:
        ready = (
            self._source_image is not None
            and all(edit.text().strip() for edit in self._path_edits)
            and bool(self.selected_class_edit.text().strip())
        )
        self.export_button.setEnabled(ready)

    def _submit(self) -> None:
        if not self.export_button.isEnabled():
            self.status_label.setText("Export failed: choose all destinations and a selected class.")
            return
        source_image = self._source_image
        callback = self._export_callback
        if source_image is None:
            self.status_label.setText("Export failed: no source image configured.")
            return
        try:
            outcome = callback(
                self.csv_path_edit.text().strip(),
                self.json_path_edit.text().strip(),
                self.png_path_edit.text().strip(),
                source_image,
                self._rows,
                selected_class=self.selected_class_edit.text().strip(),
                confidence_map=self._confidence_map,
                grid_rects=self._grid_rects,
                overwrite=self.overwrite_checkbox.isChecked(),
            )
        except Exception as error:  # callback owns persistence/error policy
            self.status_label.setText(f"Export failed: {error}")
            return
        success = outcome.success if isinstance(outcome, ExportBundleResult) else outcome is True
        if success:
            self.status_label.setText("Exported CSV, JSON, and PNG results.")
            return
        error = getattr(outcome, "error", None) or "export callback returned failure"
        self.status_label.setText(f"Export failed: {error}")


ExportControls = ResultExportControls


__all__ = ["ExportCallback", "ExportControls", "ResultExportControls"]
