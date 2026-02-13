"""Small, value-only Widgets for Detection Review proposals.

The controls render an already-built Review Queue.  Review persistence stays
behind the injected callback; this widget never opens SQLite or writes Grid
Annotations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .detection_windows import Rect
from .proposal_generation import DefectProposal
from .proposal_queue import ReviewQueueFilters, ReviewQueueItem, filter_review_queue


ReviewCallback = Callable[[str, str, Rect, Mapping[str, Any]], Any]


class ProposalReviewControls(QWidget):
    """List Review Queue values and request append-only review decisions."""

    _LOW_CONFIDENCE_THRESHOLD = 0.50
    _SPINBOX_MAX = 2_000_000_000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: tuple[ReviewQueueItem, ...] = ()
        self._review_callback: ReviewCallback | None = None
        self._selected_proposal_id: str | None = None

        self.low_confidence_checkbox = QCheckBox("Low confidence (< 0.50)", self)
        self.low_confidence_checkbox.setObjectName("proposalLowConfidenceCheckBox")
        self.conflict_checkbox = QCheckBox("Class conflict only", self)
        self.conflict_checkbox.setObjectName("proposalConflictOnlyCheckBox")
        self.disagreement_checkbox = QCheckBox("Run disagreement only", self)
        self.disagreement_checkbox.setObjectName("proposalDisagreementOnlyCheckBox")

        self.status_filter = QComboBox(self)
        self.status_filter.setObjectName("proposalStatusFilter")
        self.status_filter.addItems(("All", "Unreviewed", "Accepted", "Rejected", "Corrected"))
        filter_row = QHBoxLayout()
        filter_row.addWidget(self.low_confidence_checkbox)
        filter_row.addWidget(self.conflict_checkbox)
        filter_row.addWidget(self.disagreement_checkbox)
        filter_row.addWidget(QLabel("Status", self))
        filter_row.addWidget(self.status_filter)

        self.proposal_table = QTableWidget(0, 6, self)
        self.proposal_table.setObjectName("proposalListWidget")
        self.proposal_table.setHorizontalHeaderLabels(
            ("Proposal", "Class", "Source Rect", "Mean Confidence", "Peak Confidence", "Status")
        )
        self.proposal_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.proposal_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.proposal_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.proposal_table.currentCellChanged.connect(self._selection_changed)

        self.correction_x_spin = QSpinBox(self)
        self.correction_x_spin.setObjectName("proposalCorrectionXSpinBox")
        self.correction_y_spin = QSpinBox(self)
        self.correction_y_spin.setObjectName("proposalCorrectionYSpinBox")
        self.correction_width_spin = QSpinBox(self)
        self.correction_width_spin.setObjectName("proposalCorrectionWidthSpinBox")
        self.correction_height_spin = QSpinBox(self)
        self.correction_height_spin.setObjectName("proposalCorrectionHeightSpinBox")
        self._set_correction_ranges()
        correction_group = QGroupBox("Correct source-pixel rectangle", self)
        correction_form = QFormLayout(correction_group)
        correction_form.addRow("x", self.correction_x_spin)
        correction_form.addRow("y", self.correction_y_spin)
        correction_form.addRow("width", self.correction_width_spin)
        correction_form.addRow("height", self.correction_height_spin)

        self.accept_button = QPushButton("Accept", self)
        self.accept_button.setObjectName("acceptProposalButton")
        self.reject_button = QPushButton("Reject", self)
        self.reject_button.setObjectName("rejectProposalButton")
        self.correct_button = QPushButton("Correct", self)
        self.correct_button.setObjectName("correctProposalButton")
        action_row = QHBoxLayout()
        action_row.addWidget(self.accept_button)
        action_row.addWidget(self.reject_button)
        action_row.addWidget(self.correct_button)

        self.status_label = QLabel("No proposals configured.", self)
        self.status_label.setObjectName("proposalReviewStatusLabel")
        self.status_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addLayout(filter_row)
        layout.addWidget(self.proposal_table)
        layout.addWidget(correction_group)
        layout.addLayout(action_row)
        layout.addWidget(self.status_label)

        self.low_confidence_checkbox.toggled.connect(self._refresh)
        self.conflict_checkbox.toggled.connect(self._refresh)
        self.disagreement_checkbox.toggled.connect(self._refresh)
        self.status_filter.currentIndexChanged.connect(self._refresh)
        self.accept_button.clicked.connect(lambda: self._submit("accepted"))
        self.reject_button.clicked.connect(lambda: self._submit("rejected"))
        self.correct_button.clicked.connect(lambda: self._submit("corrected"))
        self._set_action_enabled(False)

    @property
    def items(self) -> tuple[ReviewQueueItem, ...]:
        """Return the current in-memory queue, including UI review updates."""

        return self._items

    def configure(
        self,
        items: Iterable[ReviewQueueItem] | Iterable[DefectProposal],
        review_callback: ReviewCallback | None,
    ) -> None:
        """Bind pure queue/proposal values and an injected review callback."""

        if review_callback is not None and not callable(review_callback):
            raise TypeError("review_callback must be callable or None")
        values = tuple(items)
        if values and all(isinstance(value, ReviewQueueItem) for value in values):
            queue = filter_review_queue(values)
        elif values and all(isinstance(value, DefectProposal) for value in values):
            queue = filter_review_queue(values)
        elif not values:
            queue = ()
        else:
            raise TypeError("items must contain only ReviewQueueItem or DefectProposal values")
        self._items = tuple(queue)
        self._review_callback = review_callback
        self._selected_proposal_id = None
        self.low_confidence_checkbox.setChecked(False)
        self.conflict_checkbox.setChecked(False)
        self.disagreement_checkbox.setChecked(False)
        self.status_filter.setCurrentIndex(0)
        self._refresh()
        self.status_label.setText(f"{len(self._items)} proposal(s) ready for review.")

    def _set_correction_ranges(self) -> None:
        self.correction_x_spin.setRange(0, self._SPINBOX_MAX)
        self.correction_y_spin.setRange(0, self._SPINBOX_MAX)
        self.correction_width_spin.setRange(1, self._SPINBOX_MAX)
        self.correction_height_spin.setRange(1, self._SPINBOX_MAX)

    def _filtered_items(self) -> tuple[ReviewQueueItem, ...]:
        status_text = self.status_filter.currentText().strip().lower()
        statuses = None if status_text == "all" else (status_text,)
        filters = ReviewQueueFilters(
            low_confidence_threshold=(
                self._LOW_CONFIDENCE_THRESHOLD
                if self.low_confidence_checkbox.isChecked()
                else None
            ),
            cross_class_conflict=True if self.conflict_checkbox.isChecked() else None,
            provenance_disagreement=True if self.disagreement_checkbox.isChecked() else None,
            statuses=statuses,
        )
        return filter_review_queue(self._items, filters)

    def _refresh(self, *_args: object) -> None:
        selected_id = self._selected_proposal_id
        visible = self._filtered_items()
        self.proposal_table.blockSignals(True)
        try:
            self.proposal_table.setRowCount(len(visible))
            for row, item in enumerate(visible):
                values = (
                    item.proposal_id,
                    item.class_name,
                    _format_rect(item.source_rect),
                    f"{item.mean_confidence:.3f}",
                    f"{item.peak_confidence:.3f}",
                    item.status.title(),
                )
                for column, value in enumerate(values):
                    cell = QTableWidgetItem(value)
                    if column == 0:
                        cell.setData(Qt.ItemDataRole.UserRole, item.proposal_id)
                    self.proposal_table.setItem(row, column, cell)
            target_row = next(
                (row for row, item in enumerate(visible) if item.proposal_id == selected_id),
                0 if visible else -1,
            )
            self.proposal_table.setCurrentCell(target_row, 0)
        finally:
            self.proposal_table.blockSignals(False)
        if visible:
            self._selection_changed(self.proposal_table.currentRow(), 0, -1, -1)
        else:
            self._selected_proposal_id = None
            self._set_action_enabled(False)

    def _selection_changed(self, row: int, _column: int, _previous_row: int, _previous_column: int) -> None:
        if row < 0 or row >= self.proposal_table.rowCount():
            self._selected_proposal_id = None
            self._set_action_enabled(False)
            return
        cell = self.proposal_table.item(row, 0)
        proposal_id = cell.data(Qt.ItemDataRole.UserRole) if cell is not None else None
        if not isinstance(proposal_id, str):
            self._selected_proposal_id = None
            self._set_action_enabled(False)
            return
        self._selected_proposal_id = proposal_id
        item = self._selected_item()
        if item is not None:
            self._set_correction_values(item.source_rect)
        self._set_action_enabled(item is not None)

    def _selected_item(self) -> ReviewQueueItem | None:
        proposal_id = self._selected_proposal_id
        if proposal_id is None:
            return None
        return next((item for item in self._items if item.proposal_id == proposal_id), None)

    def _set_correction_values(self, rect: Rect) -> None:
        values = (
            (self.correction_x_spin, rect.x),
            (self.correction_y_spin, rect.y),
            (self.correction_width_spin, rect.width),
            (self.correction_height_spin, rect.height),
        )
        for widget, value in values:
            blocked = widget.blockSignals(True)
            widget.setValue(int(value))
            widget.blockSignals(blocked)

    def _set_action_enabled(self, enabled: bool) -> None:
        ready = enabled and self._review_callback is not None
        self.accept_button.setEnabled(ready)
        self.reject_button.setEnabled(ready)
        self.correct_button.setEnabled(ready)
        self.correction_x_spin.setEnabled(enabled)
        self.correction_y_spin.setEnabled(enabled)
        self.correction_width_spin.setEnabled(enabled)
        self.correction_height_spin.setEnabled(enabled)

    def _submit(self, status: str) -> None:
        item = self._selected_item()
        callback = self._review_callback
        if item is None or callback is None:
            self.status_label.setText("Select a proposal and configure a review callback first.")
            return
        source_rect = item.source_rect
        if status == "corrected":
            source_rect = Rect(
                self.correction_x_spin.value(),
                self.correction_y_spin.value(),
                self.correction_width_spin.value(),
                self.correction_height_spin.value(),
            )
            if source_rect == item.source_rect:
                self.status_label.setText("Correction must change the source rectangle.")
                return
        provenance = dict(item.proposal.provenance)
        try:
            callback(item.proposal_id, status, source_rect, provenance)
        except Exception as error:  # callback owns persistence/error policy
            self.status_label.setText(f"Review not recorded: {error}")
            return
        self._items = tuple(
            ReviewQueueItem(
                proposal=current.proposal,
                status=status if current.proposal_id == item.proposal_id else current.status,
                source_rect=source_rect if current.proposal_id == item.proposal_id else current.source_rect,
                revision=current.revision,
                cross_class_conflict=current.cross_class_conflict,
                provenance_disagreement=current.provenance_disagreement,
            )
            for current in self._items
        )
        self._refresh()
        self.status_label.setText(f"{status.title()} {item.proposal_id}.")


def _format_rect(rect: Rect) -> str:
    return f"({rect.x}, {rect.y}, {rect.width}, {rect.height})"


__all__ = ["ProposalReviewControls", "ReviewCallback"]
