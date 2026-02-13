import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
)

from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.proposal_generation import DefectProposal
from wafer_defect_studio.proposal_queue import ReviewQueueItem


class ProposalUiTest(unittest.TestCase):
    def test_queue_filters_and_review_actions_are_ui_only(self):
        app = QApplication.instance() or QApplication([])
        low = DefectProposal(
            "proposal-1-low",
            "scratch",
            Rect(10, 20, 5, 4),
            20,
            0.72,
            0.42,
            {"image_asset_id": "wafer-1", "run_id": "run-1"},
        )
        high = DefectProposal(
            "proposal-2-high",
            "particle",
            Rect(12, 22, 6, 5),
            30,
            0.98,
            0.91,
            {"image_asset_id": "wafer-1", "run_id": "run-2"},
        )
        items = (
            ReviewQueueItem(
                low,
                "unreviewed",
                low.source_rect,
                cross_class_conflict=True,
                provenance_disagreement=True,
            ),
            ReviewQueueItem(high, "unreviewed", high.source_rect),
        )
        calls = []
        window = MainWindow()
        window.configure_proposal_review(
            items,
            lambda proposal_id, status, source_rect, provenance: calls.append(
                (proposal_id, status, source_rect, provenance)
            ),
        )
        window.show()
        app.processEvents()

        table = window.findChild(QTableWidget, "proposalListWidget")
        self.assertEqual(table.rowCount(), 2)
        self.assertIn("scratch", table.item(0, 1).text())
        self.assertIn("(10, 20, 5, 4)", table.item(0, 2).text())
        self.assertIn("0.420", table.item(0, 3).text())
        self.assertIn("0.720", table.item(0, 4).text())
        self.assertEqual(table.item(0, 5).text(), "Unreviewed")

        low_checkbox = window.findChild(QCheckBox, "proposalLowConfidenceCheckBox")
        low_checkbox.click()
        self.assertEqual(table.rowCount(), 1)
        self.assertEqual(table.item(0, 0).text(), "proposal-1-low")
        low_checkbox.click()
        window.findChild(QCheckBox, "proposalConflictOnlyCheckBox").click()
        self.assertEqual(table.rowCount(), 1)
        window.findChild(QCheckBox, "proposalConflictOnlyCheckBox").click()
        window.findChild(QCheckBox, "proposalDisagreementOnlyCheckBox").click()
        self.assertEqual(table.rowCount(), 1)
        window.findChild(QCheckBox, "proposalDisagreementOnlyCheckBox").click()
        status_filter = window.findChild(QComboBox, "proposalStatusFilter")
        status_filter.setCurrentText("Accepted")
        self.assertEqual(table.rowCount(), 0)
        status_filter.setCurrentText("All")

        table.selectRow(1)
        window.findChild(QPushButton, "acceptProposalButton").click()
        self.assertEqual(calls[0][:3], ("proposal-2-high", "accepted", high.source_rect))
        self.assertEqual(table.item(1, 5).text(), "Accepted")

        table.selectRow(1)
        x_spin = window.findChild(QSpinBox, "proposalCorrectionXSpinBox")
        y_spin = window.findChild(QSpinBox, "proposalCorrectionYSpinBox")
        width_spin = window.findChild(QSpinBox, "proposalCorrectionWidthSpinBox")
        height_spin = window.findChild(QSpinBox, "proposalCorrectionHeightSpinBox")
        x_spin.setValue(1)
        y_spin.setValue(2)
        width_spin.setValue(3)
        height_spin.setValue(4)
        window.findChild(QPushButton, "correctProposalButton").click()
        self.assertEqual(calls[1][:3], ("proposal-2-high", "corrected", Rect(1, 2, 3, 4)))
        self.assertEqual(table.item(1, 5).text(), "Corrected")
        self.assertIn("Corrected", window.findChild(QLabel, "proposalReviewStatusLabel").text())
        window.close()


if __name__ == "__main__":
    unittest.main()
