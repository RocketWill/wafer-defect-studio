import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QCheckBox, QLineEdit, QLabel, QPushButton

from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.proposal_generation import DefectProposal
from wafer_defect_studio.result_export import ExportBundleResult, build_reviewed_rows


class ExportUiTest(unittest.TestCase):
    def test_result_export_controls_gate_paths_and_report_callback_failure(self):
        app = QApplication.instance() or QApplication([])
        proposal = DefectProposal(
            "proposal-1",
            "scratch",
            Rect(1, 1, 2, 2),
            4,
            0.9,
            0.8,
            {},
        )
        rows = build_reviewed_rows(
            (proposal,),
            project_id="project",
            image_asset_id="image",
            run_id="run",
            profile_id="profile",
        )
        source = QImage(8, 6, QImage.Format_Grayscale8)
        source.fill(80)
        calls = []

        def failing_callback(*args, **kwargs):
            calls.append((args, kwargs))
            return ExportBundleResult(False, (), "injected failure")

        window = MainWindow()
        window.configure_result_export(source, rows, "scratch", export_callback=failing_callback)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            edits = (
                window.findChild(QLineEdit, "resultExportCsvPathEdit"),
                window.findChild(QLineEdit, "resultExportJsonPathEdit"),
                window.findChild(QLineEdit, "resultExportPngPathEdit"),
            )
            for edit, suffix in zip(edits, (".csv", ".json", ".png")):
                self.assertIsNotNone(edit)
                edit.setText(str(root / f"result{suffix}"))
            button = window.findChild(QPushButton, "exportResultButton")
            self.assertIsNotNone(button)
            self.assertTrue(button.isEnabled())
            overwrite = window.findChild(QCheckBox, "resultExportOverwriteCheckBox")
            overwrite.setChecked(True)
            button.click()
            self.assertEqual(calls[0][1]["overwrite"], True)
            status = window.findChild(QLabel, "resultExportStatusLabel")
            self.assertIn("Export failed", status.text())
            self.assertNotIn("Exported", status.text())
        window.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
