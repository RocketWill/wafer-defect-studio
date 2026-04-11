import os
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QFileDialog, QDockWidget, QLabel, QTableWidget

import tests.test_proposal_review as proposal_review_test
from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.proposal_generation import DefectProposal
from wafer_defect_studio.proposal_store import save_defect_proposal


class ProjectProposalReviewContextTest(unittest.TestCase):
    def test_review_workspace_loads_selected_run_queue_without_project_writes(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path, run_id, profile_id = (
                proposal_review_test.ProposalReviewTest()._project_with_detection_run(root)
            )
            proposal = DefectProposal(
                "proposal-1",
                "scratch",
                Rect(11, 13, 5, 4),
                12,
                0.91,
                0.74,
                {
                    "detection_run_id": run_id,
                    "profile_id": profile_id,
                    "image_asset_id": "image-1",
                },
            )
            save_defect_proposal(
                project_path,
                proposal,
                detection_run_id=run_id,
                profile_id=profile_id,
            )
            database = project_path / "project.sqlite"
            before = database.read_bytes()
            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
            window = MainWindow(settings=settings)
            window.show()
            app.processEvents()
            try:
                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(project_path),
                ):
                    window.open_project_action.trigger()
                window.workspace_actions["Review"].trigger()
                app.processEvents()

                dock = window.findChild(QDockWidget, "proposalReviewDock")
                table = window.findChild(QTableWidget, "proposalListWidget")
                status = window.findChild(QLabel, "proposalReviewStatusLabel")
                self.assertTrue(dock.isVisible())
                self.assertEqual(table.rowCount(), 1)
                self.assertEqual(table.item(0, 0).text(), "proposal-1")
                self.assertIn("1 proposal", status.text())
                self.assertEqual(database.read_bytes(), before)
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
