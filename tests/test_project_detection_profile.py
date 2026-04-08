import os
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QComboBox, QFileDialog, QLabel, QPushButton, QSpinBox

import tests.test_detection_run as detection_run_test
from wafer_defect_studio.detection_run import load_detection_profile
from wafer_defect_studio.evaluation_run import create_evaluation
from wafer_defect_studio.main_window import MainWindow


class ProjectDetectionProfileTest(unittest.TestCase):
    def test_active_project_creates_profile_from_approved_evaluation(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = detection_run_test.DetectionRunTest()._project_with_approved_evaluation(root)
            create_evaluation(
                project_path,
                "run-1",
                {"macro_f1": 0.2},
                {"target_satisfied": False},
                {},
                evaluation_id="evaluation-unapproved",
            )
            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
            window = MainWindow(settings=settings)
            window.show()
            app.processEvents()
            database = project_path / "project.sqlite"
            try:
                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(project_path),
                ):
                    window.open_project_action.trigger()
                window.workspace_actions["Detect"].trigger()
                app.processEvents()

                evaluations = window.findChild(
                    QComboBox, "detectionProfileEvaluationComboBox"
                )
                create_button = window.findChild(
                    QPushButton, "createDetectionProfileButton"
                )
                status = window.findChild(
                    QLabel, "detectionProfileCreationStatusLabel"
                )
                self.assertGreaterEqual(evaluations.findData("evaluation-1"), 0)
                self.assertTrue(
                    evaluations.model().item(evaluations.findData("evaluation-1")).isEnabled()
                )
                unapproved_index = evaluations.findData("evaluation-unapproved")
                self.assertGreaterEqual(unapproved_index, 0)
                self.assertFalse(evaluations.model().item(unapproved_index).isEnabled())

                window.findChild(QSpinBox, "detectionProfileWindowWidthSpinBox").setValue(32)
                window.findChild(QSpinBox, "detectionProfileWindowHeightSpinBox").setValue(24)
                window.findChild(QSpinBox, "detectionProfileStrideXSpinBox").setValue(16)
                window.findChild(QSpinBox, "detectionProfileStrideYSpinBox").setValue(12)
                before = database.read_bytes()
                create_button.click()
                app.processEvents()

                profiles = window.findChild(QComboBox, "detectionProfileComboBox")
                self.assertGreaterEqual(profiles.count(), 1)
                profile_id = profiles.currentData()
                self.assertTrue(profile_id)
                profile = load_detection_profile(project_path, str(profile_id))
                self.assertEqual(profile.evaluation_id, "evaluation-1")
                self.assertEqual(profile.window_size, (32, 24))
                self.assertEqual(profile.stride, (16, 12))
                self.assertIn("Created Detection Profile", status.text())
                self.assertNotEqual(database.read_bytes(), before)
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
