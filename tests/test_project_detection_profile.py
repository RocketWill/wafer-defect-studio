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
from wafer_defect_studio.evaluation_run import create_evaluation, record_evaluation_decision
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.training_run import RunConfig, create_training_run


class ProjectDetectionProfileTest(unittest.TestCase):
    def test_v3_evaluation_requires_checkpoint_patch_geometry_for_profile(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = detection_run_test.DetectionRunTest()._project_with_approved_evaluation(root)
            run = create_training_run(
                project_path,
                RunConfig(
                    "snapshot-1",
                    "split-1",
                    patch_size=128,
                    patch_stride=64,
                    bag_pooling="max",
                ),
                run_id="run-v3",
            )
            evaluation = create_evaluation(
                project_path,
                run.run_id,
                {"macro_f1": 0.9},
                {"target_satisfied": True},
                {},
                evaluation_id="evaluation-v3",
            )
            record_evaluation_decision(
                project_path, evaluation.evaluation_id, "validated", actor="engineer"
            )
            record_evaluation_decision(
                project_path,
                evaluation.evaluation_id,
                "approved",
                actor="engineer",
                notes="approved for patch detection",
            )
            window = MainWindow(
                settings=QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
            )
            window.show()
            app.processEvents()
            try:
                with patch.object(QFileDialog, "getExistingDirectory", return_value=str(project_path)):
                    window.open_project_action.trigger()
                window.workspace_actions["Detect"].trigger()
                evaluations = window.findChild(QComboBox, "detectionProfileEvaluationComboBox")
                evaluations.setCurrentIndex(evaluations.findData("evaluation-v3"))
                app.processEvents()
                width = window.findChild(QSpinBox, "detectionProfileWindowWidthSpinBox")
                height = window.findChild(QSpinBox, "detectionProfileWindowHeightSpinBox")
                stride_x = window.findChild(QSpinBox, "detectionProfileStrideXSpinBox")
                stride_y = window.findChild(QSpinBox, "detectionProfileStrideYSpinBox")
                create = window.findChild(QPushButton, "createDetectionProfileButton")
                status = window.findChild(QLabel, "detectionProfileCreationStatusLabel")
                self.assertEqual(
                    (width.value(), height.value(), stride_x.value(), stride_y.value()),
                    (128, 128, 64, 64),
                )
                stride_x.setValue(65)
                app.processEvents()
                self.assertFalse(create.isEnabled())
                self.assertIn("must match Patch Classification", status.text())
                stride_x.setValue(64)
                app.processEvents()
                self.assertTrue(create.isEnabled())
                create.click()
                profile_id = window.findChild(QComboBox, "detectionProfileComboBox").currentData()
                profile = load_detection_profile(project_path, str(profile_id))
                self.assertEqual(profile.training_run_id, "run-v3")
                self.assertEqual((profile.window_size, profile.stride), ((128, 128), (64, 64)))
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()

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
