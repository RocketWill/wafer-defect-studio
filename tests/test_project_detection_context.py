import os
import queue
import sqlite3
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QFileDialog, QLabel, QPushButton

import tests.test_detection_run as detection_run_test
from wafer_defect_studio import project
from wafer_defect_studio.detection_run import create_detection_profile, create_detection_run
from wafer_defect_studio.detection_worker import DetectionRequest, run_detection_worker
from wafer_defect_studio.main_window import MainWindow


class ProjectDetectionContextTest(unittest.TestCase):
    def test_reopen_restores_detection_context_and_missing_context_disables_run(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = detection_run_test.DetectionRunTest()._project_with_approved_evaluation(root)
            profile = create_detection_profile(
                project_path,
                evaluation_id="evaluation-1",
                window_size=(2, 2),
                stride=(2, 2),
                reflect_padding=True,
                thresholds={"scratch": 0.6},
                center_weighting="uniform",
                map_generation={},
                profile_id="profile-1",
            )
            stage = root / "detection-stage"
            request = DetectionRequest(
                request_id="detection-1",
                source=np.asarray([[0, 255], [255, 0]], dtype=np.uint8),
                staging_path=stage,
                run_id="detection-1",
                profile_id=profile.profile_id,
                approved_evaluation_id="evaluation-1",
                class_names=("scratch",),
                window_size=(2, 2),
                stride=(2, 2),
            )
            run_detection_worker(request, queue.Queue(), threading.Event())
            create_detection_run(
                project_path,
                profile_id=profile.profile_id,
                evaluation_id="evaluation-1",
                source_fingerprints={"image-1": "fp-1"},
                provenance={"coordinate_system": "source-image-pixels"},
                status="completed",
                artifact_path=stage,
                run_id="detection-1",
            )
            database = project_path / "project.sqlite"
            database_bytes = database.read_bytes()
            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)

            first = MainWindow(settings=settings)
            first.show()
            app.processEvents()
            try:
                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(project_path),
                ):
                    first.open_project_action.trigger()
                first.workspace_actions["Detect"].trigger()
                app.processEvents()
                first.findChild(QComboBox, "detectionClassSelector").setCurrentText("scratch")
                first.findChild(QCheckBox, "detectionImageLayerCheckBox").setChecked(False)
                app.processEvents()
            finally:
                first.close()
                first.deleteLater()
                app.processEvents()

            second = MainWindow(settings=settings)
            second.show()
            app.processEvents()
            try:
                self.assertEqual(second.current_workspace, "Detect")
                self.assertEqual(
                    second.findChild(QComboBox, "detectionProfileComboBox").currentData(),
                    "profile-1",
                )
                self.assertEqual(
                    second.findChild(QComboBox, "detectionRunComboBox").currentData(),
                    "detection-1",
                )
                self.assertEqual(
                    second.findChild(QComboBox, "detectionClassSelector").currentText(),
                    "scratch",
                )
                self.assertFalse(
                    second.findChild(QCheckBox, "detectionImageLayerCheckBox").isChecked()
                )
            finally:
                second.close()
                second.deleteLater()
                app.processEvents()

            project_id = project.open_project(project_path).project_id
            settings.setValue(f"detectionContext/{project_id}/profileId", "missing-profile")
            settings.setValue(f"detectionContext/{project_id}/runId", "missing-run")
            settings.sync()
            third = MainWindow(settings=settings)
            third.show()
            app.processEvents()
            try:
                status = third.findChild(QLabel, "detectionInventoryStatusLabel")
                start = third.findChild(QPushButton, "startDetectionButton")
                self.assertIn("Detection Profile unavailable: missing-profile", status.text())
                self.assertFalse(start.isEnabled())
                self.assertEqual(database.read_bytes(), database_bytes)
            finally:
                third.close()
                third.deleteLater()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
