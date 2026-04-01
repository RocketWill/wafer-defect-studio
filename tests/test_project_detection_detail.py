import os
import queue
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QComboBox, QFileDialog, QLabel

import tests.test_detection_run as detection_run_test
from wafer_defect_studio.detection_run import create_detection_profile, create_detection_run
from wafer_defect_studio.detection_worker import DetectionRequest, run_detection_worker
from wafer_defect_studio.main_window import MainWindow


class ProjectDetectionDetailTest(unittest.TestCase):
    def test_selected_completed_detection_run_renders_native_cam_detail_read_only(self):
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
                reflect_padding=True,
                center_weighting="uniform",
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
                window.workspace_actions["Detect"].trigger()
                app.processEvents()

                classes = window.findChild(QComboBox, "detectionClassSelector")
                map_preview = window.findChild(QLabel, "detectionMapPreviewLabel")
                confidence = window.findChild(QLabel, "detectionConfidenceLabel")
                disclaimer = window.findChild(QLabel, "detectionDisclaimerLabel")
                self.assertEqual(classes.currentText(), "scratch")
                self.assertIn("source 2×2 px", map_preview.text())
                self.assertIn("(0, 0)", confidence.text())
                self.assertIn("Approximate localization", disclaimer.text())
                self.assertIn("not a segmentation mask", disclaimer.text())
                self.assertEqual(database.read_bytes(), database_bytes)
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
