import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtCore import QThreadPool
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QPushButton
from unittest.mock import patch

import tests.test_detection_run as detection_run_test
from wafer_defect_studio.cam_detection import CamDetectionArtifact
from wafer_defect_studio.detection_run import create_detection_profile, create_detection_run
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.proposal_store import load_defect_proposals


class ProjectProposalGenerationTest(unittest.TestCase):
    def test_completed_project_detection_run_generates_persisted_proposals(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = detection_run_test.DetectionRunTest()._project_with_approved_evaluation(root)
            profile = create_detection_profile(
                project_path,
                evaluation_id="evaluation-1",
                window_size=(2, 2),
                stride=(2, 2),
                thresholds={"scratch": 0.5},
                profile_id="profile-1",
            )
            staging = root / "detection-stage"
            staging.mkdir()
            artifact = CamDetectionArtifact(
                maps=np.asarray(
                    [
                        [[0.0], [0.8]],
                        [[0.0], [0.9]],
                    ],
                    dtype=np.float32,
                ),
                coverage=np.ones((2, 2), dtype=np.float32),
                class_names=("scratch",),
                source_width=2,
                source_height=2,
                source_transform={"coordinate_system": "source-image pixels"},
                window_settings={"window_count": 1},
                provenance={
                    "run_id": "detection-1",
                    "profile_id": profile.profile_id,
                    "evaluation_id": "evaluation-1",
                },
            )
            artifact.write_json(staging / "maps.json")
            (staging / "provenance.json").write_text(
                json.dumps(artifact.provenance), encoding="utf-8"
            )
            create_detection_run(
                project_path,
                profile_id=profile.profile_id,
                evaluation_id="evaluation-1",
                source_fingerprints={"image-1": "fp-1"},
                provenance={"coordinate_system": "source-image-pixels"},
                status="completed",
                artifact_path=staging,
                run_id="detection-1",
            )

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
                generate = window.findChild(QPushButton, "generateProposalsButton")
                status = window.findChild(QLabel, "proposalGenerationStatusLabel")
                self.assertIsNotNone(generate)
                generate.click()
                QThreadPool.globalInstance().waitForDone(5000)
                app.processEvents()

                deadline = time.monotonic() + 3
                while (
                    "Generated 1" not in status.text()
                    and time.monotonic() < deadline
                ):
                    QTest.qWait(25)
                proposals = load_defect_proposals(
                    project_path, detection_run_id="detection-1"
                )
                self.assertEqual(len(proposals), 1, status.text())
                self.assertEqual(proposals[0].class_name, "scratch")
                self.assertIn("Generated 1", status.text())
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
