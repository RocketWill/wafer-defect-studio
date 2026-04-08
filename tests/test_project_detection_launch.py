import os
import queue
import sqlite3
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QFileDialog, QLabel, QPushButton

import tests.test_detection_run as detection_run_test
from wafer_defect_studio.detection_run import create_detection_profile
from wafer_defect_studio.detection_worker import DetectionTerminal
from wafer_defect_studio.image_asset import ImageAsset
from wafer_defect_studio.main_window import MainWindow


class _FakeHandle:
    def __init__(self):
        self.queue = queue.Queue()
        self.cancelled = False

    def cancel(self, _reason="user"):
        self.cancelled = True
        return True

    def is_alive(self):
        return True


class ProjectDetectionLaunchTest(unittest.TestCase):
    def test_project_detection_launch_persists_cancelled_run_without_blocking(self):
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
            source_path = root / "wafer.png"
            image = QImage(2, 2, QImage.Format.Format_Grayscale8)
            image.bits()[:4] = bytes((0, 64, 128, 255))
            image.save(str(source_path), "PNG")
            asset = ImageAsset(
                "image-1",
                source_path,
                2,
                2,
                "uint8",
                "PNG",
                "fp-1",
            )
            database = project_path / "project.sqlite"
            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
            window = MainWindow(settings=settings)
            window.show()
            app.processEvents()
            handle = _FakeHandle()
            started = []

            def launcher(request):
                started.append(request)
                return handle

            try:
                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(project_path),
                ):
                    window.open_project_action.trigger()
                window.show_wafer_image(asset)
                window.workspace_actions["Detect"].trigger()
                window.configure_project_detection(launcher=launcher)
                app.processEvents()

                start = window.findChild(QPushButton, "startDetectionButton")
                start.click()
                self.assertEqual(
                    len(started),
                    1,
                    f"enabled={start.isEnabled()} status={window.findChild(QLabel, 'detectionStatusLabel').text()} profile={window.findChild(QComboBox, 'detectionProfileComboBox').currentData()}",
                )
                request = started[0]
                self.assertEqual(request.profile_id, profile.profile_id)
                self.assertEqual(request.approved_evaluation_id, "evaluation-1")
                self.assertEqual(request.source.shape, (2, 2))
                self.assertTrue(request.run_id)
                handle.queue.put(
                    DetectionTerminal(
                        request.request_id,
                        "cancelled",
                        "Detection cancelled; no map set was published.",
                    )
                )
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    QTest.qWait(25)
                    connection = sqlite3.connect(database)
                    try:
                        row = connection.execute(
                            "SELECT status FROM detection_runs WHERE detection_run_id = ?",
                            (request.run_id,),
                        ).fetchone()
                    finally:
                        connection.close()
                    if row is not None:
                        break
                self.assertEqual(row, ("cancelled",))
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
