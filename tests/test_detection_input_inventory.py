import os
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QComboBox, QFileDialog, QLabel

import tests.test_detection_run as detection_run_test
from wafer_defect_studio.detection_run import create_detection_profile, create_detection_run
from wafer_defect_studio.main_window import MainWindow


class DetectionInputInventoryTest(unittest.TestCase):
    def test_detect_workspace_lists_valid_and_unavailable_profiles_and_runs_read_only(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = detection_run_test.DetectionRunTest()._project_with_approved_evaluation(root)
            profile = create_detection_profile(
                project_path,
                evaluation_id="evaluation-1",
                window_size=(32, 24),
                stride=(16, 12),
                reflect_padding=True,
                thresholds={"scratch": 0.6},
                center_weighting="hann",
                map_generation={"smoothing": 3, "minimum_area": 4},
                profile_id="profile-1",
            )
            create_detection_run(
                project_path,
                profile_id=profile.profile_id,
                evaluation_id="evaluation-1",
                source_fingerprints={"image-1": "fp-1"},
                provenance={"coordinate_system": "source-image-pixels", "window_count": 8},
                run_id="detection-1",
            )
            connection = sqlite3.connect(project_path / "project.sqlite")
            try:
                connection.execute(
                    "INSERT INTO detection_profiles VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        "profile-corrupt",
                        "2026-01-02T00:00:00+00:00",
                        "evaluation-1",
                        "run-1",
                        "model-fingerprint",
                        "{bad-json",
                    ),
                )
                connection.execute(
                    "INSERT INTO detection_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "detection-corrupt",
                        "2026-01-02T00:00:00+00:00",
                        "evaluation-1",
                        "run-1",
                        "profile-1",
                        "model-fingerprint",
                        "{bad-json",
                        "{}",
                        "created",
                        None,
                        None,
                    ),
                )
                connection.commit()
            finally:
                connection.close()

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

                profiles = window.findChild(QComboBox, "detectionProfileComboBox")
                runs = window.findChild(QComboBox, "detectionRunComboBox")
                status = window.findChild(QLabel, "detectionInventoryStatusLabel")
                self.assertEqual(profiles.itemData(0), "profile-1")
                self.assertTrue(profiles.model().item(0).isEnabled())
                corrupt_profile = profiles.findData("profile-corrupt")
                self.assertGreaterEqual(corrupt_profile, 0)
                self.assertFalse(profiles.model().item(corrupt_profile).isEnabled())
                self.assertIn("Unavailable Detection Profile: profile-corrupt", profiles.itemText(corrupt_profile))
                self.assertEqual(runs.itemData(0), "detection-1")
                self.assertTrue(runs.model().item(0).isEnabled())
                corrupt_run = runs.findData("detection-corrupt")
                self.assertGreaterEqual(corrupt_run, 0)
                self.assertFalse(runs.model().item(corrupt_run).isEnabled())
                self.assertIn("Unavailable Detection Run: detection-corrupt", runs.itemText(corrupt_run))
                self.assertIn("unavailable", status.text().lower())
                self.assertEqual(database.read_bytes(), database_bytes)
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
