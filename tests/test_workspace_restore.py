import os
import sqlite3
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.job_state import JobKind, JobSnapshot, start_job
from wafer_defect_studio.job_store import create_job
from wafer_defect_studio.main_window import MainWindow


class WorkspaceRestoreTest(unittest.TestCase):
    def _schema18_project(self, root: Path) -> Path:
        path = root / "project"
        project.create_project(path)
        connection = sqlite3.connect(path / "project.sqlite")
        try:
            for statement in (
                project._DEFECT_CLASSES_TABLE_SQL,
                project._GRID_ANNOTATIONS_TABLE_SQL,
                project._IMAGE_REVIEWS_TABLE_SQL,
                project._DATA_GROUPS_TABLE_SQL,
                project._IMAGE_DATA_GROUPS_TABLE_SQL,
                project._TRAINING_SCOPE_TABLE_SQL,
                project._DATASET_SNAPSHOTS_TABLE_SQL,
                project._DATASET_SPLITS_TABLE_SQL,
                project._TRAINING_RUNS_TABLE_SQL,
                project._EVALUATION_RUNS_TABLE_SQL,
                project._EVALUATION_DECISIONS_TABLE_SQL,
                project._DETECTION_PROFILES_TABLE_SQL,
                project._DETECTION_RUNS_TABLE_SQL,
                project._DEFECT_PROPOSALS_TABLE_SQL,
                project._PROPOSAL_REVIEW_REVISIONS_TABLE_SQL,
                project._PROPOSAL_CONVERSIONS_TABLE_SQL,
            ):
                connection.execute(statement)
            connection.execute("UPDATE project_metadata SET schema_version = 18")
            connection.execute("PRAGMA user_version = 18")
            connection.commit()
        finally:
            connection.close()
        return path

    @staticmethod
    def _write_grayscale_png(path: Path) -> None:
        image = QImage(5, 4, QImage.Format.Format_Grayscale8)
        image.fill(80)
        if not image.save(str(path), "PNG"):
            raise AssertionError(f"failed to save {path}")

    @staticmethod
    def _wait_for(window: MainWindow, predicate) -> None:
        deadline = time.monotonic() + 5
        app = QApplication.instance()
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(10)

    def test_startup_restores_valid_workspace_image_and_surfaces_stale_job(self):
        app = QApplication.instance() or QApplication([])
        now = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = self._schema18_project(root)
            source = root / "wafer.png"
            self._write_grayscale_png(source)
            asset = image_asset.register_wafer_image(project_path, source)
            stale = start_job(
                JobSnapshot(
                    "stale-job",
                    JobKind.DETECTION,
                    total=3,
                    heartbeat_at=(now - timedelta(seconds=120)).isoformat(),
                ),
                heartbeat_at=(now - timedelta(seconds=120)).isoformat(),
            )
            create_job(project_path, stale)
            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
            settings.setValue("activeProjectPath", str(project_path))
            settings.setValue("currentWorkspace", "Review")
            settings.setValue("currentImageAssetId", asset.image_asset_id)
            settings.sync()

            with patch("wafer_defect_studio.main_window.datetime") as clock:
                clock.now.return_value = now
                window = MainWindow(settings=settings)
            window.show()
            try:
                self._wait_for(
                    window,
                    lambda: window.statusBar().currentMessage().startswith("Ready"),
                )
                self.assertEqual(window.active_project_path, project_path.resolve())
                self.assertEqual(window.current_workspace, "Review")
                self.assertEqual(window.current_image_asset.image_asset_id, asset.image_asset_id)
                self.assertEqual([job.job_id for job in window._jobs_controls.jobs], ["stale-job"])
                self.assertEqual(window._jobs_controls.jobs[0].status.value, "interrupted")
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()

    def test_startup_does_not_load_changed_source_and_warns(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for source_state, expected_warning in (
                ("changed", "Changed Source"),
                ("missing", "Missing Source"),
            ):
                with self.subTest(source_state=source_state):
                    project_path = root / f"project-{source_state}"
                    project.create_project(project_path)
                    source = root / f"wafer-{source_state}.png"
                    self._write_grayscale_png(source)
                    asset = image_asset.register_wafer_image(project_path, source)
                    if source_state == "changed":
                        source.write_bytes(source.read_bytes() + b"changed")
                    else:
                        source.unlink()
                    settings = QSettings(
                        str(root / f"settings-{source_state}.ini"),
                        QSettings.Format.IniFormat,
                    )
                    settings.setValue("activeProjectPath", str(project_path))
                    settings.setValue("currentWorkspace", "Annotate")
                    settings.setValue("currentImageAssetId", asset.image_asset_id)
                    settings.sync()

                    window = MainWindow(settings=settings)
                    window.show()
                    try:
                        app.processEvents()
                        self.assertEqual(window.active_project_path, project_path.resolve())
                        self.assertIsNone(window.current_image_asset)
                        self.assertIn(expected_warning, window.statusBar().currentMessage())
                    finally:
                        window.close()
                        window.deleteLater()
                        app.processEvents()


if __name__ == "__main__":
    unittest.main()
