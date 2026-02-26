import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDockWidget, QLabel, QPushButton, QTableWidget

from wafer_defect_studio import project
from wafer_defect_studio.job_state import JobKind, JobSnapshot, JobStatus, start_job
from wafer_defect_studio.job_store import create_job
from wafer_defect_studio.main_window import MainWindow


class JobStartupTest(unittest.TestCase):
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

    def test_startup_recovers_stale_jobs_before_showing_panel(self):
        app = QApplication.instance() or QApplication([])
        now = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            path = self._schema18_project(Path(temporary))
            stage = path / "runs" / ".staging" / "stale-job"
            stage.mkdir(parents=True)
            stale = start_job(
                JobSnapshot(
                    "stale-job",
                    JobKind.DETECTION,
                    total=3,
                    heartbeat_at=(now - timedelta(seconds=120)).isoformat(),
                    staged_artifact_path=str(stage),
                ),
                heartbeat_at=(now - timedelta(seconds=120)).isoformat(),
            )
            oom = JobSnapshot(
                "oom-job",
                JobKind.TRAINING,
                JobStatus.FAILED,
                error_code="out_of_memory",
                message="CUDA out of memory",
            )
            create_job(path, stale)
            create_job(path, oom)
            calls = []

            window = MainWindow()
            window.show()
            window.configure_jobs_for_project(
                path,
                now=now,
                process_is_alive=lambda _job: True,
                action_callback=lambda snapshot, action: calls.append((snapshot.job_id, action)) or True,
            )
            dock = window.findChild(QDockWidget, "jobsDock")
            self.assertTrue(dock.isEnabled())
            self.assertTrue(dock.isVisible())
            table = window.findChild(QTableWidget, "jobsTable")
            self.assertEqual(table.rowCount(), 2)
            rows = {table.item(row, 0).text(): row for row in range(table.rowCount())}
            self.assertEqual(table.item(rows["stale-job"], 2).text(), "Interrupted")
            self.assertEqual(table.item(rows["oom-job"], 2).text(), "Failed")
            table.selectRow(rows["stale-job"])
            self.assertTrue(window.findChild(QPushButton, "restartJobButton").isEnabled())
            self.assertTrue(window.findChild(QPushButton, "inspectJobLogButton").isEnabled())
            self.assertTrue(stage.exists())
            window.close()

            failed_window = MainWindow()
            failed_window.show()
            with patch("wafer_defect_studio.main_window.recover_stale_jobs", side_effect=RuntimeError("database unavailable")):
                failed_window.configure_jobs_for_project(path, now=now)
            self.assertIn(
                "Jobs recovery failed: database unavailable",
                failed_window.findChild(QLabel, "jobsStatusLabel").text(),
            )
            self.assertTrue(failed_window.findChild(QDockWidget, "jobsDock").isVisible())
            failed_window.close()

        app.processEvents()


if __name__ == "__main__":
    unittest.main()
