import sqlite3
import tempfile
import unittest
from pathlib import Path

from wafer_defect_studio import project
from wafer_defect_studio.job_state import (
    JobKind,
    JobSnapshot,
    JobStatus,
    finish_job,
    start_job,
    update_job_progress,
)
from wafer_defect_studio.job_store import (
    JobStoreError,
    create_job,
    list_jobs,
    load_job,
    save_job,
)


class JobStoreTest(unittest.TestCase):
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

    def test_create_migrates_and_round_trips_progress_with_terminal_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._schema18_project(Path(temporary))
            queued = JobSnapshot(
                "job-α",
                JobKind.TRAINING,
                JobStatus.QUEUED,
                total=8,
                message="等待資料",
                log_path="logs/訓練.log",
                staged_artifact_path="runs/暫存",
            )
            self.assertEqual(create_job(path, queued), queued)
            self.assertEqual(project.open_project(path).schema_version, 19)
            self.assertEqual(load_job(path, queued.job_id), queued)

            with self.assertRaises(JobStoreError):
                create_job(path, queued)

            progress = update_job_progress(
                start_job(queued),
                completed=3,
                phase="epoch",
                eta_seconds=4.5,
                message="進行中",
            )
            self.assertEqual(save_job(path, progress), progress)
            self.assertEqual(load_job(path, queued.job_id), progress)
            self.assertEqual(list_jobs(path), (progress,))

            failed = finish_job(progress, JobStatus.FAILED, error_code="oom")
            self.assertEqual(save_job(path, failed), failed)
            with self.assertRaises(JobStoreError):
                save_job(path, progress)


if __name__ == "__main__":
    unittest.main()
