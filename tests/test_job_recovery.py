import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from wafer_defect_studio import project
from wafer_defect_studio.job_recovery import (
    StagingDisposition,
    recover_stale_jobs,
    scan_staging,
)
from wafer_defect_studio.job_state import JobKind, JobSnapshot, JobStatus, start_job
from wafer_defect_studio.job_store import create_job, save_job


class JobRecoveryTest(unittest.TestCase):
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

    def test_recover_stale_active_jobs_and_scan_without_mutating_staging(self):
        now = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            path = self._schema18_project(Path(temporary))
            staging = path / "runs" / ".staging"
            staging.mkdir()
            referenced = staging / "stale-job"
            referenced.mkdir()
            orphan = staging / "orphan.bin"
            orphan.write_text("keep", encoding="utf-8")
            missing = staging / "missing-job"

            stale = start_job(
                JobSnapshot(
                    "stale-job",
                    JobKind.DETECTION,
                    total=4,
                    staged_artifact_path=str(referenced),
                    heartbeat_at=(now - timedelta(seconds=120)).isoformat(),
                ),
                heartbeat_at=(now - timedelta(seconds=120)).isoformat(),
            )
            fresh = start_job(
                JobSnapshot(
                    "fresh-job",
                    JobKind.DETECTION,
                    total=4,
                    heartbeat_at=(now - timedelta(seconds=5)).isoformat(),
                ),
                heartbeat_at=(now - timedelta(seconds=5)).isoformat(),
            )
            queued = JobSnapshot("queued-job", JobKind.TRAINING)
            missing_reference = start_job(
                JobSnapshot(
                    "missing-job",
                    JobKind.TRAINING,
                    staged_artifact_path=str(missing),
                    heartbeat_at=(now - timedelta(seconds=120)).isoformat(),
                ),
                heartbeat_at=(now - timedelta(seconds=120)).isoformat(),
            )
            for snapshot in (stale, fresh, queued, missing_reference):
                create_job(path, snapshot)

            recovered = recover_stale_jobs(
                path,
                now=now,
                stale_after_seconds=60,
                process_is_alive=lambda job: job.job_id != "missing-job",
            )
            self.assertEqual([job.job_id for job in recovered], ["missing-job", "stale-job"])
            self.assertTrue(all(job.status is JobStatus.INTERRUPTED for job in recovered))
            self.assertEqual((path / "runs" / ".staging" / "orphan.bin").read_text(encoding="utf-8"), "keep")

            records = scan_staging(path)
            by_path = {record.path: record for record in records}
            self.assertEqual(by_path[referenced].disposition, StagingDisposition.REFERENCED)
            self.assertEqual(by_path[missing].disposition, StagingDisposition.MISSING)
            self.assertEqual(by_path[orphan].disposition, StagingDisposition.UNREFERENCED)
            self.assertTrue(referenced.exists())
            self.assertTrue(orphan.exists())


if __name__ == "__main__":
    unittest.main()
