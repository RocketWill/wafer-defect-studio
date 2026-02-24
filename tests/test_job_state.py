import unittest

from wafer_defect_studio.job_state import (
    JobKind,
    JobStatus,
    JobSnapshot,
    finish_job,
    request_job_cancel,
    start_job,
    update_job_progress,
)


class JobStateTest(unittest.TestCase):
    def test_truthful_progress_cancellation_and_completion_guard(self):
        queued = JobSnapshot("job-1", JobKind.DETECTION, JobStatus.QUEUED, total=10)
        running = start_job(queued, heartbeat_at="2026-08-22T12:00:00+00:00")
        self.assertEqual(running.status, JobStatus.RUNNING)
        progress = update_job_progress(
            running,
            completed=4,
            phase="windows",
            eta_seconds=2.5,
            message="Evaluating",
            heartbeat_at="2026-08-22T12:00:01+00:00",
        )
        self.assertEqual((progress.completed, progress.total, progress.phase), (4, 10, "windows"))
        self.assertEqual(progress.eta_seconds, 2.5)

        cancelling = request_job_cancel(progress)
        self.assertEqual(cancelling.status, JobStatus.CANCELLING)
        interrupted = finish_job(cancelling, JobStatus.INTERRUPTED, error_code="worker_killed")
        self.assertEqual(interrupted.status, JobStatus.INTERRUPTED)
        with self.assertRaises(ValueError):
            update_job_progress(interrupted, completed=5)

        with self.assertRaises(ValueError):
            finish_job(running, JobStatus.COMPLETED)
        completed = finish_job(
            progress,
            JobStatus.COMPLETED,
            artifact_validated=True,
            staged_artifact_path="stage/maps",
        )
        self.assertEqual(completed.status, JobStatus.COMPLETED)
        with self.assertRaises(ValueError):
            finish_job(completed, JobStatus.FAILED, error_code="late")


if __name__ == "__main__":
    unittest.main()
