import unittest
from types import SimpleNamespace

from wafer_defect_studio.job_state import JobKind, JobSnapshot, JobStatus
from wafer_defect_studio.job_worker_bridge import (
    WorkerBridgeError,
    apply_worker_message,
    request_worker_cancel,
)


class _Handle:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.reasons = []

    def cancel(self, reason="user"):
        self.reasons.append(reason)
        return self.accepted


class JobWorkerBridgeTest(unittest.TestCase):
    def test_progress_cancel_and_validated_publication_gates_completion(self):
        queued = JobSnapshot("job-1", JobKind.DETECTION, total=2)
        progress = apply_worker_message(
            queued,
            SimpleNamespace(
                request_id="job-1",
                phase="windows",
                completed=1,
                total=2,
                eta_seconds=3.0,
                message="Evaluating",
            ),
        )
        self.assertEqual((progress.status, progress.completed), (JobStatus.RUNNING, 1))

        handle = _Handle()
        cancelling = request_worker_cancel(progress, handle, reason="user")
        self.assertEqual(cancelling.status, JobStatus.CANCELLING)
        self.assertEqual(handle.reasons, ["user"])

        terminal = SimpleNamespace(
            request_id="job-1",
            status="completed",
            message="Worker completed",
            error_code=None,
            artifact_staging_path="stage/job-1",
        )
        corrupt = apply_worker_message(progress, terminal, validate_artifact=lambda _path: False)
        self.assertEqual((corrupt.status, corrupt.error_code), (JobStatus.FAILED, "corrupt_staging"))

        running = progress
        published = apply_worker_message(
            running,
            terminal,
            validate_artifact=lambda path: path == "stage/job-1",
            publish_artifact=lambda path: "published/job-1" if path == "stage/job-1" else None,
        )
        self.assertEqual(published.status, JobStatus.COMPLETED)
        self.assertEqual(published.staged_artifact_path, "published/job-1")

        failed_publish = apply_worker_message(
            running,
            terminal,
            validate_artifact=lambda _path: True,
            publish_artifact=lambda _path: None,
        )
        self.assertEqual(
            (failed_publish.status, failed_publish.error_code),
            (JobStatus.FAILED, "artifact_publish_failed"),
        )

        with self.assertRaises(WorkerBridgeError):
            request_worker_cancel(progress, _Handle(False))


if __name__ == "__main__":
    unittest.main()
