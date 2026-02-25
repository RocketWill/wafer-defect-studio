import unittest

from wafer_defect_studio.job_actions import (
    FailureKind,
    RecoveryAction,
    available_recovery_actions,
    diagnose_failure,
    plan_recovery,
)
from wafer_defect_studio.job_state import JobKind, JobSnapshot, JobStatus


class JobActionsTest(unittest.TestCase):
    def test_failure_diagnostics_and_safe_recovery_plans(self):
        oom = JobSnapshot(
            "oom-1",
            JobKind.TRAINING,
            JobStatus.FAILED,
            error_code="out_of_memory",
            message="CUDA out of memory",
            log_path="logs/oom.log",
            staged_artifact_path="stage/oom",
        )
        diagnostic = diagnose_failure(oom)
        self.assertEqual(diagnostic.kind, FailureKind.OUT_OF_MEMORY)
        self.assertEqual(
            available_recovery_actions(oom),
            (RecoveryAction.CLONE_ADJUSTED, RecoveryAction.INSPECT_LOG),
        )
        clone = plan_recovery(
            oom,
            RecoveryAction.CLONE_ADJUSTED,
            "oom-2",
            config_overrides={"batch_size": 2},
            current_batch_size=4,
        )
        self.assertEqual(clone.new_job.status, JobStatus.QUEUED)
        self.assertEqual((clone.new_job.attempt, clone.new_job.parent_job_id), (2, "oom-1"))
        self.assertEqual(clone.config_overrides["batch_size"], 2)
        with self.assertRaises(ValueError):
            plan_recovery(
                oom,
                RecoveryAction.CLONE_ADJUSTED,
                "oom-3",
                config_overrides={"batch_size": 4},
                current_batch_size=4,
            )

        source = r"Z:\資料\wafer.tif"
        missing = JobSnapshot(
            "missing-1",
            JobKind.DETECTION,
            JobStatus.FAILED,
            error_code="missing_source",
            message="Source was not found",
        )
        self.assertEqual(diagnose_failure(missing, source).source_path, source)
        self.assertEqual(
            available_recovery_actions(missing),
            (RecoveryAction.INSPECT_LOG, RecoveryAction.RESTART),
        )

        disk = JobSnapshot(
            "disk-1",
            JobKind.EVALUATION,
            JobStatus.FAILED,
            error_code="disk_full",
            message="No space left on device",
            log_path="logs/disk.log",
            staged_artifact_path="stage/disk",
        )
        disk_plan = plan_recovery(disk, RecoveryAction.RETRY, "disk-2")
        self.assertEqual(disk_plan.new_job.parent_job_id, "disk-1")
        self.assertEqual(disk_plan.diagnostic.staged_artifact_path, "stage/disk")
        self.assertEqual(disk_plan.new_job.attempt, 2)

        completed = JobSnapshot("done", JobKind.DETECTION, JobStatus.COMPLETED)
        with self.assertRaises(ValueError):
            plan_recovery(completed, RecoveryAction.RETRY, "done-2")


if __name__ == "__main__":
    unittest.main()
