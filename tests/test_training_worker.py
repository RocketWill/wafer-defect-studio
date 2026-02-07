import json
import multiprocessing as mp
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from wafer_defect_studio.training_protocol import (
    ProgressMessage,
    TerminalMessage,
    TrainingConfig,
    TrainingRequest,
    decode_message,
)
from wafer_defect_studio.training_run import validate_staged_artifacts
from wafer_defect_studio.training_worker import start_training_worker


class TrainingWorkerTest(unittest.TestCase):
    def _request(self, staging: Path, *, batch_size: int = 1) -> TrainingRequest:
        return TrainingRequest(
            request_id="worker-test",
            config=TrainingConfig(
                snapshot_id="snapshot-1",
                split_id="split-1",
                class_count=2,
                epochs=1,
                batch_size=batch_size,
                device="cpu",
                weights_policy="none",
            ),
            artifact_staging_path=staging,
        )

    def _collect(self, handle):
        messages = []
        while True:
            message = decode_message(handle.queue.get(timeout=30))
            messages.append(message)
            if isinstance(message, TerminalMessage):
                handle.join(timeout=30)
                self.assertFalse(handle.is_alive())
                return messages

    def test_spawned_training_reports_progress_cancel_and_oom_truthfully(self):
        # The spawn context is required on Windows and proves the worker does
        # not depend on a GUI thread or a project SQLite connection.
        self.assertEqual(mp.get_context("spawn").get_start_method(), "spawn")
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            completed_request = self._request(root / "completed")
            completed = start_training_worker(
                completed_request,
                synthetic_steps=1,
                step_delay=0.0,
            )
            completed_messages = self._collect(completed)
            progress = [item for item in completed_messages if isinstance(item, ProgressMessage)]
            terminal = completed_messages[-1]
            self.assertTrue(progress)
            self.assertEqual(progress[-1].phase, "train")
            self.assertEqual(progress[-1].epoch, 1)
            self.assertEqual(progress[-1].step, 1)
            self.assertIsNotNone(progress[-1].loss)
            self.assertIn("batch_size=1", progress[-1].message)
            self.assertIsInstance(terminal, TerminalMessage)
            self.assertEqual(terminal.status, "completed")
            self.assertTrue((root / "completed" / "manifest.json").is_file())
            manifest = json.loads((root / "completed" / "manifest.json").read_text())
            self.assertIn("model.pt", {entry["path"] for entry in manifest["files"]})
            validated = validate_staged_artifacts(root / "completed")
            self.assertEqual({path.name for path in validated}, {"model.pt", "metrics.json"})

            cancelled_request = self._request(root / "cancelled")
            cancelled = start_training_worker(
                cancelled_request,
                synthetic_steps=2,
                step_delay=0.2,
            )
            cancelled.cancel()
            cancelled_messages = self._collect(cancelled)
            self.assertEqual(cancelled_messages[-1].status, "cancelled")
            self.assertIn("cancel", cancelled_messages[-1].message.lower())
            self.assertFalse((root / "cancelled" / "manifest.json").exists())

            oom_request = self._request(root / "oom", batch_size=7)
            oom = start_training_worker(
                oom_request,
                synthetic_steps=1,
                inject_oom_step=1,
            )
            oom_messages = self._collect(oom)
            oom_terminal = oom_messages[-1]
            self.assertEqual(oom_terminal.status, "failed")
            self.assertEqual(oom_terminal.error_code, "out_of_memory")
            self.assertIn("batch_size=7", oom_terminal.message)
            self.assertIn("unchanged", oom_terminal.message.lower())
            self.assertFalse((root / "oom" / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
