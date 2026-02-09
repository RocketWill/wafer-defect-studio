import os
import queue
import time
import unittest
from dataclasses import dataclass

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.training_protocol import (
    ProgressMessage,
    TerminalMessage,
    TrainingConfig,
    TrainingRequest,
)


@dataclass
class _FakeHandle:
    queue: queue.Queue
    alive: bool = True
    cancelled: bool = False

    def cancel(self, reason="user"):
        self.cancelled = True
        return True

    def is_alive(self):
        return self.alive


class TrainingUiTest(unittest.TestCase):
    def test_background_training_progress_cancel_and_oom_clone(self):
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        request = TrainingRequest(
            request_id="ui-run",
            config=TrainingConfig(
                snapshot_id="snapshot-1",
                split_id="split-1",
                class_count=2,
                epochs=2,
                batch_size=4,
                device="auto",
            ),
            artifact_staging_path="runs/.staging/ui-run",
        )
        handle = _FakeHandle(queue.Queue())
        started = []
        clones = []

        def launcher(value):
            started.append(value)
            return handle

        window.configure_training(request, launcher=launcher, clone_callback=lambda: clones.append(True))
        window.show()
        app.processEvents()
        start = window.findChild(QPushButton, "startTrainingButton")
        cancel = window.findChild(QPushButton, "cancelTrainingButton")
        clone = window.findChild(QPushButton, "cloneOomTrainingButton")
        status = window.findChild(QLabel, "trainingStatusLabel")
        progress = window.findChild(QLabel, "trainingProgressLabel")
        start.click()
        self.assertEqual(started, [request])
        self.assertFalse(start.isEnabled())
        self.assertTrue(cancel.isEnabled())

        handle.queue.put(
            ProgressMessage(
                request_id="ui-run",
                phase="train",
                epoch=1,
                total_epochs=2,
                step=1,
                total_steps=4,
                loss=0.25,
                eta_seconds=1.5,
                message="device=cuda batch_size=4",
            )
        )
        QTest.qWait(100)
        self.assertIn("Epoch: 1/2", progress.text())
        self.assertIn("Loss: 0.250000", progress.text())
        cancel.click()
        self.assertTrue(handle.cancelled)
        handle.queue.put(
            TerminalMessage(
                request_id="ui-run",
                status="failed",
                error_code="out_of_memory",
                message="Out of memory; batch_size=4 unchanged.",
            )
        )
        QTest.qWait(100)
        self.assertIn("failed", status.text().lower())
        self.assertTrue(clone.isEnabled())
        clone.click()
        self.assertEqual(clones, [True])
        self.assertFalse(clone.isEnabled())
        window.close()


if __name__ == "__main__":
    unittest.main()
