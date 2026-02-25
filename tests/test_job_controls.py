import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDockWidget, QLabel, QPushButton, QTableWidget

from wafer_defect_studio.job_actions import RecoveryAction
from wafer_defect_studio.job_controls import JobsControls
from wafer_defect_studio.job_state import JobKind, JobSnapshot, JobStatus
from wafer_defect_studio.main_window import MainWindow


class JobControlsTest(unittest.TestCase):
    def test_jobs_panel_sorts_states_and_routes_nonmodal_actions(self):
        app = QApplication.instance() or QApplication([])
        running = JobSnapshot(
            "run-1",
            JobKind.DETECTION,
            JobStatus.RUNNING,
            phase="windows",
            completed=1,
            total=3,
            eta_seconds=4.0,
            message="Evaluating",
        )
        oom = JobSnapshot(
            "oom-1",
            JobKind.TRAINING,
            JobStatus.FAILED,
            error_code="out_of_memory",
            message="CUDA out of memory",
        )
        calls = []

        def action_callback(snapshot, action):
            calls.append((snapshot.job_id, action))
            return action != "cancel"

        controls = JobsControls()
        controls.configure((running, oom), action_callback=action_callback)
        table = controls.findChild(QTableWidget, "jobsTable")
        self.assertEqual([table.item(row, 0).text() for row in range(table.rowCount())], ["oom-1", "run-1"])
        self.assertEqual(table.item(0, 2).text(), "Failed")
        self.assertIn("1/3", table.item(1, 4).text())

        table.selectRow(1)
        cancel = controls.findChild(QPushButton, "cancelJobButton")
        self.assertTrue(cancel.isEnabled())
        cancel.click()
        self.assertEqual(calls[-1], ("run-1", "cancel"))
        self.assertIn("failed", controls.findChild(QLabel, "jobsStatusLabel").text().lower())

        table.selectRow(0)
        clone = controls.findChild(QPushButton, "cloneJobButton")
        inspect = controls.findChild(QPushButton, "inspectJobLogButton")
        self.assertTrue(clone.isEnabled())
        self.assertTrue(inspect.isEnabled())
        clone.click()
        inspect.click()
        self.assertEqual(calls[-2:], [("oom-1", RecoveryAction.CLONE_ADJUSTED), ("oom-1", RecoveryAction.INSPECT_LOG)])
        self.assertIsNone(app.activeModalWidget())

        window = MainWindow()
        dock = window.findChild(QDockWidget, "jobsDock")
        self.assertFalse(dock.isEnabled())
        self.assertFalse(dock.isVisible())
        window.show()
        window.configure_jobs((running, oom), action_callback=action_callback)
        self.assertTrue(dock.isEnabled())
        self.assertTrue(dock.isVisible())
        window.close()
        controls.close()


if __name__ == "__main__":
    unittest.main()
