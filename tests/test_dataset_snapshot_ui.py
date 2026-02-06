import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QPushButton

from wafer_defect_studio.dataset_diagnostics import PreviewImage, preview_dataset
from wafer_defect_studio.training_scope import DataGroup
from wafer_defect_studio.defect_class import DefectClass
from wafer_defect_studio.main_window import MainWindow


class DatasetSnapshotUiTest(unittest.TestCase):
    def test_user_sees_scope_warnings_and_nonblocking_snapshot_result(self):
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        preview = preview_dataset(
            ("scratch",),
            (PreviewImage("wafer-1", "line-a", ("scratch",), 1),),
        )
        calls = []

        def create_snapshot(group_ids, class_codes):
            calls.append((group_ids, class_codes))
            return "snapshot-42", preview

        window.configure_dataset_snapshot(
            (DataGroup("line-a", "Line A"),),
            (DefectClass("scratch", "Scratch", "#cc4444"),),
            preview,
            create_snapshot,
        )
        window.show()
        app.processEvents()

        warning = window.findChild(QLabel, "datasetWarningsLabel")
        status = window.findChild(QLabel, "datasetSnapshotStatusLabel")
        create = window.findChild(QPushButton, "createDatasetSnapshotButton")
        self.assertIn("Select at least 10", warning.text())

        create.click()
        self.assertEqual(status.text(), "Select at least one Data Group and Defect Class.")
        self.assertEqual(calls, [])

        window.findChild(QCheckBox, "dataGroup_line-aCheckBox").click()
        window.findChild(QCheckBox, "snapshotClass_scratchCheckBox").click()
        create.click()
        self.assertFalse(create.isEnabled())
        self.assertEqual(status.text(), "Creating snapshot…")

        deadline = time.monotonic() + 2
        while not create.isEnabled() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)

        self.assertEqual(calls, [(('line-a',), ('scratch',))])
        self.assertIn("snapshot-42", status.text())
        self.assertIn("eligible: 1", status.text())
        window.close()


if __name__ == "__main__":
    unittest.main()
