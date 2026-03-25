import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QCheckBox, QFileDialog, QLabel, QPushButton

from wafer_defect_studio import project
from wafer_defect_studio.defect_class import DefectClass, save_defect_classes
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.training_scope import DataGroup, save_data_groups


class DatasetSnapshotProjectContextTest(unittest.TestCase):
    def test_open_project_populates_dataset_options_without_writes(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty_project = root / "empty"
            seeded_project = root / "seeded"
            project.create_project(empty_project)
            project.create_project(seeded_project)
            save_defect_classes(
                seeded_project,
                (
                    DefectClass("scratch", "Scratch", "#cc4444", order=0),
                    DefectClass("archived", "Archived", "#777777", order=1, enabled=False),
                    DefectClass("particle", "Particle", "#4488cc", order=2),
                ),
            )
            save_data_groups(
                seeded_project,
                (
                    DataGroup("line-b", "Line B", order=1),
                    DataGroup("line-empty", "Empty Line", order=2),
                    DataGroup("line-a", "Line A", order=0),
                ),
            )
            seeded_bytes = (seeded_project / "project.sqlite").read_bytes()
            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
            window = MainWindow(settings=settings)
            window.show()
            app.processEvents()
            try:
                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(empty_project),
                ):
                    window.open_project_action.trigger()
                window.workspace_actions["Dataset"].trigger()
                app.processEvents()
                self.assertTrue(
                    window.findChild(QLabel, "datasetGroupsEmptyState").isVisible()
                )
                self.assertTrue(
                    window.findChild(QLabel, "datasetClassesEmptyState").isVisible()
                )
                self.assertFalse(
                    window.findChild(QPushButton, "createDatasetSnapshotButton").isEnabled()
                )

                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(seeded_project),
                ):
                    window.open_project_action.trigger()
                app.processEvents()

                self.assertEqual(
                    [
                        box.objectName()
                        for box in window.findChildren(QCheckBox)
                        if box.objectName().startswith(("dataGroup_", "snapshotClass_"))
                    ],
                    [
                        "dataGroup_line-aCheckBox",
                        "dataGroup_line-bCheckBox",
                        "dataGroup_line-emptyCheckBox",
                        "snapshotClass_scratchCheckBox",
                        "snapshotClass_particleCheckBox",
                    ],
                )
                self.assertFalse(
                    window.findChild(QLabel, "datasetGroupsEmptyState").isVisible()
                )
                self.assertFalse(
                    window.findChild(QLabel, "datasetClassesEmptyState").isVisible()
                )
                self.assertFalse(
                    window.findChild(QPushButton, "createDatasetSnapshotButton").isEnabled()
                )
                self.assertEqual((seeded_project / "project.sqlite").read_bytes(), seeded_bytes)
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
