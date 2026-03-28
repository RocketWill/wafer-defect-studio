import os
import sqlite3
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QCheckBox, QFileDialog, QLabel, QPushButton

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.annotation import GridAnnotation, save_grid_annotation
from wafer_defect_studio.defect_class import DefectClass, save_defect_classes
from wafer_defect_studio.effective_area import confirm_effective_wafer_area, set_effective_ellipse
from wafer_defect_studio.grid_profile import save_grid_profile
from wafer_defect_studio.image_grid_placement import set_image_grid_origin
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.review import mark_image_reviewed
from wafer_defect_studio.training_scope import DataGroup, assign_image_to_data_group, save_data_groups


class DatasetSnapshotContextRestoreTest(unittest.TestCase):
    def test_valid_and_missing_snapshot_context_restore_without_project_writes(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = _seed_project(root)
            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)

            first = MainWindow(settings=settings)
            first.show()
            app.processEvents()
            with patch.object(QFileDialog, "getExistingDirectory", return_value=str(project_path)):
                first.open_project_action.trigger()
            first.workspace_actions["Dataset"].trigger()
            app.processEvents()
            first.findChild(QCheckBox, "dataGroup_line-aCheckBox").click()
            first.findChild(QCheckBox, "snapshotClass_scratchCheckBox").click()
            create = first.findChild(QPushButton, "createDatasetSnapshotButton")
            create.click()
            deadline = time.monotonic() + 5
            while not create.isEnabled() and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)
            self.assertTrue(create.isEnabled())
            first.close()
            first.deleteLater()
            app.processEvents()

            database = project_path / "project.sqlite"
            database_bytes = database.read_bytes()
            second = MainWindow(settings=settings)
            second.show()
            app.processEvents()
            self.assertTrue(second.findChild(QCheckBox, "dataGroup_line-aCheckBox").isChecked())
            self.assertTrue(second.findChild(QCheckBox, "snapshotClass_scratchCheckBox").isChecked())
            self.assertIn("Restored snapshot:", second.statusBar().currentMessage())
            second.close()
            second.deleteLater()
            app.processEvents()

            project_id = project.open_project(project_path).project_id
            settings.setValue(f"datasetContext/{project_id}/snapshotId", "missing")
            settings.sync()
            third = MainWindow(settings=settings)
            third.show()
            app.processEvents()
            self.assertIn("Snapshot unavailable: missing", third.statusBar().currentMessage())
            self.assertEqual(database.read_bytes(), database_bytes)
            third.close()
            third.deleteLater()
            app.processEvents()


def _seed_project(root: Path) -> Path:
    project_path = root / "project"
    project.create_project(project_path)
    source = root / "wafer.png"
    image = QImage(2, 2, QImage.Format.Format_Grayscale8)
    image.bits()[:4] = bytes((0, 64, 128, 255))
    image.save(str(source), "PNG")
    asset = image_asset.register_wafer_image(project_path, source)
    profile = save_grid_profile(project_path, 2, 2)
    set_image_grid_origin(project_path, asset.image_asset_id, profile.grid_profile_id, 0, 0)
    set_effective_ellipse(project_path, asset.image_asset_id, 1, 1, 1, 1)
    confirm_effective_wafer_area(project_path, asset.image_asset_id)
    save_defect_classes(project_path, (DefectClass("scratch", "Scratch", "#cc4444"),))
    save_grid_annotation(project_path, GridAnnotation(asset.image_asset_id, 0, 0, ("scratch",)))
    mark_image_reviewed(project_path, asset.image_asset_id)
    save_data_groups(project_path, (DataGroup("line-a", "Line A"),))
    assign_image_to_data_group(project_path, asset.image_asset_id, "line-a")
    return project_path


if __name__ == "__main__":
    unittest.main()
