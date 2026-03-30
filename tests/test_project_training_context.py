import os
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QComboBox, QFileDialog, QLabel, QPushButton, QSpinBox

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.annotation import GridAnnotation, save_grid_annotation
from wafer_defect_studio.dataset_split import create_dataset_split
from wafer_defect_studio.dataset_workflow import create_project_dataset_snapshot
from wafer_defect_studio.defect_class import DefectClass, save_defect_classes
from wafer_defect_studio.effective_area import confirm_effective_wafer_area, set_effective_ellipse
from wafer_defect_studio.grid_profile import save_grid_profile
from wafer_defect_studio.image_grid_placement import set_image_grid_origin
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.review import mark_image_reviewed
from wafer_defect_studio.training_scope import DataGroup, assign_image_to_data_group, save_data_groups


class ProjectTrainingContextTest(unittest.TestCase):
    def test_reopen_restores_training_context_and_missing_snapshot_is_not_ready(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path, snapshot_id, split_id = _seed_project(root)
            database = project_path / "project.sqlite"
            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)

            first = MainWindow(settings=settings)
            first.show()
            app.processEvents()
            with patch.object(
                QFileDialog,
                "getExistingDirectory",
                return_value=str(project_path),
            ):
                first.open_project_action.trigger()
            first.workspace_actions["Train"].trigger()
            first.findChild(QSpinBox, "trainingEpochsSpinBox").setValue(3)
            first.findChild(QSpinBox, "trainingSeedSpinBox").setValue(17)
            app.processEvents()
            first.close()
            first.deleteLater()
            app.processEvents()

            database_bytes = database.read_bytes()
            second = MainWindow(settings=settings)
            second.show()
            app.processEvents()
            self.assertEqual(second.current_workspace, "Train")
            self.assertEqual(
                second.findChild(QComboBox, "trainingSnapshotComboBox").currentData(),
                snapshot_id,
            )
            self.assertEqual(
                second.findChild(QComboBox, "trainingSplitComboBox").currentData(),
                split_id,
            )
            self.assertEqual(
                second.findChild(QSpinBox, "trainingEpochsSpinBox").value(), 3
            )
            self.assertEqual(second.findChild(QSpinBox, "trainingSeedSpinBox").value(), 17)
            self.assertTrue(second.findChild(QPushButton, "startTrainingButton").isEnabled())
            second.close()
            second.deleteLater()
            app.processEvents()

            project_id = project.open_project(project_path).project_id
            settings.setValue(f"trainingContext/{project_id}/snapshotId", "missing")
            settings.sync()
            third = MainWindow(settings=settings)
            third.show()
            app.processEvents()
            self.assertIn(
                "Snapshot unavailable: missing",
                third.findChild(QLabel, "trainingContextStatusLabel").text(),
            )
            self.assertFalse(third.findChild(QPushButton, "startTrainingButton").isEnabled())
            self.assertEqual(database.read_bytes(), database_bytes)
            third.close()
            third.deleteLater()
            app.processEvents()


def _seed_project(root: Path) -> tuple[Path, str, str]:
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
    snapshot_id, _preview = create_project_dataset_snapshot(
        project_path, ("line-a",), ("scratch",)
    )
    split = create_dataset_split(project_path, snapshot_id, 42)
    return project_path, snapshot_id, split.split_id


if __name__ == "__main__":
    unittest.main()
