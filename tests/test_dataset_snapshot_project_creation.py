import os
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
from wafer_defect_studio.dataset_snapshot import load_dataset_snapshot
from wafer_defect_studio.defect_class import DefectClass, save_defect_classes
from wafer_defect_studio.effective_area import (
    confirm_effective_wafer_area,
    set_effective_ellipse,
)
from wafer_defect_studio.grid_profile import save_grid_profile
from wafer_defect_studio.image_grid_placement import set_image_grid_origin
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.review import mark_image_reviewed
from wafer_defect_studio.training_scope import (
    DataGroup,
    TrainingScope,
    assign_image_to_data_group,
    load_training_scope,
    save_data_groups,
)


class DatasetSnapshotProjectCreationTest(unittest.TestCase):
    def test_project_bound_selection_creates_snapshot_asynchronously(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = root / "project"
            project.create_project(project_path)
            source = root / "wafer.png"
            _write_grayscale(source)
            asset = image_asset.register_wafer_image(project_path, source)
            profile = save_grid_profile(project_path, 2, 2)
            set_image_grid_origin(project_path, asset.image_asset_id, profile.grid_profile_id, 0, 0)
            set_effective_ellipse(project_path, asset.image_asset_id, 1, 1, 1, 1)
            confirm_effective_wafer_area(project_path, asset.image_asset_id)
            save_defect_classes(
                project_path,
                (DefectClass("scratch", "Scratch", "#cc4444"),),
            )
            save_grid_annotation(
                project_path,
                GridAnnotation(asset.image_asset_id, 0, 0, ("scratch",)),
            )
            mark_image_reviewed(project_path, asset.image_asset_id)
            save_data_groups(project_path, (DataGroup("line-a", "Line A"),))
            assign_image_to_data_group(project_path, asset.image_asset_id, "line-a")

            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
            window = MainWindow(settings=settings)
            window.show()
            app.processEvents()
            try:
                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(project_path),
                ):
                    window.open_project_action.trigger()
                window.workspace_actions["Dataset"].trigger()
                app.processEvents()
                window.findChild(QCheckBox, "dataGroup_line-aCheckBox").click()
                window.findChild(QCheckBox, "snapshotClass_scratchCheckBox").click()
                app.processEvents()
                create = window.findChild(QPushButton, "createDatasetSnapshotButton")
                self.assertTrue(create.isEnabled())
                create.click()
                self.assertFalse(create.isEnabled())

                deadline = time.monotonic() + 5
                while not create.isEnabled() and time.monotonic() < deadline:
                    app.processEvents()
                    time.sleep(0.01)

                self.assertTrue(create.isEnabled())
                self.assertIn(
                    "Created",
                    window.findChild(QLabel, "datasetSnapshotStatusLabel").text(),
                )
                scope = load_training_scope(project_path)
                self.assertEqual(scope, TrainingScope(("line-a",), ("scratch",)))
                snapshot_id = _snapshot_id(project_path)
                self.assertIsNotNone(load_dataset_snapshot(project_path, snapshot_id))
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


def _write_grayscale(path: Path) -> None:
    image = QImage(2, 2, QImage.Format.Format_Grayscale8)
    bits = image.bits()
    bits[:4] = bytes((0, 64, 128, 255))
    if not image.save(str(path), "PNG"):
        raise AssertionError(f"Unable to write {path}")
    del bits, image


def _snapshot_id(project_path: Path) -> str:
    import sqlite3

    connection = sqlite3.connect(project_path / "project.sqlite")
    try:
        row = connection.execute("SELECT snapshot_id FROM dataset_snapshots").fetchone()
    finally:
        connection.close()
    if row is None:
        raise AssertionError("Dataset Snapshot was not persisted")
    return row[0]


if __name__ == "__main__":
    unittest.main()
