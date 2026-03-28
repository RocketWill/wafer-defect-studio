import os
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QCheckBox, QFileDialog, QLabel, QPushButton

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.defect_class import DefectClass, save_defect_classes
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.training_scope import (
    DataGroup,
    assign_image_to_data_group,
    save_data_groups,
)


class DatasetSnapshotProjectPreviewTest(unittest.TestCase):
    def test_selection_shows_truthful_project_preview_without_writes(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = root / "project"
            project.create_project(project_path)
            sources = {}
            for name in ("available", "unreviewed", "changed", "missing"):
                source = root / f"{name}.png"
                image = QImage(2, 2, QImage.Format.Format_Grayscale8)
                image.fill(80)
                self.assertTrue(image.save(str(source), "PNG"))
                sources[name] = source
            assets = {
                name: image_asset.register_wafer_image(project_path, source)
                for name, source in sources.items()
            }
            save_defect_classes(
                project_path,
                (DefectClass("scratch", "Scratch", "#cc4444"),),
            )
            save_data_groups(project_path, (DataGroup("line-a", "Line A"),))
            for asset in assets.values():
                assign_image_to_data_group(project_path, asset.image_asset_id, "line-a")
            _seed_review_facts(project_path, assets)
            sources["changed"].write_bytes(sources["changed"].read_bytes() + b"changed")
            sources["missing"].unlink()
            database = project_path / "project.sqlite"
            database_bytes = database.read_bytes()

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

                warnings = window.findChild(QLabel, "datasetWarningsLabel")
                status = window.findChild(QLabel, "datasetSnapshotStatusLabel")
                self.assertIn("invalid source", warnings.text().lower())
                self.assertIn("unreviewed", warnings.text().lower())
                self.assertIn("eligible: 1", status.text())
                self.assertIn("invalid: 3", status.text())
                self.assertIn("Groups — line-a: 1", status.text())
                self.assertIn("Classes — scratch: 1", status.text())
                self.assertTrue(
                    window.findChild(QPushButton, "createDatasetSnapshotButton").isEnabled()
                )
                self.assertEqual(database.read_bytes(), database_bytes)
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


def _seed_review_facts(project_path: Path, assets: dict[str, image_asset.ImageAsset]) -> None:
    connection = sqlite3.connect(project_path / "project.sqlite")
    try:
        connection.executemany(
            "INSERT INTO image_reviews(image_asset_id, reviewed) VALUES (?, ?)",
            (
                (assets["available"].image_asset_id, 1),
                (assets["unreviewed"].image_asset_id, 0),
                (assets["changed"].image_asset_id, 1),
                (assets["missing"].image_asset_id, 1),
            ),
        )
        connection.execute(
            "INSERT INTO grid_annotations(image_asset_id, row, column, class_codes_json) "
            "VALUES (?, 0, 0, ?)",
            (assets["available"].image_asset_id, '["scratch"]'),
        )
        connection.execute(
            "INSERT INTO grid_annotations(image_asset_id, row, column, class_codes_json) "
            "VALUES (?, 0, 0, ?)",
            (assets["unreviewed"].image_asset_id, '["scratch"]'),
        )
        connection.commit()
    finally:
        connection.close()


if __name__ == "__main__":
    unittest.main()
