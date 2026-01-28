import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QGraphicsEllipseItem, QGraphicsPixmapItem

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.effective_area import (
    EffectiveWaferArea,
    EllipseGeometry,
    load_effective_wafer_area,
    set_effective_ellipse,
)
from wafer_defect_studio.main_window import MainWindow


class EllipseEffectiveAreaTest(unittest.TestCase):
    def test_ellipse_migrates_persists_reopens_and_rebuilds_overlay(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "project"
            project.create_project(project_path)
            source_path = workspace / "wafer.png"
            _write_fixture(source_path)
            asset = image_asset.register_wafer_image(project_path, source_path)
            _downgrade_to_v5(project_path)

            self.assertEqual(project.open_project(project_path).schema_version, 5)
            self.assertIsNone(load_effective_wafer_area(project_path, asset.image_asset_id))
            area = set_effective_ellipse(
                project_path, asset.image_asset_id, 50, 40, 30, 20
            )
            expected = EffectiveWaferArea(
                asset.image_asset_id,
                "ellipse",
                EllipseGeometry(50, 40, 30, 20),
                False,
            )
            self.assertEqual(area, expected)
            self.assertEqual(project.open_project(project_path).schema_version, 6)
            with sqlite3.connect(project_path / "project.sqlite") as connection:
                row = connection.execute(
                    "SELECT shape, geometry_json, confirmed FROM effective_wafer_areas"
                ).fetchone()
            connection.close()
            self.assertEqual(row, ("ellipse", '{"center_x":50,"center_y":40,"radius_x":30,"radius_y":20}', 0))
            self.assertEqual(load_effective_wafer_area(project_path, asset.image_asset_id), expected)

            # An edit replaces the row and always clears an earlier confirmation.
            with sqlite3.connect(project_path / "project.sqlite") as connection:
                connection.execute(
                    "UPDATE effective_wafer_areas SET confirmed = 1 WHERE image_asset_id = ?",
                    (asset.image_asset_id,),
                )
            connection.close()
            edited = set_effective_ellipse(project_path, asset.image_asset_id, 48, 38, 25, 15)
            self.assertFalse(edited.confirmed)
            self.assertEqual(load_effective_wafer_area(project_path, asset.image_asset_id), edited)

            window = MainWindow()
            window.resize(500, 400)
            window.show()
            app.processEvents()
            window.show_wafer_image(asset)
            view = window.centralWidget()
            view.set_effective_wafer_area(edited)
            app.processEvents()
            items = view.scene().items()
            self.assertEqual(len(items), 2)
            ellipse = next(item for item in items if isinstance(item, QGraphicsEllipseItem))
            self.assertEqual(ellipse.rect(), QRectF(23, 23, 50, 30))
            self.assertEqual(ellipse.zValue(), 2)
            self.assertIsInstance(next(item for item in items if isinstance(item, QGraphicsPixmapItem)), QGraphicsPixmapItem)

            # The stored area survives a scene rebuild and can be replaced/cleared.
            window.show_wafer_image(asset)
            app.processEvents()
            self.assertEqual(
                next(item for item in view.scene().items() if isinstance(item, QGraphicsEllipseItem)).rect(),
                QRectF(23, 23, 50, 30),
            )
            replacement = EffectiveWaferArea(
                asset.image_asset_id, "ellipse", EllipseGeometry(50, 40, 10, 8), False
            )
            view.set_effective_wafer_area(replacement)
            self.assertEqual(
                len([item for item in view.scene().items() if isinstance(item, QGraphicsEllipseItem)]),
                1,
            )
            view.set_effective_wafer_area(None)
            self.assertFalse(any(isinstance(item, QGraphicsEllipseItem) for item in view.scene().items()))
            window.close()
            del view, window
            app.processEvents()


def _write_fixture(path: Path) -> None:
    image = QImage(100, 80, QImage.Format_Grayscale8)
    image.fill(0)
    if not image.save(str(path), "PNG"):
        raise AssertionError(f"Unable to write fixture: {path}")
    del image


def _downgrade_to_v5(project_path: Path) -> None:
    with sqlite3.connect(project_path / "project.sqlite") as connection:
        connection.execute("DROP TABLE IF EXISTS effective_wafer_areas")
        connection.execute("UPDATE project_metadata SET schema_version = 5")
        connection.execute("PRAGMA user_version = 5")


if __name__ == "__main__":
    unittest.main()
