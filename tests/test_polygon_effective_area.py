import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QGraphicsPixmapItem, QGraphicsPolygonItem

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.effective_area import (
    EffectiveWaferArea,
    PolygonGeometry,
    SourcePoint,
    load_effective_wafer_area,
    set_effective_polygon,
)
from wafer_defect_studio.main_window import MainWindow


class PolygonEffectiveAreaTest(unittest.TestCase):
    def test_polygon_persists_reopens_replaces_and_rejects_invalid_without_overwrite(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "project"
            project.create_project(project_path)
            source_path = workspace / "wafer.png"
            _write_fixture(source_path)
            asset = image_asset.register_wafer_image(project_path, source_path)

            valid = set_effective_polygon(
                project_path,
                asset.image_asset_id,
                [(10, 10), (40, 10), (80, 10), (80, 60), (10, 60)],
            )
            expected = EffectiveWaferArea(
                asset.image_asset_id,
                "polygon",
                PolygonGeometry(
                    (
                        SourcePoint(10, 10),
                        SourcePoint(40, 10),
                        SourcePoint(80, 10),
                        SourcePoint(80, 60),
                        SourcePoint(10, 60),
                    )
                ),
                False,
            )
            self.assertEqual(valid, expected)
            self.assertEqual(load_effective_wafer_area(project_path, asset.image_asset_id), expected)
            connection = sqlite3.connect(project_path / "project.sqlite")
            try:
                row = connection.execute(
                    "SELECT shape, geometry_json, confirmed FROM effective_wafer_areas"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(
                row,
                (
                    "polygon",
                    '{"vertices":[[10,10],[40,10],[80,10],[80,60],[10,60]]}',
                    0,
                ),
            )

            for invalid in (
                [(10, 10), (80, 10), (80, 60), (10, 60), (10, 10)],
                [(10, 10), (80, 60), (10, 60), (80, 10)],
                [(10, 10), (80, 10), (40, 10), (80, 60), (10, 60)],
                [(10, 10), (80, 10), (80, 60), (10, 81)],
            ):
                with self.assertRaises(ValueError):
                    set_effective_polygon(project_path, asset.image_asset_id, invalid)
                self.assertEqual(load_effective_wafer_area(project_path, asset.image_asset_id), expected)

            window = MainWindow()
            window.resize(500, 400)
            window.show()
            app.processEvents()
            window.show_wafer_image(asset)
            view = window.centralWidget()
            view.set_effective_wafer_area(valid)
            app.processEvents()
            polygons = [item for item in view.scene().items() if isinstance(item, QGraphicsPolygonItem)]
            self.assertEqual(len(polygons), 1)
            polygon = polygons[0]
            self.assertEqual(polygon.zValue(), 2)
            self.assertEqual(polygon.polygon().count(), 5)
            self.assertEqual(polygon.polygon()[0], QPointF(10, 10))
            self.assertEqual(polygon.polygon()[-1], QPointF(10, 60))

            window.show_wafer_image(asset)
            app.processEvents()
            self.assertEqual(
                len([item for item in view.scene().items() if isinstance(item, QGraphicsPolygonItem)]),
                1,
            )
            replacement = set_effective_polygon(
                project_path,
                asset.image_asset_id,
                [(20, 20), (70, 20), (60, 55), (20, 60)],
            )
            view.set_effective_wafer_area(replacement)
            self.assertEqual(
                len([item for item in view.scene().items() if isinstance(item, QGraphicsPolygonItem)]),
                1,
            )
            view.set_effective_wafer_area(None)
            self.assertFalse(any(isinstance(item, QGraphicsPolygonItem) for item in view.scene().items()))
            self.assertTrue(any(isinstance(item, QGraphicsPixmapItem) for item in view.scene().items()))
            window.close()
            del view, window
            app.processEvents()


def _write_fixture(path: Path) -> None:
    image = QImage(100, 80, QImage.Format_Grayscale8)
    image.fill(80)
    if not image.save(str(path), "PNG"):
        raise AssertionError(f"Unable to write fixture: {path}")
    del image


if __name__ == "__main__":
    unittest.main()
