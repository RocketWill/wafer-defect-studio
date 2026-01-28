import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication, QDockWidget, QPushButton, QSpinBox

from tests.test_grid_overlay import _render, _vertical_hits, _write_fixture
from wafer_defect_studio import image_asset, project
from wafer_defect_studio.grid_profile import load_grid_profiles, save_grid_profile
from wafer_defect_studio.image_grid_placement import (
    ImageGridPlacement,
    load_image_grid_placement,
    set_image_grid_origin,
)
from wafer_defect_studio.main_window import MainWindow


class ImageGridOriginTest(unittest.TestCase):
    def test_two_image_origins_persist_bind_and_apply_only(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "project"
            project.create_project(project_path)
            source_one = workspace / "wafer-one.png"
            source_two = workspace / "wafer-two.png"
            _write_fixture(source_one)
            _write_fixture(source_two)
            asset_one = image_asset.register_wafer_image(project_path, source_one)
            asset_two = image_asset.register_wafer_image(project_path, source_two)
            profile = save_grid_profile(project_path, 64, 32)
            _downgrade_to_v4(project_path)

            self.assertEqual(project.open_project(project_path).schema_version, 4)
            self.assertIsNone(load_image_grid_placement(project_path, asset_one.image_asset_id))
            first_placement = set_image_grid_origin(
                project_path, asset_one.image_asset_id, profile.grid_profile_id, 7, 5
            )
            second_placement = set_image_grid_origin(
                project_path, asset_two.image_asset_id, profile.grid_profile_id, 11, 9
            )
            self.assertEqual(project.open_project(project_path).schema_version, 5)
            self.assertEqual(
                first_placement,
                ImageGridPlacement(asset_one.image_asset_id, profile.grid_profile_id, 1, 7, 5),
            )
            self.assertEqual(
                second_placement,
                ImageGridPlacement(asset_two.image_asset_id, profile.grid_profile_id, 1, 11, 9),
            )
            self.assertEqual(load_image_grid_placement(project_path, asset_one.image_asset_id), first_placement)
            self.assertEqual(load_image_grid_placement(project_path, asset_two.image_asset_id), second_placement)

            window = MainWindow()
            window.resize(500, 400)
            window.show()
            app.processEvents()
            window.show_wafer_image(asset_one)
            window.set_grid_profile(project_path, profile)
            origin_dock = window.findChild(QDockWidget, "gridOriginDock")
            origin_x = window.findChild(QSpinBox, "gridOriginXSpinBox")
            origin_y = window.findChild(QSpinBox, "gridOriginYSpinBox")
            apply_origin = window.findChild(QPushButton, "applyGridOriginButton")
            self.assertTrue(origin_dock.isEnabled())
            self.assertEqual((origin_x.value(), origin_y.value()), (7, 5))
            self.assertEqual((origin_x.minimum(), origin_x.maximum()), (0, 63))
            self.assertEqual((origin_y.minimum(), origin_y.maximum()), (0, 31))
            self.assertFalse(apply_origin.isEnabled())

            view = window.centralWidget()
            view.resetTransform()
            view.scale(1.0, 1.0)
            before_database = (project_path / "project.sqlite").read_bytes()
            self.assertEqual(_vertical_hits(view, _render(view), (7, 11, 20), 20), (7,))
            origin_x.setValue(20)
            origin_y.setValue(6)
            self.assertTrue(apply_origin.isEnabled())
            self.assertEqual((project_path / "project.sqlite").read_bytes(), before_database)
            self.assertEqual(_vertical_hits(view, _render(view), (7, 11, 20), 20), (7,))

            apply_origin.click()
            app.processEvents()
            applied = load_image_grid_placement(project_path, asset_one.image_asset_id)
            self.assertEqual((applied.origin_x, applied.origin_y), (20, 6))
            self.assertFalse(apply_origin.isEnabled())
            self.assertEqual(_vertical_hits(view, _render(view), (7, 11, 20), 20), (20,))

            window.show_wafer_image(asset_two)
            app.processEvents()
            self.assertEqual((origin_x.value(), origin_y.value()), (11, 9))
            self.assertEqual(_vertical_hits(view, _render(view), (11, 20), 20), (11,))

            window.close()
            del view, window
            app.processEvents()

            reopened_window = MainWindow()
            reopened_window.resize(500, 400)
            reopened_window.show()
            app.processEvents()
            reopened_window.show_wafer_image(asset_one)
            reopened_window.set_grid_profile(project_path, profile)
            reopened_view = reopened_window.centralWidget()
            self.assertEqual(
                (reopened_window.findChild(QSpinBox, "gridOriginXSpinBox").value(),
                 reopened_window.findChild(QSpinBox, "gridOriginYSpinBox").value()),
                (20, 6),
            )
            reopened_window.close()
            del reopened_view, reopened_window
            app.processEvents()


def _downgrade_to_v4(project_path: Path) -> None:
    with sqlite3.connect(project_path / "project.sqlite") as connection:
        connection.execute("DROP TABLE IF EXISTS image_grid_placements")
        connection.execute("UPDATE project_metadata SET schema_version = 4")
        connection.execute("PRAGMA user_version = 4")


if __name__ == "__main__":
    unittest.main()
