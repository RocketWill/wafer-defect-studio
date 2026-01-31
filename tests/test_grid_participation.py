import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QSpinBox

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.effective_area import (
    EffectiveWaferArea,
    EllipseGeometry,
    participating_annotation_grids,
    set_effective_ellipse,
    set_effective_polygon,
)
from wafer_defect_studio.grid_geometry import annotation_grids
from wafer_defect_studio.grid_profile import save_grid_profile
from wafer_defect_studio.main_window import MainWindow


class GridParticipationTest(unittest.TestCase):
    def test_participation_rule_and_ui_count_follow_applied_transitions_only(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "project"
            project.create_project(project_path)
            source_path = workspace / "wafer.png"
            _write_fixture(source_path)
            asset = image_asset.register_wafer_image(project_path, source_path)
            profile = save_grid_profile(project_path, 20, 20)
            ellipse = set_effective_ellipse(project_path, asset.image_asset_id, 50, 40, 30, 20)

            grids = annotation_grids(100, 80, 20, 20)
            self.assertEqual(
                participating_annotation_grids(grids, ellipse),
                tuple(grid for grid in grids if grid.center_x in (30.0, 50.0, 70.0) and grid.center_y in (30.0, 50.0)),
            )
            self.assertEqual(len(participating_annotation_grids(grids, ellipse)), 6)
            self.assertEqual(
                participating_annotation_grids(
                    grids,
                    EffectiveWaferArea(asset.image_asset_id, "ellipse", ellipse.geometry, True),
                ),
                participating_annotation_grids(grids, ellipse),
            )

            window = MainWindow()
            window.resize(500, 400)
            window.show()
            app.processEvents()
            window.show_wafer_image(asset)
            window.set_grid_profile(project_path, profile)
            window.set_effective_wafer_area(ellipse)
            count_label = window.findChild(QLabel, "participatingGridCountLabel")
            self.assertEqual(count_label.text(), "Participating: 6")

            origin_x = window.findChild(QSpinBox, "gridOriginXSpinBox")
            apply_origin = window.findChild(QPushButton, "applyGridOriginButton")
            origin_x.setValue(10)
            self.assertEqual(count_label.text(), "Participating: 6")
            apply_origin.click()
            app.processEvents()
            self.assertEqual(count_label.text(), "Participating: 4")

            width_spin = window.findChild(QSpinBox, "gridWidthSpinBox")
            height_spin = window.findChild(QSpinBox, "gridHeightSpinBox")
            apply_profile = window.findChild(QPushButton, "applyGridProfileButton")
            width_spin.setValue(10)
            height_spin.setValue(20)
            self.assertEqual(count_label.text(), "Participating: 4")
            apply_profile.click()
            app.processEvents()
            self.assertEqual(count_label.text(), "Participating: 12")

            polygon = set_effective_polygon(
                project_path,
                asset.image_asset_id,
                [(40, 20), (80, 20), (80, 60), (40, 60)],
            )
            window.set_effective_wafer_area(polygon)
            self.assertEqual(count_label.text(), "Participating: 8")
            window.set_effective_wafer_area(None)
            self.assertEqual(count_label.text(), "Participating: 0")
            window.close()
            del window
            app.processEvents()


def _write_fixture(path: Path) -> None:
    image = QImage(100, 80, QImage.Format_Grayscale8)
    image.fill(80)
    if not image.save(str(path), "PNG"):
        raise AssertionError(f"Unable to write fixture: {path}")
    del image


if __name__ == "__main__":
    unittest.main()
