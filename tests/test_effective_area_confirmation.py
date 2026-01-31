import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.effective_area import (
    EffectiveWaferAreaError,
    confirmed_participating_grids,
    confirm_effective_wafer_area,
    load_effective_wafer_area,
    set_effective_ellipse,
    set_effective_polygon,
)
from wafer_defect_studio.grid_profile import save_grid_profile
from wafer_defect_studio.grid_geometry import annotation_grids
from wafer_defect_studio.main_window import MainWindow


class EffectiveAreaConfirmationTest(unittest.TestCase):
    def test_confirm_gate_persists_reopens_and_edits_clear_confirmation(self):
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
            with self.assertRaisesRegex(EffectiveWaferAreaError, "Effective wafer area is unconfirmed"):
                confirmed_participating_grids(grids, ellipse)

            window = MainWindow()
            window.resize(500, 400)
            window.show()
            app.processEvents()
            window.show_wafer_image(asset)
            window.set_grid_profile(project_path, profile)
            label = window.findChild(QLabel, "effectiveAreaConfirmationLabel")
            confirm_button = window.findChild(QPushButton, "confirmEffectiveWaferAreaButton")
            self.assertEqual(label.text(), "Unconfirmed")
            self.assertTrue(confirm_button.isEnabled())

            confirmed = confirm_effective_wafer_area(project_path, asset.image_asset_id)
            window.set_effective_wafer_area(confirmed)
            app.processEvents()
            self.assertTrue(confirmed.confirmed)
            self.assertEqual(label.text(), "Confirmed")
            self.assertFalse(confirm_button.isEnabled())
            self.assertEqual(len(confirmed_participating_grids(grids, confirmed)), 6)
            self.assertEqual(load_effective_wafer_area(project_path, asset.image_asset_id), confirmed)

            window.close()
            del window
            app.processEvents()
            reopened_window = MainWindow()
            reopened_window.resize(500, 400)
            reopened_window.show()
            app.processEvents()
            reopened_window.show_wafer_image(asset)
            reopened_window.set_grid_profile(project_path, profile)
            reopened_label = reopened_window.findChild(QLabel, "effectiveAreaConfirmationLabel")
            reopened_button = reopened_window.findChild(QPushButton, "confirmEffectiveWaferAreaButton")
            self.assertEqual(reopened_label.text(), "Confirmed")
            self.assertFalse(reopened_button.isEnabled())

            edited_ellipse = set_effective_ellipse(
                project_path, asset.image_asset_id, 48, 38, 25, 15
            )
            reopened_window.set_effective_wafer_area(edited_ellipse)
            self.assertEqual(reopened_label.text(), "Unconfirmed")
            self.assertTrue(reopened_button.isEnabled())

            edited_polygon = set_effective_polygon(
                project_path,
                asset.image_asset_id,
                [(40, 20), (80, 20), (80, 60), (40, 60)],
            )
            reopened_window.set_effective_wafer_area(edited_polygon)
            self.assertEqual(reopened_label.text(), "Unconfirmed")
            self.assertTrue(reopened_button.isEnabled())
            reopened_window.close()
            del reopened_window
            app.processEvents()


def _write_fixture(path: Path) -> None:
    image = QImage(100, 80, QImage.Format_Grayscale8)
    image.fill(80)
    if not image.save(str(path), "PNG"):
        raise AssertionError(f"Unable to write fixture: {path}")
    del image


if __name__ == "__main__":
    unittest.main()
