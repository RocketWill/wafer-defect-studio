import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication, QDockWidget, QLabel, QPushButton, QSlider, QSpinBox

from tests.test_grid_overlay import _render, _vertical_hits, _write_fixture
from wafer_defect_studio import image_asset, project
from wafer_defect_studio.grid_profile import load_grid_profiles, save_grid_profile
from wafer_defect_studio.main_window import MainWindow


class GridControlsTest(unittest.TestCase):
    def test_draft_controls_apply_one_revision_and_refresh_overlay(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "project"
            project.create_project(project_path)
            source_path = workspace / "wafer.png"
            _write_fixture(source_path)
            asset = image_asset.register_wafer_image(project_path, source_path)
            first = save_grid_profile(project_path, 64, 32)

            window = MainWindow()
            window.resize(500, 400)
            window.show()
            app.processEvents()
            window.show_wafer_image(asset)
            app.processEvents()

            dock = window.findChild(QDockWidget, "gridProfileDock")
            self.assertIsNotNone(dock)
            self.assertFalse(dock.isEnabled())
            self.assertEqual(dock.windowTitle(), "Grid Profile")
            width_spin = window.findChild(QSpinBox, "gridWidthSpinBox")
            height_spin = window.findChild(QSpinBox, "gridHeightSpinBox")
            slider = window.findChild(QSlider, "gridSizeSlider")
            apply_button = window.findChild(QPushButton, "applyGridProfileButton")
            self.assertEqual((width_spin.minimum(), width_spin.maximum()), (1, 100000))
            self.assertEqual((height_spin.minimum(), height_spin.maximum()), (1, 100000))
            self.assertEqual((slider.minimum(), slider.maximum(), slider.singleStep(), slider.pageStep()), (1, 100000, 1, 16))
            self.assertEqual([label.text() for label in window.findChildren(QLabel)], ["Cell width (px)", "Cell height (px)"])

            window.set_grid_profile(project_path, first)
            self.assertTrue(dock.isEnabled())
            self.assertTrue(dock.isVisible())
            self.assertEqual((width_spin.value(), height_spin.value(), slider.value()), (64, 32, 64))
            self.assertFalse(apply_button.isEnabled())

            view = window.centralWidget()
            view.resetTransform()
            view.scale(1.0, 1.0)
            before = project_path.joinpath("project.sqlite").read_bytes()
            before_render = _render(view)
            self.assertEqual(_vertical_hits(view, before_render, (64, 80), 20), (64,))

            slider.setValue(96)
            self.assertEqual((width_spin.value(), height_spin.value()), (96, 96))
            width_spin.setValue(80)
            height_spin.setValue(40)
            self.assertEqual((width_spin.value(), height_spin.value()), (80, 40))
            self.assertTrue(apply_button.isEnabled())
            self.assertEqual(project_path.joinpath("project.sqlite").read_bytes(), before)
            unchanged_render = _render(view)
            self.assertEqual(_vertical_hits(view, unchanged_render, (64, 80), 20), (64,))

            apply_button.click()
            app.processEvents()
            profiles = load_grid_profiles(project_path)
            self.assertEqual(len(profiles), 2)
            self.assertEqual((profiles[-1].version, profiles[-1].cell_width, profiles[-1].cell_height), (2, 80, 40))
            self.assertEqual((width_spin.value(), height_spin.value(), slider.value()), (80, 40, 80))
            self.assertFalse(apply_button.isEnabled())
            applied_render = _render(view)
            self.assertEqual(_vertical_hits(view, applied_render, (64, 80), 20), (80,))

            window.close()
            del view, window
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
