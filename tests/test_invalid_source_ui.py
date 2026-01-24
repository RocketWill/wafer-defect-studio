import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.wafer_view import LoadedWaferImage
from array import array


class InvalidSourceUiTest(unittest.TestCase):
    def test_recheck_invalidates_stale_available_request(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "project"
            project.create_project(project_path)
            source_path = workspace / "wafer.tiff"
            _write_source(source_path)
            registered = image_asset.register_wafer_image(project_path, source_path)
            reopened = image_asset.load_image_assets(project_path)[0]

            window = MainWindow()
            request = Mock(return_value=41)
            window._wafer_loader.request = request
            window.show()
            app.processEvents()
            try:
                window.load_wafer_image(reopened)
                self.assertEqual(window.statusBar().currentMessage(), "Loading")
                self.assertEqual(request.call_count, 1)
                old_token = window._latest_load_token

                source_path.unlink()
                window.load_wafer_image(reopened)
                self.assertEqual(window.statusBar().currentMessage(), "Missing Source")
                self.assertEqual(request.call_count, 1)
                missing_token = window._latest_load_token
                self.assertNotEqual(missing_token, old_token)

                source_path.write_bytes(b"changed source bytes")
                window.load_wafer_image(reopened)
                self.assertEqual(window.statusBar().currentMessage(), "Changed Source")
                self.assertEqual(request.call_count, 1)
                changed_token = window._latest_load_token
                self.assertNotEqual(changed_token, missing_token)

                stale_loaded = LoadedWaferImage(2, 2, "uint16", array("H", [1, 1, 1, 1]))
                stale_image = QImage(2, 2, QImage.Format_Grayscale16)
                window._on_load_ready(old_token, stale_loaded, stale_image)
                self.assertIsNone(window._loaded_wafer_image)
                self.assertEqual(window._latest_load_token, changed_token)
            finally:
                window.close()
                del window
                app.processEvents()


def _write_source(path: Path) -> None:
    image = QImage(5000, 4000, QImage.Format_Grayscale16)
    image.fill(0x1234)
    if not image.save(str(path), "TIFF"):
        raise AssertionError(f"Unable to write source: {path}")
    del image


if __name__ == "__main__":
    unittest.main()
