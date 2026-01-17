import hashlib
from array import array
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QGraphicsPixmapItem, QGraphicsView

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.main_window import MainWindow


class WaferViewTest(unittest.TestCase):
    def test_show_wafer_image_preserves_native_pixels_and_fits_display(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "project"
            project.create_project(project_path)
            source_path = workspace / "wafer.tiff"
            _write_fixture(source_path)
            source_fingerprint = _sha256(source_path)
            asset = image_asset.register_wafer_image(project_path, source_path)

            window = MainWindow()
            window.resize(800, 600)
            window.show()
            app.processEvents()
            loaded = window.show_wafer_image(asset)
            app.processEvents()

            self.assertEqual((loaded.width, loaded.height), (5000, 4000))
            self.assertEqual(loaded.dtype, "uint16")
            self.assertEqual(len(loaded.pixels), 5000 * 4000)
            self.assertEqual(loaded.pixels[0], 0x1234)
            self.assertEqual(loaded.pixels[1], 0xABCD)
            self.assertEqual(loaded.pixels[-1], 0xFEDC)
            self.assertEqual(_sha256(source_path), source_fingerprint)

            central_widget = window.centralWidget()
            self.assertIsInstance(central_widget, QGraphicsView)
            scene_items = central_widget.scene().items()
            self.assertEqual(len(scene_items), 1)
            self.assertIsInstance(scene_items[0], QGraphicsPixmapItem)
            pixmap = scene_items[0].pixmap()
            self.assertLess(pixmap.width(), asset.width)
            self.assertLess(pixmap.height(), asset.height)
            self.assertAlmostEqual(
                pixmap.width() / pixmap.height(),
                asset.width / asset.height,
                places=2,
            )

            window.close()
            del pixmap, central_widget, window, loaded
            app.processEvents()


def _write_fixture(path: Path) -> None:
    image = QImage(5000, 4000, QImage.Format_Grayscale16)
    image.fill(0)
    bits = image.bits()
    stride = image.bytesPerLine()
    for x, y, value in ((0, 0, 0x1234), (1, 0, 0xABCD), (4999, 3999, 0xFEDC)):
        offset = y * stride + x * 2
        bits[offset : offset + 2] = array("H", [value]).tobytes()
    if not image.save(str(path), "TIFF"):
        raise AssertionError(f"Unable to write fixture: {path}")
    del bits, image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
