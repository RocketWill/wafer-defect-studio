from array import array
import os
from pathlib import Path
import re
import sqlite3
from tempfile import TemporaryDirectory
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtGui import QImage, qRgb
from PySide6.QtWidgets import QApplication

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.main_window import MainWindow


class GrayscaleFormatsTest(unittest.TestCase):
    def test_register_and_display_grayscale_formats_without_source_mutation(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "project"
            project.create_project(project_path)

            fixtures = (
                ("gray16.tiff", "TIFF", QImage.Format_Grayscale16, "uint16", [0x1234, 0xABCD, 0xFEDC, 0x0102, 0x3456, 0x789A]),
                ("gray16.png", "PNG", QImage.Format_Grayscale16, "uint16", [0x2345, 0xBCDE, 0xEDCB, 0x0203, 0x4567, 0x89AB]),
                ("gray8.png", "PNG", QImage.Format_Grayscale8, "uint8", [3, 17, 255, 128, 64, 9]),
                ("gray8.bmp", "BMP", QImage.Format_Indexed8, "uint8", [2, 19, 100, 200, 250, 33]),
            )
            for filename, image_format, qimage_format, dtype, values in fixtures:
                source_path = workspace / filename
                _write_grayscale(source_path, image_format, qimage_format, values)
                source_before = source_path.read_bytes()

                asset = image_asset.register_wafer_image(project_path, source_path)

                self.assertEqual(asset.path, source_path.resolve())
                self.assertEqual((asset.width, asset.height), (3, 2))
                self.assertEqual((asset.format, asset.dtype), (image_format, dtype))
                self.assertEqual(source_path.read_bytes(), source_before)
                connection = sqlite3.connect(project_path / "project.sqlite")
                try:
                    row = connection.execute(
                        "SELECT path, width, height, dtype, format FROM image_assets WHERE image_asset_id = ?",
                        (asset.image_asset_id,),
                    ).fetchone()
                finally:
                    connection.close()
                self.assertEqual(row, (str(asset.path), 3, 2, dtype, image_format))

                window = MainWindow()
                window.resize(320, 240)
                window.show()
                app.processEvents()
                try:
                    loaded = window.show_wafer_image(asset)
                    self.assertEqual((loaded.width, loaded.height, loaded.dtype), (3, 2, dtype))
                    self.assertEqual(loaded.pixels.typecode, "H" if dtype == "uint16" else "B")
                    self.assertEqual(list(loaded.pixels), values)
                finally:
                    window.close()
                    del window
                    app.processEvents()

            reject_project = workspace / "reject-project"
            project.create_project(reject_project)
            for filename, image_format in (("color.png", "PNG"), ("color.bmp", "BMP")):
                source_path = workspace / filename
                _write_color(source_path, image_format)
                with self.assertRaisesRegex(
                    image_asset.ImageAssetError,
                    re.escape("Color images are not supported; provide a grayscale TIFF, PNG, or BMP."),
                ):
                    image_asset.register_wafer_image(reject_project, source_path)

            connection = sqlite3.connect(reject_project / "project.sqlite")
            try:
                row_count = connection.execute("SELECT COUNT(*) FROM image_assets").fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(row_count, 0)


def _write_grayscale(path: Path, image_format: str, qimage_format, values: list[int]) -> None:
    image = QImage(3, 2, qimage_format)
    if qimage_format == QImage.Format_Indexed8:
        image.setColorCount(256)
        for index in range(256):
            image.setColor(index, qRgb(index, index, index))
    bits = image.bits()
    stride = image.bytesPerLine()
    if qimage_format == QImage.Format_Grayscale16:
        row_bytes = array("H", values[:3]).tobytes()
        for y in range(2):
            bits[y * stride : y * stride + len(row_bytes)] = array("H", values[y * 3 : y * 3 + 3]).tobytes()
    else:
        for y in range(2):
            bits[y * stride : y * stride + 3] = bytes(values[y * 3 : y * 3 + 3])
    if not image.save(str(path), image_format):
        raise AssertionError(f"Unable to write grayscale fixture: {path}")
    del bits, image


def _write_color(path: Path, image_format: str) -> None:
    image = QImage(3, 2, QImage.Format_RGB32)
    image.fill(qRgb(42, 42, 42))
    if not image.save(str(path), image_format):
        raise AssertionError(f"Unable to write color fixture: {path}")
    del image


if __name__ == "__main__":
    unittest.main()
