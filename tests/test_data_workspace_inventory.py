import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QFileDialog, QTableWidget

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.main_window import MainWindow


class DataWorkspaceInventoryTest(unittest.TestCase):
    @staticmethod
    def _write_grayscale_png(path: Path, width: int, height: int, value: int) -> None:
        image = QImage(width, height, QImage.Format.Format_Grayscale8)
        image.fill(value)
        if not image.save(str(path), "PNG"):
            raise AssertionError(f"failed to save {path}")

    @staticmethod
    def _write_grayscale_jpeg(path: Path) -> None:
        image = QImage(3, 2, QImage.Format.Format_Grayscale8)
        image.fill(160)
        if not image.save(str(path), "JPEG"):
            raise AssertionError(f"failed to save {path}")

    def test_project_inventory_renders_metadata_and_source_health_without_mutation(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = root / "project"
            project.create_project(project_path)
            available_path = root / "available.png"
            changed_path = root / "changed.png"
            missing_path = root / "missing.png"
            jpeg_path = root / "lossy.jpg"
            self._write_grayscale_png(available_path, 7, 5, 40)
            self._write_grayscale_png(changed_path, 9, 6, 80)
            self._write_grayscale_png(missing_path, 11, 8, 120)
            self._write_grayscale_jpeg(jpeg_path)
            assets = tuple(
                image_asset.register_wafer_image(project_path, source)
                for source in (available_path, changed_path, missing_path, jpeg_path)
            )
            available_bytes = available_path.read_bytes()
            changed_path.write_bytes(changed_path.read_bytes() + b"changed")
            missing_path.unlink()
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
                app.processEvents()

                inventory = window.findChild(QTableWidget, "imageInventoryTable")
                self.assertIsNotNone(inventory)
                self.assertEqual(inventory.rowCount(), len(assets))
                rows = {
                    inventory.item(row, 0).text().splitlines()[0]: [
                        inventory.item(row, column).text()
                        for column in range(inventory.columnCount())
                    ]
                    for row in range(inventory.rowCount())
                }
                self.assertIn(available_path.name, rows)
                self.assertIn(changed_path.name, rows)
                self.assertIn(missing_path.name, rows)
                self.assertIn(jpeg_path.name, rows)

                available_row = rows[available_path.name]
                self.assertIn(str(available_path.resolve()), available_row[0])
                self.assertEqual(available_row[1], "7 × 5 px")
                self.assertIn("uint8", available_row[2])
                self.assertIn("PNG", available_row[2])
                self.assertEqual(available_row[3], "Available")

                changed_row = rows[changed_path.name]
                self.assertEqual(changed_row[1], "9 × 6 px")
                self.assertEqual(changed_row[3], "Changed Source")

                missing_row = rows[missing_path.name]
                self.assertEqual(missing_row[1], "11 × 8 px")
                self.assertEqual(missing_row[3], "Missing Source")

                jpeg_row = rows[jpeg_path.name]
                self.assertIn("uint8", jpeg_row[2])
                self.assertIn("JPEG", jpeg_row[2])
                self.assertIn("Lossy source", jpeg_row[2])
                self.assertEqual(available_path.read_bytes(), available_bytes)
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
