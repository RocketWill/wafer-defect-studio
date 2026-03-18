import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QTableWidget

from wafer_defect_studio import project
from wafer_defect_studio.main_window import MainWindow


class DataWorkspaceImportTest(unittest.TestCase):
    @staticmethod
    def _write_grayscale_jpeg(path: Path) -> None:
        image = QImage(6, 4, QImage.Format.Format_Grayscale8)
        image.fill(160)
        if not image.save(str(path), "JPEG"):
            raise AssertionError(f"failed to save {path}")

    @staticmethod
    def _wait_for(window: MainWindow, predicate) -> None:
        app = QApplication.instance()
        deadline = time.monotonic() + 5
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(10)

    def test_import_refreshes_inventory_and_preserves_lossy_disclosure(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = root / "project"
            project.create_project(project_path)
            source_path = root / "wafer-lossy.jpg"
            self._write_grayscale_jpeg(source_path)
            source_before = source_path.read_bytes()
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

                with patch.object(
                    QFileDialog,
                    "getOpenFileName",
                    return_value=(str(source_path), "JPEG (*.jpg *.jpeg)"),
                ):
                    window.import_wafer_image_action.trigger()
                inventory = window.findChild(QTableWidget, "imageInventoryTable")
                self.assertIsNotNone(inventory)
                self._wait_for(
                    window,
                    lambda: window.current_image_asset is not None
                    and window.loaded_wafer_image is not None
                    and window.statusBar().currentMessage().startswith("Ready"),
                )

                self.assertIsNotNone(window.current_image_asset)
                self.assertEqual(window.loaded_wafer_image.width, 6)
                self.assertEqual(window.loaded_wafer_image.height, 4)
                self.assertEqual(inventory.rowCount(), 1)
                row = [inventory.item(0, column).text() for column in range(inventory.columnCount())]
                self.assertIn(source_path.name, row[0])
                self.assertIn("6 × 4 px", row[1])
                self.assertIn("uint8", row[2])
                self.assertIn("JPEG", row[2])
                self.assertIn("Lossy source", row[2])
                self.assertEqual(row[3], "Available")
                self.assertEqual(source_path.read_bytes(), source_before)
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
