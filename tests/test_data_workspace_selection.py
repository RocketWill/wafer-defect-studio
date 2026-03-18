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
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QTableWidget

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.main_window import MainWindow


class DataWorkspaceSelectionTest(unittest.TestCase):
    @staticmethod
    def _write_grayscale_png(path: Path, width: int, height: int, value: int) -> None:
        image = QImage(width, height, QImage.Format.Format_Grayscale8)
        image.fill(value)
        if not image.save(str(path), "PNG"):
            raise AssertionError(f"failed to save {path}")

    @staticmethod
    def _wait_for(window: MainWindow, predicate) -> None:
        app = QApplication.instance()
        deadline = time.monotonic() + 5
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(10)

    def test_available_selection_previews_and_invalid_source_preserves_current_image(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = root / "project"
            project.create_project(project_path)
            available_path = root / "available.png"
            changed_path = root / "changed.png"
            missing_path = root / "missing.png"
            self._write_grayscale_png(available_path, 7, 5, 40)
            self._write_grayscale_png(changed_path, 9, 6, 80)
            self._write_grayscale_png(missing_path, 11, 8, 120)
            assets = tuple(
                image_asset.register_wafer_image(project_path, source)
                for source in (available_path, changed_path, missing_path)
            )
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
                inspector = window.findChild(QLabel, "imageInspectorEmptyState")
                self.assertIsNotNone(inventory)
                self.assertIsNotNone(inspector)
                rows = {
                    inventory.item(row, 0).text().splitlines()[0]: row
                    for row in range(inventory.rowCount())
                }

                inventory.selectRow(rows[available_path.name])
                self._wait_for(
                    window,
                    lambda: window.current_image_asset is not None
                    and window.current_image_asset.path == available_path.resolve(),
                )
                self.assertIsNotNone(window.loaded_wafer_image)
                self.assertEqual(window.loaded_wafer_image.width, 7)
                self.assertEqual(window.loaded_wafer_image.height, 5)
                self.assertIn(available_path.name, inspector.text())
                self.assertIn("7 × 5 px", inspector.text())
                self.assertIn("uint8 · PNG", inspector.text())
                self.assertIn("Available", inspector.text())

                previous_asset = window.current_image_asset
                inventory.selectRow(rows[missing_path.name])
                app.processEvents()
                self.assertIs(window.current_image_asset, previous_asset)
                self.assertIn("Missing Source", inspector.text())
                self.assertIn("Missing Source", window.statusBar().currentMessage())

                inventory.selectRow(rows[changed_path.name])
                app.processEvents()
                self.assertIs(window.current_image_asset, previous_asset)
                self.assertIn("Changed Source", inspector.text())
                self.assertIn("Changed Source", window.statusBar().currentMessage())
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
