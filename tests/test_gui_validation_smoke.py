import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog

from wafer_defect_studio.main_window import MainWindow


class GuiValidationSmokeTest(unittest.TestCase):
    def test_file_actions_create_import_and_display_wafer_image(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
            settings.clear()
            settings.sync()
            project_path = root / "gui-project"
            project_path.mkdir()
            source_path = root / "wafer.png"
            _write_grayscale_png(source_path, width=5, height=4, value=80)
            source_before = source_path.read_bytes()

            window = MainWindow(settings=settings)
            window.show()
            app.processEvents()
            try:
                file_menu = next(
                    action.menu()
                    for action in window.menuBar().actions()
                    if action.text() == "File" and action.menu() is not None
                )
                create_action = next(
                    action
                    for action in file_menu.actions()
                    if action.objectName() == "createProjectAction"
                )
                import_action = next(
                    action
                    for action in file_menu.actions()
                    if action.objectName() == "importWaferImageAction"
                )
                self.assertFalse(import_action.isEnabled())

                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(project_path),
                ):
                    create_action.trigger()
                self.assertEqual(window.active_project_path, project_path.resolve())
                self.assertTrue(import_action.isEnabled())

                with patch.object(
                    QFileDialog,
                    "getOpenFileName",
                    return_value=(str(source_path), "PNG"),
                ):
                    import_action.trigger()

                deadline = time.monotonic() + 5
                while (
                    window.statusBar().currentMessage() != "Ready"
                    and time.monotonic() < deadline
                ):
                    app.processEvents()
                    QTest.qWait(10)
                self.assertEqual(window.statusBar().currentMessage(), "Ready")
                self.assertIsNotNone(window.current_image_asset)
                self.assertIsNotNone(window.loaded_wafer_image)
                self.assertEqual(window.current_image_asset.path, source_path.resolve())
                loaded = window.loaded_wafer_image
                self.assertEqual((loaded.width, loaded.height, loaded.dtype), (5, 4, "uint8"))
                self.assertEqual(source_path.read_bytes(), source_before)
            finally:
                deadline = time.monotonic() + 5
                while window._load_threads and time.monotonic() < deadline:
                    app.processEvents()
                    QTest.qWait(10)
                window.close()
                window.deleteLater()
                app.processEvents()


def _write_grayscale_png(path: Path, *, width: int, height: int, value: int) -> None:
    image = QImage(width, height, QImage.Format_Grayscale8)
    image.fill(value)
    if not image.save(str(path), "PNG"):
        raise AssertionError(f"Unable to write grayscale source: {path}")
    del image


if __name__ == "__main__":
    unittest.main()
