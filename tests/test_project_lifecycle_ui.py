import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QDockWidget, QFileDialog, QListWidget

from wafer_defect_studio import image_asset
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.project import create_project, open_project


class ProjectLifecycleUiTest(unittest.TestCase):
    def test_create_project_action_creates_and_activates_project(self):
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.show()
        app.processEvents()
        try:
            self.assertIsNone(window.active_project_path)
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
            self.assertEqual(create_action.text(), "Create Project…")

            with TemporaryDirectory() as temporary_directory:
                project_path = Path(temporary_directory) / "new-project"
                project_path.mkdir()

                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(project_path),
                ):
                    create_action.trigger()

                self.assertEqual(window.active_project_path, project_path.resolve())
                self.assertIn(project_path.name, window.windowTitle())
                self.assertEqual(open_project(project_path).path, project_path.resolve())

                title_after_create = window.windowTitle()
                with patch.object(QFileDialog, "getExistingDirectory", return_value=""):
                    create_action.trigger()
                self.assertEqual(window.active_project_path, project_path.resolve())
                self.assertEqual(window.windowTitle(), title_after_create)
        finally:
            window.close()

    def test_open_project_action_validates_and_activates_project(self):
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
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
            open_action = next(
                action
                for action in file_menu.actions()
                if action.objectName() == "openProjectAction"
            )

            with TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                current_path = root / "current-project"
                current_path.mkdir()
                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(current_path),
                ):
                    create_action.trigger()

                target_path = root / "target-project"
                target_path.mkdir()
                create_project(target_path)

                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(target_path),
                ):
                    open_action.trigger()

                self.assertEqual(window.active_project_path, target_path.resolve())
                self.assertIn(target_path.name, window.windowTitle())
                title_after_open = window.windowTitle()

                with patch.object(QFileDialog, "getExistingDirectory", return_value=""):
                    open_action.trigger()
                self.assertEqual(window.active_project_path, target_path.resolve())
                self.assertEqual(window.windowTitle(), title_after_open)

                invalid_path = root / "invalid-project"
                invalid_path.mkdir()
                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(invalid_path),
                ):
                    open_action.trigger()
                self.assertEqual(window.active_project_path, target_path.resolve())
                self.assertEqual(window.windowTitle(), title_after_open)
        finally:
            window.close()

    def test_import_wafer_image_action_registers_and_displays_native_source(self):
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
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

            with TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                project_path = root / "project"
                project_path.mkdir()
                source_path = root / "wafer.png"
                _write_grayscale_png(source_path, width=4, height=3, value=37)
                source_before = source_path.read_bytes()

                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(project_path),
                ):
                    create_action.trigger()
                self.assertTrue(import_action.isEnabled())

                with patch.object(
                    QFileDialog,
                    "getOpenFileName",
                    return_value=(str(source_path), "PNG"),
                ):
                    import_action.trigger()

                deadline = time.monotonic() + 3
                while (
                    window.statusBar().currentMessage() != "Ready"
                    and time.monotonic() < deadline
                ):
                    app.processEvents()
                    time.sleep(0.01)
                self.assertEqual(window.statusBar().currentMessage(), "Ready")
                project_hub_dock = window.findChild(QDockWidget, "projectHubDock")
                self.assertIsNotNone(project_hub_dock)
                self.assertFalse(project_hub_dock.isVisible())

                asset = window.current_image_asset
                loaded = window.loaded_wafer_image
                self.assertIsNotNone(asset)
                self.assertIsNotNone(loaded)
                self.assertEqual(asset.path, source_path.resolve())
                self.assertEqual((asset.width, asset.height, asset.dtype), (4, 3, "uint8"))
                self.assertEqual((loaded.width, loaded.height, loaded.dtype), (4, 3, "uint8"))
                self.assertEqual(loaded.pixels.typecode, "B")
                self.assertEqual(list(loaded.pixels), [37] * 12)
                self.assertEqual(source_path.read_bytes(), source_before)
                self.assertEqual(image_asset.load_image_assets(project_path)[0].asset, asset)

                previous_asset = asset
                previous_loaded = loaded
                with patch.object(QFileDialog, "getOpenFileName", return_value=("", "")):
                    import_action.trigger()
                self.assertEqual(window.current_image_asset, previous_asset)
                self.assertEqual(window.loaded_wafer_image, previous_loaded)

                color_path = root / "color.png"
                _write_color_png(color_path)
                with patch.object(
                    QFileDialog,
                    "getOpenFileName",
                    return_value=(str(color_path), "PNG"),
                ):
                    import_action.trigger()
                self.assertEqual(window.current_image_asset, previous_asset)
                self.assertEqual(window.loaded_wafer_image, previous_loaded)
                self.assertEqual(source_path.read_bytes(), source_before)
        finally:
            window.close()

    def test_project_hub_persists_recent_projects_and_opens_valid_entries(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
            settings.clear()
            settings.sync()

            project_path = root / "project"
            project_path.mkdir()
            target_path = root / "target-project"
            target_path.mkdir()
            create_project(target_path)

            window = MainWindow(settings=settings)
            window.show()
            app.processEvents()
            try:
                central_widget = window.centralWidget()
                hub = window.findChild(QListWidget, "projectHubList")
                self.assertIsNotNone(hub)
                self.assertTrue(hub.isVisible())

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
                open_action = next(
                    action
                    for action in file_menu.actions()
                    if action.objectName() == "openProjectAction"
                )

                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(project_path),
                ):
                    create_action.trigger()
                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(target_path),
                ):
                    open_action.trigger()
                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(target_path),
                ):
                    open_action.trigger()

                self.assertIs(window.centralWidget(), central_widget)
                self.assertEqual(hub.count(), 2)
                self.assertIn(str(target_path.resolve()), hub.item(0).text())

                window.close()
                app.processEvents()
                fresh_window = MainWindow(settings=settings)
                fresh_window.show()
                app.processEvents()
                try:
                    fresh_hub = fresh_window.findChild(QListWidget, "projectHubList")
                    self.assertIsNotNone(fresh_hub)
                    self.assertTrue(fresh_hub.isVisible())
                    self.assertEqual(fresh_hub.count(), 2)
                    self.assertIn(str(target_path.resolve()), fresh_hub.item(0).text())

                    valid_item = fresh_hub.item(0)
                    fresh_hub.itemActivated.emit(valid_item)
                    self.assertEqual(fresh_window.active_project_path, target_path.resolve())
                    self.assertIn(target_path.name, fresh_window.windowTitle())

                    active_path = fresh_window.active_project_path
                    active_title = fresh_window.windowTitle()
                    invalid_path = root / "invalid-project"
                    invalid_path.mkdir()
                    fresh_hub.insertItem(0, str(invalid_path))
                    fresh_hub.itemActivated.emit(fresh_hub.item(0))
                    self.assertEqual(fresh_window.active_project_path, active_path)
                    self.assertEqual(fresh_window.windowTitle(), active_title)
                finally:
                    fresh_window.close()
            finally:
                if not window.isHidden():
                    window.close()

    def test_project_hub_displays_source_health_without_mutating_sources(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            available_path = root / "available-project"
            available_path.mkdir()
            create_project(available_path)
            available_source = root / "available.png"
            _write_grayscale_png(available_source, width=2, height=2, value=37)
            available_before = available_source.read_bytes()
            available_asset = image_asset.register_wafer_image(available_path, available_source)

            missing_path = root / "missing-project"
            missing_path.mkdir()
            create_project(missing_path)
            missing_source = root / "missing.png"
            _write_grayscale_png(missing_source, width=2, height=2, value=41)
            image_asset.register_wafer_image(missing_path, missing_source)
            missing_source.unlink()

            changed_path = root / "changed-project"
            changed_path.mkdir()
            create_project(changed_path)
            changed_source = root / "changed.png"
            _write_grayscale_png(changed_source, width=2, height=2, value=43)
            image_asset.register_wafer_image(changed_path, changed_source)
            changed_source.write_bytes(b"changed source bytes")

            empty_path = root / "empty-project"
            empty_path.mkdir()
            create_project(empty_path)

            invalid_path = root / "invalid-project"
            invalid_path.mkdir()
            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
            settings.setValue(
                "recentProjects",
                [
                    str(invalid_path),
                    str(empty_path),
                    str(changed_path),
                    str(missing_path),
                    str(available_path),
                ],
            )
            settings.sync()

            window = MainWindow(settings=settings)
            window.show()
            app.processEvents()
            try:
                hub = window.findChild(QListWidget, "projectHubList")
                self.assertIsNotNone(hub)
                items = {
                    item.data(Qt.ItemDataRole.UserRole): item.text()
                    for item in (hub.item(index) for index in range(hub.count()))
                }
                expected_statuses = {
                    invalid_path: "Project Unavailable",
                    empty_path: "No Wafer Images",
                    changed_path: "Changed Source",
                    missing_path: "Missing Source",
                    available_path: "Available",
                }
                for project_path, status in expected_statuses.items():
                    display = items[str(project_path.resolve())]
                    self.assertIn(str(project_path.resolve()), display)
                    self.assertIn(status, display)

                self.assertEqual(available_source.read_bytes(), available_before)
                self.assertEqual(
                    image_asset.load_image_assets(available_path)[0].asset.fingerprint,
                    available_asset.fingerprint,
                )

                self.assertIsNone(window.active_project_path)
                hub.itemActivated.emit(hub.item(0))
                self.assertIsNone(window.active_project_path)
                self.assertIn("Project Unavailable", hub.item(0).text())
            finally:
                window.close()


def _write_grayscale_png(path: Path, *, width: int, height: int, value: int) -> None:
    image = QImage(width, height, QImage.Format_Grayscale8)
    image.fill(value)
    if not image.save(str(path), "PNG"):
        raise AssertionError(f"Unable to write grayscale source: {path}")
    del image


def _write_color_png(path: Path) -> None:
    image = QImage(4, 3, QImage.Format_RGB32)
    image.fill(0xFF112233)
    if not image.save(str(path), "PNG"):
        raise AssertionError(f"Unable to write color source: {path}")
    del image


if __name__ == "__main__":
    unittest.main()
