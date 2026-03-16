import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QFileDialog

from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.project import create_project


class WorkspaceProjectContextTest(unittest.TestCase):
    def test_file_actions_activate_project_context_and_gate_workspaces(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as settings_directory:
            settings = QSettings(
                str(Path(settings_directory) / "settings.ini"),
                QSettings.Format.IniFormat,
            )
            window = MainWindow(settings=settings)
            window.show()
            app.processEvents()
            try:
                self.assertIsNone(window.active_project_path)
                self.assertTrue(window.workspace_actions["Data"].isEnabled())
                self.assertTrue(
                    all(
                        not window.workspace_actions[name].isEnabled()
                        for name in window.WORKSPACES[1:]
                    )
                )
                self.assertIn("No active project", window.statusBar().currentMessage())

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
                    created_path = root / "created-project"
                    created_path.mkdir()
                    with patch.object(
                        QFileDialog,
                        "getExistingDirectory",
                        return_value=str(created_path),
                    ):
                        create_action.trigger()

                    self.assertEqual(window.active_project_path, created_path.resolve())
                    self.assertEqual(
                        window.windowTitle(),
                        f"Wafer Defect Studio — {created_path.name}",
                    )
                    self.assertIn(
                        f"Project created: {created_path.resolve()}",
                        window.statusBar().currentMessage(),
                    )
                    self.assertTrue(
                        all(
                            window.workspace_actions[name].isEnabled()
                            for name in window.WORKSPACES
                        )
                    )

                    opened_path = root / "opened-project"
                    opened_path.mkdir()
                    create_project(opened_path)
                    with patch.object(
                        QFileDialog,
                        "getExistingDirectory",
                        return_value=str(opened_path),
                    ):
                        open_action.trigger()

                    self.assertEqual(window.active_project_path, opened_path.resolve())
                    self.assertEqual(
                        window.windowTitle(),
                        f"Wafer Defect Studio — {opened_path.name}",
                    )
                    self.assertIn(
                        f"Project opened: {opened_path.resolve()}",
                        window.statusBar().currentMessage(),
                    )
                    self.assertTrue(
                        all(
                            window.workspace_actions[name].isEnabled()
                            for name in window.WORKSPACES
                        )
                    )
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
