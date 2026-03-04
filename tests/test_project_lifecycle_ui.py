import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFileDialog

from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.project import open_project


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


if __name__ == "__main__":
    unittest.main()
