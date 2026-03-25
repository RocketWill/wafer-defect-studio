import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QFileDialog, QPushButton, QTableWidget

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.training_scope import (
    DataGroup,
    DataGroupAssignment,
    load_data_groups,
    load_image_data_group_assignments,
    save_data_groups,
)


class DataWorkspaceAssignmentTest(unittest.TestCase):
    @staticmethod
    def _write_grayscale_png(path: Path) -> None:
        image = QImage(4, 3, QImage.Format.Format_Grayscale8)
        image.fill(80)
        if not image.save(str(path), "PNG"):
            raise AssertionError(f"failed to save {path}")

    @staticmethod
    def _open_project(root: Path, project_path: Path) -> MainWindow:
        settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
        window = MainWindow(settings=settings)
        window.show()
        QApplication.processEvents()
        with patch.object(
            QFileDialog,
            "getExistingDirectory",
            return_value=str(project_path),
        ):
            window.open_project_action.trigger()
        QApplication.processEvents()
        return window

    def test_assigns_selected_image_to_selected_group_and_reopens(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = root / "project"
            project.create_project(project_path)
            source = root / "wafer.png"
            self._write_grayscale_png(source)
            asset = image_asset.register_wafer_image(project_path, source)
            save_data_groups(
                project_path,
                (DataGroup("line-a", "Line A", 0), DataGroup("line-b", "Line B", 1)),
            )

            window = self._open_project(root, project_path)
            try:
                image_inventory = window.findChild(QTableWidget, "imageInventoryTable")
                group_inventory = window.findChild(QTableWidget, "dataGroupInventoryTable")
                assign_button = window.findChild(QPushButton, "assignDataGroupButton")
                self.assertIsNotNone(image_inventory)
                self.assertIsNotNone(group_inventory)
                self.assertIsNotNone(assign_button)
                self.assertEqual(assign_button.accessibleName(), "Assign Data Group")

                image_inventory.clearSelection()
                group_inventory.clearSelection()
                before = load_image_data_group_assignments(project_path)
                assign_button.click()
                self.assertEqual(load_image_data_group_assignments(project_path), before)
                self.assertIn("Select", window.statusBar().currentMessage())

                image_inventory.selectRow(0)
                group_inventory.selectRow(1)
                assign_button.click()
                self.assertEqual(
                    load_image_data_group_assignments(project_path),
                    (DataGroupAssignment(asset.image_asset_id, "line-b"),),
                )
                self.assertEqual(image_inventory.item(0, 4).text(), "line-b\nLine B")
                self.assertEqual(group_inventory.item(0, 1).text(), "0")
                self.assertEqual(group_inventory.item(1, 1).text(), "1")
                self.assertIn("assigned", window.statusBar().currentMessage())
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()

            reopened = self._open_project(root, project_path)
            try:
                image_inventory = reopened.findChild(QTableWidget, "imageInventoryTable")
                group_inventory = reopened.findChild(QTableWidget, "dataGroupInventoryTable")
                self.assertEqual(image_inventory.item(0, 4).text(), "line-b\nLine B")
                self.assertEqual(group_inventory.item(0, 1).text(), "0")
                self.assertEqual(group_inventory.item(1, 1).text(), "1")
                self.assertEqual(
                    load_data_groups(project_path),
                    (DataGroup("line-a", "Line A", 0), DataGroup("line-b", "Line B", 1)),
                )
            finally:
                reopened.close()
                reopened.deleteLater()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
