import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QLabel,
    QPushButton,
    QTableWidget,
)

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.training_scope import (
    DataGroup,
    assign_image_to_data_group,
    load_data_groups,
    save_data_groups,
)


class DataWorkspaceGroupsTest(unittest.TestCase):
    @staticmethod
    def _write_grayscale_png(path: Path, value: int) -> None:
        image = QImage(4, 3, QImage.Format.Format_Grayscale8)
        image.fill(value)
        if not image.save(str(path), "PNG"):
            raise AssertionError(f"failed to save {path}")

    def _open_project(self, root: Path, project_path: Path) -> MainWindow:
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

    def test_data_group_inventory_empty_then_seeded_counts(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = root / "empty-project"
            project.create_project(project_path)
            window = self._open_project(root, project_path)
            try:
                inventory = window.findChild(QTableWidget, "dataGroupInventoryTable")
                self.assertIsNotNone(inventory)
                self.assertEqual(inventory.accessibleName(), "Data Group inventory")
                self.assertEqual(inventory.rowCount(), 0)
                self.assertEqual(
                    inventory.horizontalHeader().accessibleName(),
                    "Data Group inventory columns",
                )
                empty = window.findChild(QLabel, "dataGroupsEmptyState")
                self.assertIsNotNone(empty)
                self.assertTrue(empty.isVisible())
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()

            project_path = root / "seeded-project"
            project.create_project(project_path)
            source_paths = []
            for index, value in enumerate((40, 80)):
                source = root / f"wafer-{index}.png"
                self._write_grayscale_png(source, value)
                source_paths.append(source)
            assets = tuple(
                image_asset.register_wafer_image(project_path, source)
                for source in source_paths
            )
            save_data_groups(
                project_path,
                (DataGroup("line-b", "Line B", 1), DataGroup("line-a", "Line A", 0)),
            )
            for asset in assets:
                assign_image_to_data_group(project_path, asset.image_asset_id, "line-a")

            window = self._open_project(root, project_path)
            try:
                inventory = window.findChild(QTableWidget, "dataGroupInventoryTable")
                self.assertIsNotNone(inventory)
                self.assertEqual(inventory.rowCount(), 2)
                self.assertEqual(
                    [inventory.item(row, 0).text() for row in range(inventory.rowCount())],
                    ["line-a\nLine A", "line-b\nLine B"],
                )
                self.assertEqual(
                    [inventory.item(row, 1).text() for row in range(inventory.rowCount())],
                    ["2", "0"],
                )
                empty = window.findChild(QLabel, "dataGroupsEmptyState")
                self.assertIsNotNone(empty)
                self.assertFalse(empty.isVisible())

                create_button = window.findChild(QPushButton, "createDataGroupButton")
                self.assertIsNotNone(create_button)
                self.assertEqual(create_button.accessibleName(), "Create Data Group")
                self.assertTrue(create_button.isEnabled())

                with patch.object(
                    QInputDialog,
                    "getText",
                    side_effect=[("line-c", True), ("Line C", True)],
                ):
                    create_button.click()
                self.assertEqual(
                    load_data_groups(project_path),
                    (
                        DataGroup("line-a", "Line A", 0),
                        DataGroup("line-b", "Line B", 1),
                        DataGroup("line-c", "Line C", 2),
                    ),
                )
                self.assertEqual(inventory.rowCount(), 3)
                self.assertEqual(inventory.item(2, 0).text(), "line-c\nLine C")
                self.assertEqual(inventory.item(2, 1).text(), "0")
                self.assertFalse(empty.isVisible())
                self.assertIn("created", window.statusBar().currentMessage())

                groups_before_duplicate = load_data_groups(project_path)
                with patch.object(
                    QInputDialog,
                    "getText",
                    side_effect=[("line-a", True), ("Another name", True)],
                ):
                    create_button.click()
                self.assertEqual(load_data_groups(project_path), groups_before_duplicate)
                self.assertIn("already exists", window.statusBar().currentMessage())

                with patch.object(
                    QInputDialog,
                    "getText",
                    return_value=("", False),
                ):
                    create_button.click()
                self.assertIn("cancelled", window.statusBar().currentMessage())

                with patch.object(
                    QInputDialog,
                    "getText",
                    return_value=("   ", True),
                ):
                    create_button.click()
                self.assertIn("ID", window.statusBar().currentMessage())
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
