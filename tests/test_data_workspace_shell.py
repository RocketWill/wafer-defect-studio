import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QTableWidget, QWidget

from wafer_defect_studio.main_window import MainWindow


class DataWorkspaceShellTest(unittest.TestCase):
    def test_data_workspace_exposes_truthful_empty_shell_without_project(self):
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.show()
        app.processEvents()
        try:
            self.assertEqual(window.current_workspace, "Data")
            workspace = window.findChild(QWidget, "dataWorkspace")
            self.assertIsNotNone(workspace)
            self.assertTrue(workspace.isVisible())
            self.assertEqual(workspace.accessibleName(), "Data workspace")

            inventory = window.findChild(QTableWidget, "imageInventoryTable")
            self.assertIsNotNone(inventory)
            self.assertEqual(inventory.accessibleName(), "Wafer Image inventory")
            self.assertEqual(inventory.rowCount(), 0)
            self.assertTrue(inventory.isEnabled())
            self.assertEqual(
                inventory.horizontalHeader().accessibleName(),
                "Wafer Image inventory columns",
            )
            self.assertEqual(
                inventory.verticalHeader().accessibleName(),
                "Wafer Image inventory rows",
            )

            groups = window.findChild(QLabel, "dataGroupsEmptyState")
            self.assertIsNotNone(groups)
            self.assertIn("Data Groups", groups.text())

            inspector = window.findChild(QLabel, "imageInspectorEmptyState")
            self.assertIsNotNone(inspector)
            self.assertIn("No active project", inspector.text())
        finally:
            window.close()
            window.deleteLater()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
