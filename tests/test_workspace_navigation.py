import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from wafer_defect_studio.main_window import MainWindow


class WorkspaceNavigationTest(unittest.TestCase):
    def test_workspace_order_selection_and_change_signal(self):
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.show()
        app.processEvents()
        try:
            actions = window.workspace_toolbar.actions()
            self.assertEqual(
                [action.text() for action in actions],
                ["Data", "Annotate", "Dataset", "Train", "Evaluate", "Detect", "Review"],
            )
            self.assertTrue(window.workspace_toolbar.isVisible())
            self.assertEqual(window.current_workspace, "Data")
            self.assertTrue(actions[0].isChecked())

            changed = []
            window.workspaceChanged.connect(changed.append)
            train_action = next(action for action in actions if action.text() == "Train")
            train_action.trigger()

            app.processEvents()
            self.assertEqual(window.current_workspace, "Train")
            self.assertTrue(train_action.isChecked())
            self.assertEqual(changed, ["Train"])
            self.assertIn("Train", window.statusBar().currentMessage())
        finally:
            window.close()
            window.deleteLater()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
