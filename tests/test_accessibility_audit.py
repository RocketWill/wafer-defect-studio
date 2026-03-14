import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QGraphicsView, QLabel, QListWidget, QPushButton

from wafer_defect_studio.accessibility_audit import audit_primary_actions, ensure_accessible_labels
from wafer_defect_studio.main_window import MainWindow


class AccessibilityAuditTest(unittest.TestCase):
    def test_main_window_primary_widgets_are_labeled_and_focusable(self):
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.show()
        app.processEvents()
        ensure_accessible_labels(window)
        self.assertEqual(audit_primary_actions(window), ())
        self.assertTrue(window.findChild(QLabel, "jobsStatusLabel").text())

        project_hub = window.findChild(QListWidget, "projectHubList")
        canvas = window.findChild(QGraphicsView, "waferCanvas")
        self.assertIsNotNone(project_hub)
        self.assertIsNotNone(canvas)
        self.assertTrue(project_hub.accessibleName())
        self.assertNotEqual(project_hub.focusPolicy(), Qt.FocusPolicy.NoFocus)
        self.assertTrue(canvas.accessibleName())
        self.assertNotEqual(canvas.focusPolicy(), Qt.FocusPolicy.NoFocus)

        probe = QPushButton("", window)
        probe.setObjectName("probeAction")
        probe.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        probe.show()
        ensure_accessible_labels(window)
        self.assertEqual(probe.accessibleName(), "probeAction")
        self.assertNotEqual(probe.focusPolicy(), Qt.FocusPolicy.NoFocus)
        self.assertEqual(audit_primary_actions(window), ())
        window.close()


if __name__ == "__main__":
    unittest.main()
