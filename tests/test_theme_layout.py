import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QPushButton

from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.ui_theme import SemanticColors, ThemeMode, apply_theme


class ThemeLayoutTest(unittest.TestCase):
    def test_system_light_tokens_focus_and_1920_layout_smoke(self):
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.resize(1920, 1080)
        window.show()
        app.processEvents()
        colors = apply_theme(window, ThemeMode.LIGHT)
        window.configure_jobs(())
        app.processEvents()
        self.assertIsInstance(colors, SemanticColors)
        self.assertIn("border", window.styleSheet())
        self.assertNotEqual(colors.text, colors.window_background)
        self.assertGreaterEqual(colors.focus_contrast, 3.0)

        for widget in window.findChildren(QPushButton):
            if widget.isVisibleTo(window):
                mapped = QRect(widget.mapTo(window, QPoint(0, 0)), widget.size())
                self.assertTrue(window.rect().contains(mapped))
        self.assertEqual(window.palette().color(QPalette.ColorRole.Window).name(), colors.window_background)
        window.close()


if __name__ == "__main__":
    unittest.main()
