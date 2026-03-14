import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.ui_theme import TypographyScale, typography_scale


class UiTypographyTest(unittest.TestCase):
    def test_system_scale_is_immutable_and_main_window_uses_body_baseline(self):
        app = QApplication.instance() or QApplication([])
        scale = typography_scale()

        self.assertIsInstance(scale, TypographyScale)
        self.assertEqual(scale.body_pt, 12.0)
        self.assertEqual(scale.caption_pt, 9.6)
        self.assertEqual(scale.section_pt, 15.0)
        self.assertEqual(scale.title_pt, 18.75)
        self.assertLess(scale.caption_pt, scale.body_pt)
        self.assertLess(scale.body_pt, scale.section_pt)
        self.assertLess(scale.section_pt, scale.title_pt)
        with self.assertRaises(AttributeError):
            scale.body_pt = 13.0

        window = MainWindow()
        try:
            self.assertEqual(window.font().family(), app.font().family())
            self.assertAlmostEqual(window.font().pointSizeF(), scale.body_pt)
        finally:
            window.close()
            window.deleteLater()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
