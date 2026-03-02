import os
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication, QMainWindow

from wafer_defect_studio import application


class ApplicationSmokeTest(unittest.TestCase):
    def test_main_shows_named_window_and_closes(self):
        shown_windows = []

        def fake_exec(app):
            windows = [widget for widget in app.topLevelWidgets() if widget.isVisible()]
            shown_windows.extend(windows)
            for widget in windows:
                self.assertIsInstance(widget, QMainWindow)
                self.assertEqual(widget.windowTitle(), "Wafer Defect Studio")
                widget.close()
            return 0

        with patch.object(QApplication, "exec", new=fake_exec):
            result = application.main([])

        self.assertEqual(result, 0)
        self.assertEqual(len(shown_windows), 1)

    def test_main_reuses_existing_application_instance(self):
        existing = QApplication.instance() or QApplication([])
        seen = []

        def fake_exec(app):
            seen.append(app)
            for widget in tuple(app.topLevelWidgets()):
                widget.close()
            return 0

        with patch.object(QApplication, "exec", new=fake_exec):
            result = application.main([])

        self.assertEqual(result, 0)
        self.assertEqual(seen, [existing])


if __name__ == "__main__":
    unittest.main()
