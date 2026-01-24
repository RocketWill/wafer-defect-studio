from array import array
import os
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from wafer_defect_studio.image_asset import ImageAsset, SourceHealth
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.wafer_view import LoadedWaferImage


class ResponsiveLoadingTest(unittest.TestCase):
    def test_latest_load_wins_while_first_decode_is_blocked(self):
        app = QApplication.instance() or QApplication([])
        first_started = threading.Event()
        release_first = threading.Event()
        calls = 0
        call_lock = threading.Lock()
        stale = LoadedWaferImage(2, 2, "uint16", array("H", [1, 1, 1, 1]))
        latest = LoadedWaferImage(3, 2, "uint16", array("H", [2, 2, 2, 2, 2, 2]))
        stale_image = QImage(2, 2, QImage.Format_Grayscale16)
        latest_image = QImage(3, 2, QImage.Format_Grayscale16)
        first_asset = ImageAsset("first", Path("first.tiff"), 2, 2, "uint16", "TIFF", "first")
        second_asset = ImageAsset("second", Path("second.tiff"), 3, 2, "uint16", "TIFF", "second")

        def fake_decode(path):
            nonlocal calls
            with call_lock:
                call_number = calls
                calls += 1
            if call_number == 0:
                first_started.set()
                release_first.wait(2)
                return stale, stale_image
            return latest, latest_image

        window = MainWindow()
        window.resize(800, 600)
        window.show()
        app.processEvents()
        try:
            with patch("wafer_defect_studio.wafer_loader._decode_wafer_image", side_effect=fake_decode), patch(
                "wafer_defect_studio.main_window._source_health", return_value=SourceHealth.AVAILABLE
            ):
                self.assertIsNone(window.load_wafer_image(first_asset))
                self.assertTrue(first_started.wait(1))
                self.assertEqual(window.statusBar().currentMessage(), "Loading")

                timer_hits = []
                timer = QTimer(window)
                timer.setSingleShot(True)
                timer.timeout.connect(lambda: timer_hits.append(True))
                timer.start(20)
                QTest.qWait(80)
                self.assertTrue(timer_hits)

                self.assertIsNone(window.load_wafer_image(second_asset))
                deadline = time.monotonic() + 2
                while window.statusBar().currentMessage() != "Ready" and time.monotonic() < deadline:
                    app.processEvents()
                    QTest.qWait(10)
                self.assertEqual(window.statusBar().currentMessage(), "Ready")
                self.assertEqual(window._loaded_wafer_image, latest)

                release_first.set()
                deadline = time.monotonic() + 2
                while window._load_threads and time.monotonic() < deadline:
                    app.processEvents()
                    QTest.qWait(10)
                self.assertFalse(window._load_threads)
                self.assertEqual(window._loaded_wafer_image, latest)
                self.assertEqual(window.statusBar().currentMessage(), "Ready")
        finally:
            release_first.set()
            window.close()
            del window
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
