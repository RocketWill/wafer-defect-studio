import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGraphicsView

from tests.test_wafer_view import _write_fixture
from wafer_defect_studio import image_asset, project
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.wafer_view import WaferView


class WaferInteractionTest(unittest.TestCase):
    def test_fit_zoom_anchor_pan_and_refit(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "project"
            project.create_project(project_path)
            source_path = workspace / "wafer.tiff"
            _write_fixture(source_path)
            asset = image_asset.register_wafer_image(project_path, source_path)

            window = MainWindow()
            window.resize(800, 600)
            window.show()
            app.processEvents()
            window.show_wafer_image(asset)
            app.processEvents()

            view = window.centralWidget()
            self.assertIsInstance(view, WaferView)
            self.assertIsInstance(view, QGraphicsView)
            fit_scale = view.transform().m11()
            self.assertLess(fit_scale, 1.0)

            anchor = view.viewport().rect().center()
            before_zoom = view.source_pixel_at(anchor)
            self.assertIsNotNone(before_zoom)
            wheel = QWheelEvent(
                QPointF(anchor),
                QPointF(view.viewport().mapToGlobal(anchor)),
                QPoint(0, 0),
                QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.ScrollUpdate,
                False,
            )
            QApplication.sendEvent(view.viewport(), wheel)
            app.processEvents()
            self.assertAlmostEqual(view.transform().m11() / fit_scale, 1.25, places=2)
            self.assertEqual(view.source_pixel_at(anchor), before_zoom)

            center = view.viewport().rect().center()
            before_pan = view.source_pixel_at(center)
            view.setFocus()
            QTest.keyPress(view, Qt.Key.Key_Space)
            self.assertEqual(view.dragMode(), QGraphicsView.DragMode.ScrollHandDrag)
            QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center)
            QTest.mouseMove(view.viewport(), center + QPoint(40, 20))
            QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center + QPoint(40, 20))
            QTest.keyRelease(view, Qt.Key.Key_Space)
            self.assertEqual(view.dragMode(), QGraphicsView.DragMode.NoDrag)
            self.assertNotEqual(view.source_pixel_at(center), before_pan)

            QTest.keyPress(view, Qt.Key.Key_Space)
            QTest.keyPress(view, Qt.Key.Key_Escape)
            self.assertEqual(view.dragMode(), QGraphicsView.DragMode.NoDrag)
            QTest.keyRelease(view, Qt.Key.Key_Space)

            QTest.keyClick(view, Qt.Key.Key_F)
            app.processEvents()
            self.assertAlmostEqual(view.transform().m11(), fit_scale, places=2)

            window.close()
            del view, window
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
