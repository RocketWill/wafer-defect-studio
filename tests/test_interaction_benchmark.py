import os
import unittest
from array import array

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from wafer_defect_studio.interaction_benchmark import measure_interactions
from wafer_defect_studio.wafer_view import LoadedWaferImage, WaferView


class InteractionBenchmarkTest(unittest.TestCase):
    def test_measurement_exercises_four_feedback_seams_without_pixel_mutation(self):
        app = QApplication.instance() or QApplication([])
        view = WaferView()
        view.resize(120, 120)
        loaded = LoadedWaferImage(4, 4, "uint8", array("B", range(16)))
        image = QImage(4, 4, QImage.Format_Grayscale8)
        image.fill(80)
        view._set_loaded_image(loaded, image)
        view.show()
        app.processEvents()
        before = view.source_pixel_at(QPoint(1, 1))

        report = measure_interactions(view, samples=1, target_ms=100.0)

        self.assertEqual(
            [measurement.name for measurement in report.measurements],
            ["pan", "zoom", "hover/source lookup", "selection"],
        )
        self.assertTrue(all(measurement.latency_ms >= 0 for measurement in report.measurements))
        self.assertEqual(report.target_ms, 100.0)
        self.assertEqual(before, view.source_pixel_at(QPoint(1, 1)))
        self.assertIn("Interaction Feedback Benchmark", report.to_markdown())
        view.close()


if __name__ == "__main__":
    unittest.main()
