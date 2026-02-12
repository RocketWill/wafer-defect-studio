import os
import queue
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QLabel, QPushButton, QSlider

from wafer_defect_studio.cam_detection import CamDetectionArtifact
from wafer_defect_studio.detection_worker import DetectionProgress, DetectionRequest, DetectionTerminal
from wafer_defect_studio.main_window import MainWindow


class _FakeHandle:
    def __init__(self):
        self.queue = queue.Queue()
        self.cancelled = False
        self.alive = True

    def cancel(self, reason="user"):
        self.cancelled = True
        return True

    def is_alive(self):
        return self.alive


class DetectionUiTest(unittest.TestCase):
    def test_layers_class_map_inspection_and_truthful_worker_state(self):
        app = QApplication.instance() or QApplication([])
        maps = np.asarray(
            [
                [[0.1, 0.8], [0.2, 0.4]],
                [[0.3, 0.5], [0.9, 0.6]],
            ],
            dtype=np.float32,
        )
        artifact = CamDetectionArtifact(
            maps=maps,
            coverage=np.ones((2, 2), dtype=np.float32),
            class_names=("scratch", "particle"),
            source_width=2,
            source_height=2,
            source_transform={"coordinate_system": "source-image pixels"},
            window_settings={"window_count": 1},
            provenance={"run_id": "run-1", "profile_id": "profile-1", "evaluation_id": "eval-1"},
        )
        request = DetectionRequest(
            request_id="ui-detection",
            source=np.zeros((2, 2), dtype=np.uint8),
            staging_path="unused-stage",
            run_id="run-1",
            profile_id="profile-1",
            class_names=artifact.class_names,
            window_size=2,
            stride=2,
        )
        handle = _FakeHandle()
        started = []

        def launcher(value):
            started.append(value)
            return handle

        window = MainWindow()
        window.configure_detection(artifact, request, launcher=launcher)
        window.show()
        app.processEvents()

        self.assertIn("Approximate localization", window.findChild(QLabel, "detectionDisclaimerLabel").text())
        self.assertIn("not a segmentation mask", window.findChild(QLabel, "detectionDisclaimerLabel").text())
        self.assertTrue(window.findChild(QCheckBox, "detectionImageLayerCheckBox").isChecked())
        self.assertTrue(window.findChild(QCheckBox, "detectionGridLayerCheckBox").isChecked())
        map_layer = window.findChild(QCheckBox, "detectionMapLayerCheckBox")
        self.assertTrue(map_layer.isChecked())
        classes = window.findChild(QComboBox, "detectionClassSelector")
        legend = window.findChild(QLabel, "detectionLegendLabel")
        map_preview = window.findChild(QLabel, "detectionMapPreviewLabel")
        self.assertEqual(classes.currentText(), "scratch")
        classes.setCurrentText("particle")
        self.assertIn("particle", legend.text())
        self.assertIn("particle", map_preview.text())
        opacity = window.findChild(QSlider, "detectionOpacitySlider")
        opacity.setValue(35)
        self.assertIn("35", window.findChild(QLabel, "detectionOpacityLabel").text())

        window.findChild(QPushButton, "inspectDetectionConfidenceButton").click()
        self.assertIn("(0, 0)", window.findChild(QLabel, "detectionConfidenceLabel").text())

        window.findChild(QPushButton, "startDetectionButton").click()
        self.assertEqual(started, [request])
        cancel = window.findChild(QPushButton, "cancelDetectionButton")
        self.assertTrue(cancel.isEnabled())
        handle.queue.put(DetectionProgress("ui-detection", "detect", 1, 3, "Evaluated window"))
        QTest.qWait(100)
        self.assertIn("1/3", window.findChild(QLabel, "detectionProgressLabel").text())
        cancel.click()
        self.assertTrue(handle.cancelled)
        handle.queue.put(DetectionTerminal("ui-detection", "cancelled", "Detection cancelled; no map set was published."))
        QTest.qWait(100)
        self.assertIn("cancelled", window.findChild(QLabel, "detectionStatusLabel").text().lower())
        window.close()


if __name__ == "__main__":
    unittest.main()
