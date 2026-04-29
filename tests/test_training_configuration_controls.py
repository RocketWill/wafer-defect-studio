import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QLabel, QPushButton, QSpinBox

from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.training_inputs import (
    TrainingInputInventory,
    TrainingInputOption,
    TrainingInputControls,
)
from wafer_defect_studio.training_configuration_controls import (
    TrainingConfigurationError,
    TrainingConfigurationControls,
)


class TrainingConfigurationControlsTest(unittest.TestCase):
    def test_model_mode_builds_legacy_or_patch_request_and_blocks_invalid_geometry(self):
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.findChild(TrainingInputControls).set_inventory(
            TrainingInputInventory(
                snapshots=(TrainingInputOption("snapshot-1", "Snapshot", True, class_count=2),),
                splits=(TrainingInputOption("split-1", "Split", True, snapshot_id="snapshot-1"),),
            )
        )
        config = window.findChild(TrainingConfigurationControls)
        mode = window.findChild(QComboBox, "trainingModelModeComboBox")
        patch_size = window.findChild(QSpinBox, "trainingPatchSizeSpinBox")
        patch_stride = window.findChild(QSpinBox, "trainingPatchStrideSpinBox")
        pooling = window.findChild(QLabel, "trainingBagPoolingLabel")
        summary = window.findChild(QLabel, "trainingRequestSummaryLabel")
        start = window.findChild(QPushButton, "startTrainingButton")

        legacy = config.build_request("legacy", "staging/legacy")
        self.assertEqual(mode.currentText(), "CAM v2")
        self.assertIsNone(legacy.config.patch_size)
        self.assertIsNone(legacy.config.patch_stride)

        mode.setCurrentText("Patch Classification v3")
        app.processEvents()
        patch_request = config.build_request("patch", "staging/patch")
        self.assertEqual((patch_size.value(), patch_stride.value()), (128, 64))
        self.assertEqual(pooling.text(), "max")
        self.assertIn("Model Patch: 128x128", summary.text())
        self.assertIn("stride: 64", summary.text())
        self.assertIn("bag pooling: max", summary.text())
        self.assertTrue(patch_size.isVisibleTo(config))
        self.assertEqual(
            (patch_request.config.patch_size, patch_request.config.patch_stride),
            (128, 64),
        )
        self.assertTrue(start.isEnabled())

        patch_stride.setValue(129)
        app.processEvents()
        self.assertFalse(start.isEnabled())
        self.assertIn("patch_stride cannot exceed patch_size", config.context_status_label.text())
        with self.assertRaisesRegex(TrainingConfigurationError, "patch_stride cannot exceed"):
            config.build_request("invalid", "staging/invalid")
        window.close()

    def test_valid_inputs_build_value_only_request_and_missing_input_is_blocked(self):
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.findChild(TrainingInputControls).set_inventory(
            TrainingInputInventory(
                snapshots=(
                    TrainingInputOption(
                        "snapshot-1", "Snapshot snapshot-1", True, class_count=2
                    ),
                ),
                splits=(
                    TrainingInputOption(
                        "split-1", "Split split-1", True, snapshot_id="snapshot-1"
                    ),
                ),
            )
        )
        config = window.findChild(TrainingConfigurationControls)
        config.findChild(QSpinBox, "trainingEpochsSpinBox").setValue(3)
        config.findChild(QSpinBox, "trainingBatchSizeSpinBox").setValue(4)
        config.findChild(QDoubleSpinBox, "trainingLearningRateSpinBox").setValue(0.002)
        config.findChild(QSpinBox, "trainingSeedSpinBox").setValue(17)

        request = config.build_request("request-1", "runs/.staging/request-1")
        self.assertEqual(request.request_id, "request-1")
        self.assertEqual(request.artifact_staging_path, "runs/.staging/request-1")
        self.assertEqual(request.config.snapshot_id, "snapshot-1")
        self.assertEqual(request.config.split_id, "split-1")
        self.assertEqual(request.config.class_count, 2)
        self.assertEqual(request.config.epochs, 3)
        self.assertEqual(request.config.batch_size, 4)
        self.assertEqual(request.config.seed, 17)
        self.assertEqual(request.config.learning_rate, 0.002)

        window.findChild(TrainingInputControls).snapshot_combo.setCurrentIndex(-1)
        with self.assertRaisesRegex(TrainingConfigurationError, "Snapshot"):
            config.build_request("request-2", "runs/.staging/request-2")
        window.close()


if __name__ == "__main__":
    unittest.main()
