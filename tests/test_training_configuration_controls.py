import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QSpinBox

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
