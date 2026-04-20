import sys
import unittest
from pathlib import Path

from PySide6.QtWidgets import QComboBox, QSpinBox


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "demo"))

from run_phase2_demo import configure_training_epochs, configure_training_weights_policy


class _FakeCombo:
    def __init__(self):
        self.selected = None

    def setCurrentText(self, value):
        self.selected = value


class _FakeSpinBox:
    def __init__(self):
        self.value = None

    def setValue(self, value):
        self.value = value


class _FakeWindow:
    def __init__(self):
        self.combo = _FakeCombo()
        self.spinbox = _FakeSpinBox()
        self.find_calls = []

    def findChild(self, widget_type, object_name):
        self.find_calls.append((widget_type, object_name))
        if widget_type is QComboBox:
            return self.combo
        if widget_type is QSpinBox:
            return self.spinbox
        raise AssertionError(f"unexpected widget type: {widget_type}")


class DemoTrainingPolicyTest(unittest.TestCase):
    def test_realistic_uses_imagenet_through_training_policy_combo(self):
        window = _FakeWindow()

        selected = configure_training_weights_policy(window, "realistic")

        self.assertEqual(selected, "imagenet")
        self.assertEqual(window.combo.selected, "imagenet")
        self.assertEqual(window.find_calls, [(QComboBox, "trainingWeightsPolicyComboBox")])

    def test_synthetic_keeps_training_from_scratch_policy(self):
        window = _FakeWindow()

        selected = configure_training_weights_policy(window, "synthetic")

        self.assertEqual(selected, "none")
        self.assertEqual(window.combo.selected, "none")
        self.assertEqual(window.find_calls, [(QComboBox, "trainingWeightsPolicyComboBox")])

    def test_realistic_uses_twenty_training_epochs_through_exact_spinbox(self):
        window = _FakeWindow()

        selected = configure_training_epochs(window, "realistic")

        self.assertEqual(selected, 20)
        self.assertEqual(window.spinbox.value, 20)
        self.assertEqual(window.find_calls, [(QSpinBox, "trainingEpochsSpinBox")])

    def test_synthetic_keeps_three_training_epochs(self):
        window = _FakeWindow()

        selected = configure_training_epochs(window, "synthetic")

        self.assertEqual(selected, 3)
        self.assertEqual(window.spinbox.value, 3)
        self.assertEqual(window.find_calls, [(QSpinBox, "trainingEpochsSpinBox")])

    def test_training_epochs_rejects_unknown_source_kind(self):
        with self.assertRaisesRegex(ValueError, "unsupported Demo source kind"):
            configure_training_epochs(_FakeWindow(), "unknown")


if __name__ == "__main__":
    unittest.main()
