import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "demo"))

from run_phase2_demo import (
    calibrated_profile_thresholds,
    source_configuration,
    validate_demo_regions,
)


class DemoRegionQualityTest(unittest.TestCase):
    def test_calibrated_profile_thresholds_are_class_ordered_and_positive(self):
        staged = {
            "thresholds": {
                "per_class": [
                    {"class_name": "particle", "threshold": 0.71},
                    {"class_name": "scratch", "threshold": 0.62},
                ]
            }
        }

        self.assertEqual(
            calibrated_profile_thresholds(staged, ("scratch", "particle")),
            {"scratch": 0.62, "particle": 0.71},
        )

    def test_zero_threshold_is_rejected(self):
        staged = {
            "thresholds": {
                "per_class": [
                    {"class_name": "scratch", "threshold": 0.0},
                    {"class_name": "particle", "threshold": 0.5},
                ]
            }
        }

        with self.assertRaisesRegex(ValueError, "positive"):
            calibrated_profile_thresholds(staged, ("scratch", "particle"))

    def test_empty_or_full_region_is_rejected(self):
        coverage = np.ones((2, 2), dtype=bool)

        with self.assertRaisesRegex(RuntimeError, "empty"):
            validate_demo_regions(
                {"scratch": np.zeros((2, 2), dtype=bool)},
                coverage,
            )

        with self.assertRaisesRegex(RuntimeError, "all covered"):
            validate_demo_regions(
                {"scratch": np.ones((2, 2), dtype=bool)},
                coverage,
            )

    def test_region_with_partial_coverage_is_accepted(self):
        validate_demo_regions(
            {"scratch": np.array([[True, False], [False, False]])},
            np.ones((2, 2), dtype=bool),
        )

    def test_realistic_source_keeps_20mp_dimensions_and_uses_large_windows(self):
        configuration = source_configuration("realistic", 4472, 4472)

        self.assertEqual(configuration["source_size"], (4472, 4472))
        self.assertEqual(configuration["grid_size"], (512, 512))
        self.assertEqual(configuration["window_size"], (512, 512))
        self.assertEqual(configuration["stride"], (512, 512))


if __name__ == "__main__":
    unittest.main()
