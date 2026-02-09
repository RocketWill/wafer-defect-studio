import unittest

import numpy as np

from wafer_defect_studio.thresholds import (
    DEFAULT_MIN_RECALL_TARGET,
    ThresholdPolicy,
    optimize_thresholds,
)


class ThresholdOptimizationTest(unittest.TestCase):
    def test_policies_are_independent_deterministic_and_show_unmet_recall(self):
        y_true = np.array(
            [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 0],
                [0, 0],
            ],
            dtype=np.int8,
        )
        y_score = np.array(
            [
                [0.9, 0.2],
                [0.4, 0.8],
                [0.3, 0.7],
                [0.2, 0.4],
                [0.1, 0.3],
                [0.05, 0.1],
            ],
            dtype=np.float32,
        )

        result = optimize_thresholds(
            y_true,
            y_score,
            class_names=("scratch", "crack"),
            policies={"scratch": ThresholdPolicy.MAX_F1, "crack": ThresholdPolicy.MAX_FPR},
            max_fpr_target=0.5,
        )

        self.assertEqual(len(result.thresholds), 2)
        self.assertAlmostEqual(result.thresholds[0], 0.4, places=6)
        self.assertAlmostEqual(result.thresholds[1], 0.3, places=6)
        self.assertEqual(result.by_class["scratch"].metrics.f1, 1.0)
        self.assertLessEqual(result.by_class["crack"].metrics.fpr, 0.5)
        self.assertTrue(result.by_class["crack"].target_satisfied)
        self.assertEqual(DEFAULT_MIN_RECALL_TARGET, 0.95)
        unmet = optimize_thresholds(
            np.zeros((2, 1), dtype=np.int8),
            np.array([[0.2], [0.1]], dtype=np.float32),
            class_names=("empty",),
            policy=ThresholdPolicy.MIN_RECALL,
        ).by_class["empty"]
        self.assertEqual(unmet.target, DEFAULT_MIN_RECALL_TARGET)
        self.assertFalse(unmet.target_satisfied)
        self.assertEqual(unmet.metrics.recall, 0.0)
        self.assertIn("optimization target, not guarantee", unmet.target_label)


if __name__ == "__main__":
    unittest.main()
