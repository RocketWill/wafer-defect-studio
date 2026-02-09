import unittest

import numpy as np

from wafer_defect_studio.grid_evaluation import (
    ClassMetrics,
    compute_grid_metrics,
)


class GridEvaluationTest(unittest.TestCase):
    def test_multilabel_metrics_are_independent_and_cover_empty_cases(self):
        y_true = np.array(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=np.int8,
        )
        y_score = np.array(
            [
                [0.9, 0.2, 0.1, 0.9],
                [0.8, 0.7, 0.2, 0.8],
                [0.3, 0.6, 0.3, 0.7],
                [0.4, 0.1, 0.2, 0.6],
                [0.2, 0.9, 0.4, 0.8],
            ],
            dtype=np.float32,
        )

        result = compute_grid_metrics(
            y_true,
            y_score,
            thresholds=(0.5, 0.5, 0.5, 0.5),
            class_names=("scratch", "crack", "empty", "false-only"),
        )

        scratch, crack, empty, false_only = result.per_class
        self.assertAlmostEqual(result.macro_f1, 0.4, places=7)
        self.assertEqual(scratch, ClassMetrics("scratch", 1.0, 2 / 3, 0.8, 3, 0.0, 2, 2, 0, 1))
        self.assertEqual(crack, ClassMetrics("crack", 2 / 3, 1.0, 0.8, 2, 1 / 3, 2, 2, 1, 0))
        self.assertEqual(empty, ClassMetrics("empty", 0.0, 0.0, 0.0, 0, 0.0, 0, 5, 0, 0))
        self.assertEqual(false_only, ClassMetrics("false-only", 0.0, 0.0, 0.0, 0, 1.0, 0, 0, 5, 0))
        self.assertFalse(hasattr(result, "iou"))
        self.assertFalse(hasattr(result, "dice"))
        self.assertFalse(hasattr(result, "pixel_accuracy"))


if __name__ == "__main__":
    unittest.main()
