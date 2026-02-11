import unittest

import numpy as np

from wafer_defect_studio.confidence_stitching import (
    StitchedConfidenceMap,
    stitch_window_scores,
)
from wafer_defect_studio.detection_windows import enumerate_inference_windows


class ConfidenceStitchingTest(unittest.TestCase):
    def test_overlapping_window_scores_are_center_weighted_per_class(self):
        windows = enumerate_inference_windows(8, 1, window_size=4, stride=2)
        result = stitch_window_scores(
            windows[:2],
            ((1.0, 0.0), (0.0, 1.0)),
            source_width=8,
            source_height=1,
            class_count=2,
        )

        self.assertIsInstance(result, StitchedConfidenceMap)
        self.assertEqual(result.confidence.shape, (1, 8, 2))
        self.assertEqual(result.coverage.shape, (1, 8))
        np.testing.assert_array_equal(result.coverage, [[1, 1, 2, 2, 1, 1, 0, 0]])
        np.testing.assert_allclose(result.confidence[0, 0], (1.0, 0.0))
        self.assertGreater(result.confidence[0, 2, 0], result.confidence[0, 2, 1])
        self.assertGreater(result.confidence[0, 3, 1], result.confidence[0, 3, 0])
        np.testing.assert_allclose(np.sum(result.confidence[0, 2]), 1.0)
        self.assertTrue(np.isnan(result.confidence[0, 7]).all())


if __name__ == "__main__":
    unittest.main()
