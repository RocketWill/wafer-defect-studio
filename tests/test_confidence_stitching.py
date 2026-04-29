import unittest

import numpy as np

from wafer_defect_studio.confidence_stitching import (
    StitchedConfidenceMap,
    stitch_window_scores,
)
from wafer_defect_studio.detection_windows import enumerate_inference_windows


class ConfidenceStitchingTest(unittest.TestCase):
    def test_model_patch_scalar_scores_blend_in_absolute_source_coordinates(self):
        patches = enumerate_inference_windows(6, 4, window_size=4, stride=2)[:2]

        result = stitch_window_scores(
            patches,
            ((0.8, 0.2), (0.4, 0.9)),
            source_width=6,
            source_height=4,
            class_count=2,
            center_weight="uniform",
        )

        expected_row = np.array(
            (
                (0.8, 0.2),
                (0.8, 0.2),
                (0.6, 0.55),
                (0.6, 0.55),
                (0.4, 0.9),
                (0.4, 0.9),
            )
        )
        np.testing.assert_allclose(
            result.confidence,
            np.repeat(expected_row[np.newaxis, :, :], 4, axis=0),
        )
        np.testing.assert_array_equal(
            result.coverage,
            np.repeat(np.array(((1, 1, 2, 2, 1, 1),)), 4, axis=0),
        )

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
