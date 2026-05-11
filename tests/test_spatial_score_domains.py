import unittest

import numpy as np
import torch

from wafer_defect_studio.cam_detection import (
    generate_all_convolutional_artifact,
    generate_cam_artifact,
)
from wafer_defect_studio.detection_windows import enumerate_inference_windows
from wafer_defect_studio.model_registry import max_pool_patch_logits


class SpatialScoreDomainTest(unittest.TestCase):
    def setUp(self):
        self.window = enumerate_inference_windows(2, 2, window_size=2, stride=2)
        self.provenance = {
            "model_id": "model",
            "profile_id": "profile",
            "evaluation_id": "evaluation",
        }

    def test_patch_detection_scores_are_absolute_sigmoid_probabilities(self):
        artifact = generate_all_convolutional_artifact(
            self.window,
            ((0.2, 0.8),),
            class_names=("scratch", "particle"),
            provenance=self.provenance,
        )

        self.assertEqual(
            artifact.window_settings["map_method"],
            "patch_classification_sigmoid",
        )
        np.testing.assert_allclose(artifact.maps, np.full((2, 2, 2), (0.2, 0.8)))

    def test_local_and_stitched_spatial_scores_preserve_absolute_probabilities(self):
        local = np.array(
            [[[[0.2], [0.4]], [[0.6], [0.8]]]],
            dtype=np.float64,
        )
        artifact = generate_all_convolutional_artifact(
            self.window,
            local,
            class_names=("scratch",),
            provenance=self.provenance,
        )

        self.assertEqual(
            artifact.window_settings["map_method"],
            "all_convolutional_sigmoid",
        )
        np.testing.assert_allclose(artifact.maps[:, :, 0], local[0, :, :, 0])

    def test_grid_and_spatial_thresholds_answer_different_questions(self):
        patch_logits = torch.tensor((((-1.0,), (2.0,)),))
        grid_score = torch.sigmoid(max_pool_patch_logits(patch_logits)).item()
        spatial_scores = torch.sigmoid(patch_logits).numpy().reshape(1, 2)

        self.assertGreaterEqual(grid_score, 0.5)
        np.testing.assert_array_equal(spatial_scores >= 0.5, ((False, True),))

    def test_cam_and_absolute_probability_thresholds_are_not_interchangeable(self):
        cam = generate_cam_artifact(
            self.window,
            np.full((1, 1, 2, 2), 0.2),
            np.ones((1, 1)),
            class_names=("scratch",),
            provenance=self.provenance,
        )
        absolute = generate_all_convolutional_artifact(
            self.window,
            np.full((1, 2, 2, 1), 0.2),
            class_names=("scratch",),
            provenance=self.provenance,
        )

        self.assertTrue(np.all(cam.maps[:, :, 0] >= 0.5))
        self.assertTrue(np.all(absolute.maps[:, :, 0] < 0.5))
        self.assertEqual(cam.to_dict()["artifact_type"], "approximate_cam")
        self.assertEqual(
            absolute.window_settings["map_method"],
            "all_convolutional_sigmoid",
        )

    def test_spatial_mil_map_method_keeps_absolute_values(self):
        artifact = generate_all_convolutional_artifact(
            self.window,
            np.full((1, 2, 2, 1), 0.2),
            class_names=("scratch",),
            provenance=self.provenance,
            map_method="spatial_mil_sigmoid",
        )
        np.testing.assert_allclose(artifact.maps, 0.2)
        self.assertEqual(artifact.provenance["map_method"], "spatial_mil_sigmoid")


if __name__ == "__main__":
    unittest.main()
