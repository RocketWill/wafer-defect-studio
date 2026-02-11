import json
import unittest

import numpy as np

from wafer_defect_studio.cam_detection import (
    CamDetectionArtifact,
    generate_cam_artifact,
)
from wafer_defect_studio.detection_windows import enumerate_inference_windows


class CamDetectionTest(unittest.TestCase):
    def test_cam_maps_keep_class_channels_and_provenance(self):
        windows = enumerate_inference_windows(4, 2, window_size=2, stride=2)
        activations = np.zeros((len(windows), 2, 2, 2), dtype=np.float32)
        activations[:, 0, 0, 0] = 1.0
        activations[:, 1, 1, 1] = 1.0

        artifact = generate_cam_artifact(
            windows,
            activations,
            np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32),
            class_names=("scratch", "particle"),
            provenance={
                "model_id": "model-1",
                "profile_id": "profile-1",
                "evaluation_id": "evaluation-1",
            },
            window_settings={"window_size": [2, 2], "stride": [2, 2]},
        )

        self.assertIsInstance(artifact, CamDetectionArtifact)
        self.assertEqual(artifact.maps.shape, (2, 4, 2))
        self.assertGreater(artifact.class_map("scratch")[0, 0], artifact.class_map("scratch")[0, 1])
        self.assertGreater(artifact.class_map("particle")[1, 1], artifact.class_map("particle")[0, 0])
        self.assertNotEqual(artifact.class_map("scratch").tolist(), artifact.class_map("particle").tolist())

        payload = json.loads(artifact.to_json())
        self.assertEqual(payload["source"]["width"], 4)
        self.assertEqual(payload["source"]["height"], 2)
        self.assertEqual(payload["provenance"]["model_id"], "model-1")
        self.assertEqual(payload["provenance"]["profile_id"], "profile-1")
        self.assertEqual(payload["provenance"]["evaluation_id"], "evaluation-1")
        self.assertEqual(len(payload["source_transform"]["windows"]), len(windows))
        self.assertIn("Approximate localization", payload["disclaimers"])
        self.assertIn("not a segmentation mask", payload["disclaimers"])


if __name__ == "__main__":
    unittest.main()
