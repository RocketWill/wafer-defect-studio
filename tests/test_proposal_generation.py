import unittest

import numpy as np

from wafer_defect_studio.cam_detection import CamDetectionArtifact
from wafer_defect_studio.proposal_generation import generate_confidence_regions, generate_proposals


class ProposalGenerationTest(unittest.TestCase):
    def test_synthetic_maps_generate_deterministic_native_proposals(self):
        maps = np.zeros((8, 13, 2), dtype=np.float64)
        maps[1, 1, 0] = 0.80
        maps[1, 2, 0] = 0.20  # one-pixel gap is closed for scratch
        maps[1, 3:5, 0] = (0.85, 0.88)
        maps[5, 8:11, 0] = 0.95  # removed by minimum-area filtering
        maps[2:5, 2:5, 1] = 0.75  # overlaps the scratch proposal, but is separate

        artifact = CamDetectionArtifact(
            maps=maps,
            coverage=np.ones((8, 13), dtype=np.int32),
            class_names=("scratch", "particle"),
            source_width=13,
            source_height=8,
            source_transform={"coordinate_system": "source-image pixels"},
            window_settings={"window_size": [7, 5]},
            provenance={"run_id": "run-1", "profile_id": "profile-1"},
        )

        settings = {
            "thresholds": {"scratch": 0.8, "particle": 0.7},
            "map_generation": {"closing_radius": 1, "minimum_area": 4},
        }
        proposals = generate_proposals(artifact, settings)

        self.assertEqual(
            [(proposal.class_name, proposal.source_rect, proposal.area) for proposal in proposals],
            [
                ("scratch", (1, 1, 4, 1), 4),
                ("particle", (2, 2, 3, 3), 9),
            ],
        )
        self.assertEqual(proposals, generate_proposals(artifact, settings))
        self.assertAlmostEqual(proposals[0].peak_confidence, 0.88)
        self.assertAlmostEqual(proposals[0].mean_confidence, (0.80 + 0.20 + 0.85 + 0.88) / 4)
        self.assertEqual(proposals[0].provenance["run_id"], "run-1")
        self.assertEqual(proposals[0].provenance["profile_id"], "profile-1")
        maps[0, 0, 0] = np.nan
        regions = generate_confidence_regions(artifact, settings)
        self.assertEqual(regions["scratch"].dtype, np.dtype(bool))
        self.assertEqual(int(regions["scratch"].sum()), proposals[0].area)
        self.assertEqual(int(regions["particle"].sum()), proposals[1].area)
        self.assertFalse(regions["scratch"][0, 0])
        self.assertEqual(
            (int(np.where(regions["scratch"])[1].min()), int(np.where(regions["scratch"])[0].min()),
             int(np.where(regions["scratch"])[1].max() - np.where(regions["scratch"])[1].min() + 1),
             int(np.where(regions["scratch"])[0].max() - np.where(regions["scratch"])[0].min() + 1)),
            proposals[0].source_rect,
        )


if __name__ == "__main__":
    unittest.main()
