import unittest

import numpy as np

from docs.demo.ticket38_dense_response_census import build_dense_case_response_census


class Ticket38DenseResponseCensusTest(unittest.TestCase):
    def test_census_separates_core_tolerance_and_fragmentation(self) -> None:
        response = np.zeros((8, 8), dtype=np.float32)
        response[1, 1] = 0.75
        response[6, 5] = 0.625
        truth = {
            "truth_components": [
                {"component_id": 0, "pixels": [[1, 1]]},
                {"component_id": 1, "pixels": [[6, 6]]},
            ]
        }
        matching = {
            "truth_component_count": 2,
            "proposal_count": 2,
            "matched_count": 1,
            "matched_pairs": [
                {"truth_component_id": 0, "proposal_id": 0, "distance_pixels": 0.0}
            ],
            "unmatched_truth_component_ids": [1],
            "unmatched_proposal_ids": [1],
            "proposal_components": [
                {"proposal_id": 0, "truth_component_ids": [0]},
                {"proposal_id": 1, "truth_component_ids": [0]},
            ],
        }

        census = build_dense_case_response_census(
            response,
            truth,
            matching,
            case_id="dense-case",
            class_code="particle",
            threshold=0.5,
            tolerance_pixels=1,
            patch_stride=4,
        )

        self.assertEqual(census["summary"], {
            "truth_component_count": 2,
            "proposal_count": 2,
            "matched_count": 1,
            "core_active_count": 1,
            "tolerance_only_count": 1,
            "below_threshold_count": 0,
            "unmatched_component_count": 1,
            "fragmented_component_count": 1,
            "fragment_excess_proposal_count": 1,
        })
        self.assertEqual(census["components"], [
            {
                "component_id": 0,
                "core_peak": 0.75,
                "core_peak_source_xy": [1, 1],
                "tolerance_peak": 0.75,
                "tolerance_peak_source_xy": [1, 1],
                "threshold_state": "core_active",
                "matched": True,
                "touching_proposal_ids": [0, 1],
                "fragmentation_count": 2,
                "truth_bbox_center_source_xy": [1.0, 1.0],
                "truth_bbox_center_patch_phase_xy": [1.0, 1.0],
            },
            {
                "component_id": 1,
                "core_peak": 0.0,
                "core_peak_source_xy": [6, 6],
                "tolerance_peak": 0.625,
                "tolerance_peak_source_xy": [5, 6],
                "threshold_state": "tolerance_only",
                "matched": False,
                "touching_proposal_ids": [],
                "fragmentation_count": 0,
                "truth_bbox_center_source_xy": [6.0, 6.0],
                "truth_bbox_center_patch_phase_xy": [2.0, 2.0],
            },
        ])


if __name__ == "__main__":
    unittest.main()
