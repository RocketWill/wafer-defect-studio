import copy
import unittest

import numpy as np

import docs.demo.ticket37_proposal_evaluation as evaluation
from docs.demo.defect_oracle import Particle, Scratch
from docs.demo.ticket35_development_corpus import DevelopmentInstance
from docs.demo.ticket37_proposal_evaluation import (
    build_rendered_truth_components,
    match_proposals_to_rendered_truth,
)


class Ticket37ProposalEvaluationTest(unittest.TestCase):
    def test_public_surface_contains_only_the_two_evaluation_seams(self) -> None:
        self.assertEqual(
            evaluation.__all__,
            ["build_rendered_truth_components", "match_proposals_to_rendered_truth"],
        )

    def test_duplicate_touching_truth_is_one_deterministic_component(self) -> None:
        instances = (
            DevelopmentInstance("particle-b", Particle("particle", (3, 3), 0)),
            DevelopmentInstance("particle-duplicate", Particle("particle", (3, 3), 0)),
            DevelopmentInstance("particle-a", Particle("particle", (4, 3), 0)),
        )

        truth = build_rendered_truth_components(instances, (8, 8))

        self.assertEqual(truth["raw_instance_count"], 3)
        self.assertEqual(truth["unique_truth_count"], 2)
        self.assertEqual(truth["duplicate_instance_count"], 1)
        self.assertEqual(truth["truth_component_count"], 1)
        self.assertEqual(
            truth["truth_components"][0]["instance_ids"],
            ["particle-a", "particle-b", "particle-duplicate"],
        )

    def test_different_descriptors_that_share_raster_support_stay_unique(self) -> None:
        truth = build_rendered_truth_components(
            (
                DevelopmentInstance("scratch-forward", Scratch("scratch", ((1, 1), (2, 1)), 0)),
                DevelopmentInstance("scratch-reverse", Scratch("scratch", ((2, 1), (1, 1)), 0)),
            ),
            (4, 4),
        )

        self.assertEqual(truth["raw_instance_count"], 2)
        self.assertEqual(truth["unique_truth_count"], 2)
        self.assertEqual(truth["duplicate_instance_count"], 0)
        self.assertEqual(truth["truth_component_count"], 1)
        self.assertEqual(truth["truth_components"][0]["truth_ids"], [0, 1])

    def test_proposal_matching_reports_precision_recall_and_merge(self) -> None:
        instances = (
            DevelopmentInstance("particle-a", Particle("particle", (1, 1), 0)),
            DevelopmentInstance("particle-b", Particle("particle", (6, 6), 0)),
        )
        truth = build_rendered_truth_components(instances, (8, 8))
        response = np.zeros((8, 8), dtype=bool)
        response[1, 1] = True
        response[6, 6] = True

        report = match_proposals_to_rendered_truth(response, truth, tolerance_pixels=0)

        self.assertEqual(report["proposal_count"], 2)
        self.assertEqual(report["matched_count"], 2)
        self.assertEqual(report["unmatched_truth_component_ids"], [])
        self.assertEqual(report["unmatched_proposal_ids"], [])
        self.assertEqual(report["cross_component_merge_proposal_ids"], [])
        self.assertEqual(report["proposal_precision"], 1.0)
        self.assertEqual(report["proposal_recall"], 1.0)
        self.assertEqual(report["unique_truth_touch_recall"], 1.0)

    def test_unique_truth_touch_recall_counts_supports_inside_one_component(self) -> None:
        truth = build_rendered_truth_components(
            (
                DevelopmentInstance("particle-a", Particle("particle", (2, 2), 0)),
                DevelopmentInstance("particle-b", Particle("particle", (3, 2), 0)),
            ),
            (8, 8),
        )
        response = np.zeros((8, 8), dtype=bool)
        response[2, 2] = True

        report = match_proposals_to_rendered_truth(response, truth, tolerance_pixels=0)

        self.assertEqual(report["truth_component_count"], 1)
        self.assertEqual(report["proposal_recall"], 1.0)
        self.assertEqual(report["unique_truth_touch_recall"], 0.5)

        response[2, 3] = True
        report = match_proposals_to_rendered_truth(response, truth, tolerance_pixels=0)
        self.assertEqual(report["unique_truth_touch_recall"], 1.0)
        self.assertEqual(report["cross_component_merge_proposal_ids"], [])

    def test_proposal_touching_two_separate_components_is_a_cross_component_merge(self) -> None:
        truth = build_rendered_truth_components(
            (
                DevelopmentInstance("particle-a", Particle("particle", (1, 1), 0)),
                DevelopmentInstance("particle-b", Particle("particle", (6, 6), 0)),
            ),
            (8, 8),
        )
        response = np.zeros((8, 8), dtype=bool)
        for coordinate in range(1, 7):
            response[coordinate, coordinate] = True

        report = match_proposals_to_rendered_truth(response, truth, tolerance_pixels=0)

        self.assertEqual(report["proposal_count"], 1)
        self.assertEqual(report["truth_component_count"], 2)
        self.assertEqual(report["matched_count"], 1)
        self.assertEqual(report["cross_component_merge_proposal_ids"], [0])
        self.assertEqual(report["unique_truth_touch_recall"], 1.0)

    def test_empty_boundaries_and_invalid_inputs_fail_closed(self) -> None:
        empty_truth = build_rendered_truth_components((), (4, 4))
        empty_report = match_proposals_to_rendered_truth(
            np.zeros((4, 4), dtype=bool), empty_truth
        )
        self.assertEqual(empty_report["truth_component_count"], 0)
        self.assertEqual(empty_report["proposal_count"], 0)
        self.assertEqual(empty_report["proposal_precision"], 1.0)
        self.assertEqual(empty_report["proposal_recall"], 1.0)
        self.assertEqual(empty_report["unique_truth_touch_recall"], 1.0)

        truth = (DevelopmentInstance("particle", Particle("particle", (1, 1), 0)),)
        with self.assertRaisesRegex(ValueError, "dtype"):
            match_proposals_to_rendered_truth(np.zeros((4, 4), dtype=np.int32), empty_truth)
        with self.assertRaisesRegex(ValueError, "finite"):
            match_proposals_to_rendered_truth(np.array([[np.nan]], dtype=np.float32), empty_truth)
        with self.assertRaisesRegex(ValueError, "threshold"):
            match_proposals_to_rendered_truth(np.zeros((4, 4), dtype=bool), empty_truth, threshold=np.nan)
        with self.assertRaisesRegex(ValueError, "tolerance"):
            match_proposals_to_rendered_truth(np.zeros((4, 4), dtype=bool), empty_truth, tolerance_pixels=-1)
        with self.assertRaisesRegex(ValueError, "same class"):
            build_rendered_truth_components(
                truth + (DevelopmentInstance("scratch", Scratch("scratch", ((0, 0), (1, 1)), 0)),),
                (4, 4),
            )
        with self.assertRaisesRegex(ValueError, "duplicate instance_id"):
            build_rendered_truth_components(
                truth + (DevelopmentInstance("particle", Particle("particle", (2, 2), 0)),),
                (4, 4),
            )
        with self.assertRaisesRegex(ValueError, "extent"):
            build_rendered_truth_components(
                (DevelopmentInstance("outside", Particle("particle", (4, 1), 0)),),
                (4, 4),
            )

    def test_rendered_truth_must_assign_each_unique_truth_to_one_component(self) -> None:
        truth = build_rendered_truth_components(
            (DevelopmentInstance("particle", Particle("particle", (1, 1), 0)),),
            (4, 4),
        )
        malformed = copy.deepcopy(truth)
        malformed["truth_components"][0]["truth_ids"] = []

        with self.assertRaisesRegex(ValueError, "truth IDs"):
            match_proposals_to_rendered_truth(np.zeros((4, 4), dtype=bool), malformed)

    def test_large_distant_support_has_no_truth_touch(self) -> None:
        truth = build_rendered_truth_components(
            (
                DevelopmentInstance(
                    "long-scratch",
                    Scratch("scratch", ((8, 8), (120, 8)), 3),
                ),
            ),
            (256, 256),
        )
        response = np.zeros((256, 256), dtype=bool)
        response[128:, 128:] = True

        report = match_proposals_to_rendered_truth(response, truth, tolerance_pixels=8)

        self.assertEqual(report["proposal_count"], 1)
        self.assertEqual(report["matched_count"], 0)
        self.assertEqual(report["unmatched_truth_component_ids"], [0])
        self.assertEqual(report["unmatched_proposal_ids"], [0])
        self.assertEqual(report["unique_truth_touch_recall"], 0.0)

    def test_matched_distance_is_global_minimum_over_proposal_support(self) -> None:
        truth = build_rendered_truth_components(
            (DevelopmentInstance("particle", Particle("particle", (2, 2), 1)),),
            (6, 6),
        )
        response = np.zeros((6, 6), dtype=bool)
        response[1, 1] = True
        response[2, 2] = True

        report = match_proposals_to_rendered_truth(response, truth, tolerance_pixels=8)

        self.assertEqual(report["matched_count"], 1)
        self.assertEqual(report["matched_pairs"][0]["distance_pixels"], 0.0)


if __name__ == "__main__":
    unittest.main()
