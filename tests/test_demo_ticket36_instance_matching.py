import unittest

import numpy as np

from docs.demo.defect_oracle import Particle, Scratch
from docs.demo.ticket35_development_corpus import DevelopmentInstance
from docs.demo.ticket36_instance_matching import match_ticket36_instance_peaks


class Ticket36InstanceMatchingTest(unittest.TestCase):
    def test_separate_particle_components_match_all(self) -> None:
        response = np.zeros((8, 8), dtype=bool)
        response[1, 1] = True
        response[6, 6] = True
        instances = (
            DevelopmentInstance("particle-a", Particle("particle", (1, 1), 0)),
            DevelopmentInstance("particle-b", Particle("particle", (6, 6), 0)),
        )

        report = match_ticket36_instance_peaks(response, instances, tolerance_pixels=0)

        self.assertEqual(report["instance_count"], 2)
        self.assertEqual(report["component_count"], 2)
        self.assertEqual(report["matched"], 2)
        self.assertEqual(report["unmatched_instance_ids"], [])
        self.assertEqual(report["unmatched_component_ids"], [])
        self.assertEqual(report["merged_component_ids"], [])
        self.assertEqual(report["recall"], 1.0)
        self.assertEqual(
            [(item["instance_id"], item["component_id"]) for item in report["matched_pairs"]],
            [("particle-a", 0), ("particle-b", 1)],
        )

    def test_one_blob_reachable_to_two_truths_matches_once_and_flags_merge(self) -> None:
        response = np.zeros((7, 8), dtype=bool)
        response[3, 2:6] = True
        instances = (
            DevelopmentInstance("particle-a", Particle("particle", (2, 3), 0)),
            DevelopmentInstance("particle-b", Particle("particle", (5, 3), 0)),
        )

        report = match_ticket36_instance_peaks(response, instances, tolerance_pixels=3)

        self.assertEqual(report["component_count"], 1)
        self.assertEqual(report["matched"], 1)
        self.assertEqual(report["unmatched_instance_ids"], ["particle-b"])
        self.assertEqual(report["unmatched_component_ids"], [])
        self.assertEqual(report["merged_component_ids"], [0])

    def test_long_single_blob_uses_all_cells_for_merge_diagnostic(self) -> None:
        response = np.ones((1, 11), dtype=bool)
        instances = (
            DevelopmentInstance("particle-left", Particle("particle", (0, 0), 0)),
            DevelopmentInstance("particle-right", Particle("particle", (10, 0), 0)),
        )

        report = match_ticket36_instance_peaks(response, instances, tolerance_pixels=1)

        self.assertEqual(report["component_count"], 1)
        self.assertEqual(report["matched"], 1)
        self.assertEqual(report["unmatched_instance_ids"], ["particle-right"])
        self.assertEqual(report["merged_component_ids"], [0])
        self.assertNotIn("_cells", report["component_peaks"][0])

    def test_augmenting_path_finds_maximum_cardinality_matching(self) -> None:
        response = np.zeros((1, 5), dtype=bool)
        response[0, 0] = True
        response[0, 4] = True
        instances = (
            DevelopmentInstance("particle-flexible", Particle("particle", (2, 0), 0)),
            DevelopmentInstance("particle-fixed", Particle("particle", (0, 0), 0)),
        )

        report = match_ticket36_instance_peaks(response, instances, tolerance_pixels=2)

        self.assertEqual(report["matched"], 2)
        self.assertEqual(report["unmatched_instance_ids"], [])
        self.assertEqual(
            [(item["instance_id"], item["component_id"]) for item in report["matched_pairs"]],
            [("particle-flexible", 1), ("particle-fixed", 0)],
        )

    def test_scratch_uses_full_polyline_segment_not_only_endpoints(self) -> None:
        response = np.zeros((5, 7), dtype=bool)
        response[2, 3] = True
        instances = (
            DevelopmentInstance(
                "scratch-midpoint",
                Scratch("scratch", ((0, 2), (6, 2)), 0),
            ),
        )

        report = match_ticket36_instance_peaks(response, instances, tolerance_pixels=0)

        self.assertEqual(report["matched"], 1)
        self.assertEqual(report["recall"], 1.0)
        self.assertEqual(report["unmatched_instance_ids"], [])

    def test_float_probability_map_keeps_highest_peak_and_uses_eight_connectivity(self) -> None:
        response = np.zeros((4, 5), dtype=np.float32)
        response[1, 1] = 0.6
        response[2, 2] = 0.9
        instances = (
            DevelopmentInstance("particle", Particle("particle", (2, 2), 0)),
        )

        report = match_ticket36_instance_peaks(response, instances, threshold=0.5, tolerance_pixels=0)

        self.assertEqual(report["component_count"], 1)
        self.assertEqual(report["component_peaks"][0]["peak"], [2.0, 2.0])
        self.assertEqual(report["matched"], 1)

    def test_unmatched_instance_and_component_ids_are_reported(self) -> None:
        response = np.zeros((8, 8), dtype=bool)
        response[1, 1] = True
        response[6, 6] = True
        instances = (
            DevelopmentInstance("particle-seen", Particle("particle", (1, 1), 0)),
            DevelopmentInstance("particle-missed", Particle("particle", (3, 3), 0)),
        )

        report = match_ticket36_instance_peaks(response, instances, tolerance_pixels=0)

        self.assertEqual(report["matched"], 1)
        self.assertEqual(report["unmatched_instance_ids"], ["particle-missed"])
        self.assertEqual(report["unmatched_component_ids"], [1])
        self.assertAlmostEqual(report["recall"], 0.5)

    def test_empty_response_and_empty_truth_boundaries_are_explicit(self) -> None:
        empty_response = np.zeros((4, 4), dtype=bool)
        empty_report = match_ticket36_instance_peaks(empty_response, (), tolerance_pixels=8)
        self.assertEqual(empty_report["instance_count"], 0)
        self.assertEqual(empty_report["component_count"], 0)
        self.assertEqual(empty_report["matched"], 0)
        self.assertEqual(empty_report["recall"], 1.0)

        missing_report = match_ticket36_instance_peaks(
            empty_response,
            (DevelopmentInstance("particle-missed", Particle("particle", (1, 1), 0)),),
            tolerance_pixels=8,
        )
        self.assertEqual(missing_report["matched"], 0)
        self.assertEqual(missing_report["unmatched_instance_ids"], ["particle-missed"])
        self.assertEqual(missing_report["recall"], 0.0)

    def test_invalid_response_truth_and_matching_parameters_fail_closed(self) -> None:
        truth = (DevelopmentInstance("particle", Particle("particle", (1, 1), 0)),)
        invalid_responses = (
            (np.zeros((2, 2, 1), dtype=bool), "2-D"),
            (np.zeros((2, 2), dtype=np.int32), "boolean or floating"),
            (np.array([[np.nan]], dtype=np.float32), "finite"),
        )
        for response, message in invalid_responses:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    match_ticket36_instance_peaks(response, truth)

        for kwargs, message in (
            ({"threshold": np.nan}, "threshold"),
            ({"threshold": 1.1}, "threshold"),
            ({"tolerance_pixels": -1}, "tolerance"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    match_ticket36_instance_peaks(np.zeros((2, 2), dtype=bool), truth, **kwargs)

        mixed_truth = (
            truth[0],
            DevelopmentInstance("scratch", Scratch("scratch", ((0, 0), (1, 1)), 0)),
        )
        with self.assertRaisesRegex(ValueError, "same class"):
            match_ticket36_instance_peaks(np.zeros((2, 2), dtype=bool), mixed_truth)

        duplicate_ids = (
            truth[0],
            DevelopmentInstance("particle", Particle("particle", (2, 2), 0)),
        )
        with self.assertRaisesRegex(ValueError, "instance_id"):
            match_ticket36_instance_peaks(np.zeros((3, 3), dtype=bool), duplicate_ids)

        with self.assertRaisesRegex(ValueError, r"instances\[0\].*center.*extent"):
            match_ticket36_instance_peaks(
                np.zeros((2, 2), dtype=bool),
                (DevelopmentInstance("particle-outside", Particle("particle", (2, 1), 0)),),
            )

        with self.assertRaisesRegex(ValueError, r"instances\[0\].*points\[1\].*extent"):
            match_ticket36_instance_peaks(
                np.zeros((2, 2), dtype=bool),
                (
                    DevelopmentInstance(
                        "scratch-outside",
                        Scratch("scratch", ((0, 0), (0, 2)), 0),
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
