import json
import unittest
from dataclasses import replace

import numpy

from docs.demo.defect_oracle import DefectOracle, Particle
from docs.demo.hard_negative_selection import (
    hard_negative_selection_json,
    resolve_hard_negative_selection,
    select_hard_negative_bags,
)
from docs.demo.wafer_quality_evidence import WaferEvidenceCase
from wafer_defect_studio.grid_geometry import annotation_grids
from wafer_defect_studio.training_input_bundle import (
    TrainingBundleSource,
    TrainingInputBundle,
    TrainingPatchBag,
)


class HardNegativeSelectionTest(unittest.TestCase):
    def setUp(self):
        grids = annotation_grids(4, 2, 2, 2)
        self.case = WaferEvidenceCase(
            "wafer.png",
            "train",
            DefectOracle(4, 2, ()),
            grids,
            numpy.array(
                [[[0.9, 0.8], [0.1, 0.2], [0.7, 0.1], [0.1, 0.1]],
                 [[0.1, 0.1], [0.1, 0.1], [0.1, 0.75], [0.1, numpy.nan]]]
            ),
        )
        sources = (TrainingBundleSource("image-1", "train", "C:/corpus/wafer.png", "fp", "uint8"),)
        bags = tuple(
            TrainingPatchBag(f"image-1:0:{column}", "image-1", 0, column, (), ())
            for column in range(2)
        )
        self.bundle = TrainingInputBundle("snapshot", "split", ("scratch", "particle"), (), sources, (), 2, bags)

    def test_round_robin_selects_unique_empty_bags_and_aggregates_triggers(self):
        payload = select_hard_negative_bags(
            (self.case,), self.bundle, ("scratch", "particle"),
            {"scratch": 0.6, "particle": 0.7}, max_bags=2,
        )

        self.assertEqual(payload["schema"], "spatial-hard-negative-selection.v1")
        self.assertEqual(payload["score_domain"], "absolute_spatial_probability")
        self.assertEqual(payload["source_split"], "train")
        self.assertEqual(payload["class_codes"], ["scratch", "particle"])
        self.assertEqual(payload["counts"], {
            "candidates_by_class": {"scratch": 2, "particle": 2},
            "unique_candidate_bags": 2,
            "selected_bags": 2,
        })
        self.assertEqual([item["bag_id"] for item in payload["selected"]], ["image-1:0:0", "image-1:0:1"])
        self.assertEqual(payload["selected"][0]["trigger_classes"], ["scratch", "particle"])
        self.assertEqual(payload["selected"][0]["peaks"], {"scratch": 0.9, "particle": 0.8})
        self.assertEqual(payload["selected"][1]["trigger_classes"], ["scratch", "particle"])
        canonical = json.loads(hard_negative_selection_json(payload))
        self.assertEqual(canonical, payload)
        self.assertEqual(resolve_hard_negative_selection(payload, ("scratch", "particle"), 2), ("image-1:0:0", "image-1:0:1"))
        self.assertEqual(resolve_hard_negative_selection(canonical, ("scratch", "particle"), 2), ("image-1:0:0", "image-1:0:1"))

    def test_excludes_any_grid_with_oracle_truth_and_requires_finite_trigger(self):
        defect_case = replace(
            self.case,
            oracle=DefectOracle(4, 2, (Particle("particle", (0, 0), 0),)),
            absolute_maps=numpy.array(
                [[[0.99, 0.99], [0.99, 0.99], [0.1, numpy.nan], [0.1, 0.1]],
                 [[0.99, 0.99], [0.99, 0.99], [0.1, 0.1], [0.1, 0.1]]]
            ),
        )
        payload = select_hard_negative_bags(
            (defect_case,), self.bundle, ("scratch", "particle"),
            {"scratch": 0.5, "particle": 0.5}, max_bags=2,
        )
        self.assertEqual(payload["selected"], [])

    def test_round_robin_gives_each_class_a_turn_before_second_scratch_bag(self):
        grids = annotation_grids(6, 2, 2, 2)
        maps = numpy.full((2, 6, 2), 0.1)
        maps[0, 0, 0] = 0.95
        maps[0, 2, 0] = 0.9
        maps[0, 4, 1] = 0.8
        case = WaferEvidenceCase("wafer.png", "train", DefectOracle(6, 2, ()), grids, maps)
        bundle = replace(
            self.bundle,
            patch_bags=tuple(
                TrainingPatchBag(f"image-1:0:{column}", "image-1", 0, column, (), ())
                for column in range(3)
            ),
        )

        payload = select_hard_negative_bags(
            (case,), bundle, ("scratch", "particle"),
            {"scratch": 0.5, "particle": 0.5}, max_bags=2,
        )

        self.assertEqual([item["bag_id"] for item in payload["selected"]],
                         ["image-1:0:0", "image-1:0:2"])

    def test_threshold_is_inclusive_and_peak_ignores_nan_and_infinity(self):
        maps = numpy.full((2, 4, 2), numpy.nan)
        maps[0, 0, 0] = numpy.inf
        maps[0, 1, 0] = 0.6
        case = replace(self.case, absolute_maps=maps)

        payload = select_hard_negative_bags(
            (case,), self.bundle, ("scratch", "particle"),
            {"scratch": 0.6, "particle": 0.7}, max_bags=1,
        )

        self.assertEqual(payload["selected"][0]["bag_id"], "image-1:0:0")
        self.assertEqual(payload["selected"][0]["peaks"], {"scratch": 0.6})

    def test_rejects_case_context_and_join_failures_with_context(self):
        with self.assertRaisesRegex(ValueError, "wafer.png.*validation"):
            select_hard_negative_bags((replace(self.case, split="validation"),), self.bundle,
                                      ("scratch", "particle"), {"scratch": .5, "particle": .5}, max_bags=1)
        with self.assertRaisesRegex(ValueError, "missing.png.*train source"):
            select_hard_negative_bags((replace(self.case, filename="missing.png"),), self.bundle,
                                      ("scratch", "particle"), {"scratch": .5, "particle": .5}, max_bags=1)
        missing_bag = replace(self.bundle, patch_bags=self.bundle.patch_bags[:1])
        with self.assertRaisesRegex(ValueError, "wafer.png.*row 0.*column 1"):
            select_hard_negative_bags((self.case,), missing_bag,
                                      ("scratch", "particle"), {"scratch": .5, "particle": .5}, max_bags=1)

    def test_resolver_rejects_tampered_contract(self):
        payload = select_hard_negative_bags((self.case,), self.bundle, ("scratch", "particle"),
                                            {"scratch": .6, "particle": .7}, max_bags=1)
        with self.assertRaisesRegex(ValueError, "candidate policy"):
            resolve_hard_negative_selection(dict(payload, candidate_policy="changed"), ("scratch", "particle"), 1)
        with self.assertRaisesRegex(ValueError, "class order"):
            resolve_hard_negative_selection(payload, ("particle", "scratch"), 1)
        with self.assertRaisesRegex(ValueError, "max_bags"):
            resolve_hard_negative_selection(payload, ("scratch", "particle"), 2)
        below_threshold = json.loads(hard_negative_selection_json(payload))
        below_threshold["selected"][0]["peaks"]["scratch"] = 0.59
        with self.assertRaisesRegex(ValueError, "scratch.*threshold"):
            resolve_hard_negative_selection(below_threshold, ("scratch", "particle"), 1)


if __name__ == "__main__":
    unittest.main()
