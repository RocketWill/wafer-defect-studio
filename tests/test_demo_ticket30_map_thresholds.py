import json
import unittest
from dataclasses import replace

import numpy

from docs.demo.defect_oracle import DefectOracle, Particle, Scratch
from docs.demo.ticket30_map_thresholds import canonical_json, calibrate_map_thresholds, resolve_map_thresholds
from docs.demo.wafer_quality_evidence import WaferEvidenceCase
from wafer_defect_studio.grid_geometry import annotation_grids


class Ticket30MapThresholdsTest(unittest.TestCase):
    def setUp(self):
        self.codes = ("scratch", "particle")
        self.cases = (
            WaferEvidenceCase(
                "validation.png",
                "validation",
                DefectOracle(
                    4,
                    2,
                    (Scratch("scratch", ((0, 0), (1, 0)), 0), Particle("particle", (3, 1), 0)),
                ),
                annotation_grids(4, 2, 2, 2),
                numpy.array(
                    [
                        [[0.8, 0.1], [0.7, 0.1], [0.2, 0.1], [0.1, 0.1]],
                        [[0.1, 0.1], [0.1, 0.1], [0.1, 0.6], [0.1, 0.9]],
                    ]
                ),
            ),
        )

    def test_same_maps_select_deterministic_thresholds_in_each_declared_domain(self):
        artifacts = [calibrate_map_thresholds(self.cases, self.codes, domain) for domain in (
            "normalized_window_cam", "patch_sigmoid", "absolute_spatial_probability"
        )]

        self.assertEqual([item["score_domain"] for item in artifacts], [
            "normalized_window_cam", "patch_sigmoid", "absolute_spatial_probability"
        ])
        for artifact in artifacts:
            self.assertEqual(artifact["selected"]["scratch"]["threshold"], 0.8)
            self.assertEqual(artifact["selected"]["particle"]["threshold"], 0.9)
        self.assertEqual(
            [{key: value for key, value in artifact.items() if key != "score_domain"} for artifact in artifacts],
            [{key: value for key, value in artifacts[0].items() if key != "score_domain"}] * 3,
        )

    def test_rejects_non_validation_cases_and_domain_or_artifact_tamper(self):
        with self.assertRaisesRegex(ValueError, "validation"):
            calibrate_map_thresholds(self.cases, self.codes, "patch_sigmoid", source_split="train")
        with self.assertRaisesRegex(ValueError, "test.png.*test"):
            calibrate_map_thresholds((replace(self.cases[0], filename="test.png", split="test"),), self.codes, "patch_sigmoid")
        payload = calibrate_map_thresholds(self.cases, self.codes, "patch_sigmoid")
        with self.assertRaisesRegex(ValueError, "score domain"):
            resolve_map_thresholds(payload, self.codes, "normalized_window_cam", self.cases)
        tampered = json.loads(canonical_json(payload))
        tampered["selected"]["scratch"]["metrics"]["grid_precision"] = 0.5
        with self.assertRaisesRegex(ValueError, "artifact.*match"):
            resolve_map_thresholds(tampered, self.codes, "patch_sigmoid", self.cases)

    def test_canonical_json_is_deterministic_and_resolver_is_strict(self):
        payload = calibrate_map_thresholds(self.cases, self.codes, "absolute_spatial_probability")
        encoded = canonical_json(payload)

        self.assertEqual(encoded, json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
        self.assertEqual(
            resolve_map_thresholds(json.loads(encoded), self.codes, "absolute_spatial_probability", self.cases),
            {"scratch": 0.8, "particle": 0.9},
        )
        for changed in (
            dict(payload, schema="wrong"),
            dict(payload, source_split="test"),
            dict(payload, class_codes=["particle", "scratch"]),
            dict(payload, targets={**payload["targets"], "grid_precision": 0.5}),
            dict(payload, selected={**payload["selected"], "scratch": {**payload["selected"]["scratch"], "threshold": 1.1}}),
        ):
            with self.assertRaises(ValueError):
                resolve_map_thresholds(changed, self.codes, "absolute_spatial_probability", self.cases)

    def test_artifact_records_reduced_quality_breakpoint_policy(self):
        payload = calibrate_map_thresholds(self.cases, self.codes, "patch_sigmoid")

        self.assertEqual(
            payload["candidate_policy"],
            "sorted_unique_finite_per_grid_and_per_defect_support_maxima_plus_0_and_1; map>=threshold",
        )


if __name__ == "__main__":
    unittest.main()
