import json
import unittest
from dataclasses import replace

import numpy

from docs.demo.defect_oracle import DefectOracle, Particle, Scratch
from docs.demo.spatial_thresholds import (
    calibrate_spatial_thresholds,
    resolve_spatial_thresholds,
    spatial_thresholds_json,
)
from docs.demo.wafer_quality_evidence import WaferEvidenceCase, compute_wafer_quality_evidence
from wafer_defect_studio.grid_geometry import annotation_grids


class SpatialThresholdsTest(unittest.TestCase):
    def setUp(self):
        grids = annotation_grids(4, 2, 2, 2)
        self.cases = (
            WaferEvidenceCase(
                "validation.png",
                "validation",
                DefectOracle(
                    4,
                    2,
                    (Scratch("scratch", ((0, 0), (1, 0)), 0), Particle("particle", (3, 1), 0)),
                ),
                grids,
                numpy.array(
                    [
                        [[0.8, 0.1], [0.7, 0.1], [0.2, 0.1], [0.1, 0.1]],
                        [[0.1, 0.1], [0.1, 0.1], [0.1, 0.6], [0.1, 0.9]],
                    ]
                ),
            ),
        )

    def test_selects_highest_feasible_threshold_per_class_and_matches_direct_masks(self):
        payload = calibrate_spatial_thresholds(self.cases, ("scratch", "particle"))

        self.assertEqual(payload["schema"], "spatial-map-thresholds.v1")
        self.assertEqual(payload["score_domain"], "absolute_spatial_probability")
        self.assertEqual(payload["source_split"], "validation")
        self.assertEqual(payload["class_codes"], ["scratch", "particle"])
        self.assertEqual(payload["selected"]["scratch"]["threshold"], 0.8)
        self.assertEqual(payload["selected"]["particle"]["threshold"], 0.9)
        self.assertTrue(payload["selected"]["scratch"]["target_satisfied"])
        self.assertTrue(payload["selected"]["particle"]["target_satisfied"])
        direct = compute_wafer_quality_evidence(
            self.cases, ("scratch", "particle"), {"scratch": 0.8, "particle": 0.9}
        )
        for code in ("scratch", "particle"):
            self.assertEqual(payload["selected"][code]["metrics"], direct["per_class"][code])
        resolved = resolve_spatial_thresholds(payload, ("scratch", "particle"))
        numpy.testing.assert_array_equal(
            self.cases[0].absolute_maps[:, :, 0] >= resolved["scratch"],
            numpy.array([[True, False, False, False], [False, False, False, False]]),
        )

    def test_fallback_is_deterministic_when_no_candidate_meets_targets(self):
        case = WaferEvidenceCase(
            "fallback.png",
            "validation",
            DefectOracle(4, 2, (Particle("particle", (0, 0), 0),)),
            annotation_grids(4, 2, 2, 2),
            numpy.array([[[0.8], [0.8], [0.8], [0.8]], [[0.8], [0.8], [0.8], [0.8]]]),
        )

        first = calibrate_spatial_thresholds((case,), ("particle",))
        second = calibrate_spatial_thresholds((case,), ("particle",))

        self.assertEqual(first, second)
        self.assertEqual(first["selected"]["particle"]["threshold"], 0.8)
        self.assertFalse(first["selected"]["particle"]["target_satisfied"])

    def test_rejects_non_validation_source_and_wrong_domain_or_class_order(self):
        with self.assertRaisesRegex(ValueError, "validation"):
            calibrate_spatial_thresholds(self.cases, ("scratch", "particle"), source_split="train")
        mixed_cases = self.cases + (replace(self.cases[0], filename="test.png", split="test"),)
        with self.assertRaisesRegex(ValueError, "test.png.*test"):
            calibrate_spatial_thresholds(mixed_cases, ("scratch", "particle"))
        payload = calibrate_spatial_thresholds(self.cases, ("scratch", "particle"))
        with self.assertRaisesRegex(ValueError, "score domain"):
            resolve_spatial_thresholds(payload, ("scratch", "particle"), "grid_probability")
        with self.assertRaisesRegex(ValueError, "class order"):
            resolve_spatial_thresholds(payload, ("particle", "scratch"))
        grid_payload = dict(payload, schema="grid-thresholds.v1", score_domain="grid_probability")
        with self.assertRaisesRegex(ValueError, "schema"):
            resolve_spatial_thresholds(grid_payload, ("scratch", "particle"))

    def test_resolver_rejects_tampered_provenance_and_selected_entries(self):
        payload = calibrate_spatial_thresholds(self.cases, ("scratch", "particle"))
        tampered_payloads = (
            (dict(payload, source_split="test"), "source split"),
            (dict(payload, candidate_policy="grid>=threshold"), "candidate policy"),
            (dict(payload, targets={**payload["targets"], "grid_precision": 0.5}), "targets"),
            (dict(payload, selected={**payload["selected"], "scratch": 0.8}), "scratch.*entry"),
            (
                dict(
                    payload,
                    selected={**payload["selected"], "scratch": {**payload["selected"]["scratch"], "threshold": True}},
                ),
                "scratch.*threshold",
            ),
            (
                dict(
                    payload,
                    selected={**payload["selected"], "scratch": {**payload["selected"]["scratch"], "threshold": 1.1}},
                ),
                "scratch.*threshold",
            ),
            (
                dict(
                    payload,
                    selected={**payload["selected"], "scratch": {**payload["selected"]["scratch"], "metrics": None}},
                ),
                "scratch.*metrics",
            ),
            (
                dict(
                    payload,
                    selected={**payload["selected"], "scratch": {**payload["selected"]["scratch"], "target_satisfied": 1}},
                ),
                "scratch.*target_satisfied",
            ),
        )
        for tampered, message in tampered_payloads:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                resolve_spatial_thresholds(tampered, ("scratch", "particle"))

    def test_canonical_json_round_trip_preserves_resolved_thresholds(self):
        payload = calibrate_spatial_thresholds(self.cases, ("scratch", "particle"))
        encoded = spatial_thresholds_json(payload)

        self.assertEqual(encoded, json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
        self.assertEqual(
            resolve_spatial_thresholds(json.loads(encoded), ("scratch", "particle")),
            {"scratch": 0.8, "particle": 0.9},
        )


if __name__ == "__main__":
    unittest.main()
