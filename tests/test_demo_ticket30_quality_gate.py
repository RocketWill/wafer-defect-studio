import copy
import json
import math
import unittest

from docs.demo.ticket30_quality_gate import evaluate_ticket30_spatial_quality


MANIFEST_SHA256 = "a" * 64
SEEDS = (17, 42, 91)
CLASSES = ("scratch", "particle")
METRIC_KEYS = (
    "defect_instances",
    "defect_coverage_recall",
    "grid_precision",
    "grid_recall",
    "normal_grid_leak_rate",
    "asserted_grid_occupancy_p95",
)


def _metrics(**overrides):
    values = {
        "defect_instances": 150,
        "defect_coverage_recall": 1.0,
        "grid_precision": 0.95,
        "grid_recall": 0.95,
        "normal_grid_leak_rate": 0.05,
        "asserted_grid_occupancy_p95": 0.25,
    }
    values.update(overrides)
    return values


def _evidence(changes=None):
    changes = changes or {}
    return {
        seed: {
            code: _metrics(**changes.get((seed, code), {}))
            for code in CLASSES
        }
        for seed in SEEDS
    }


class Ticket30QualityGateTest(unittest.TestCase):
    def test_literal_all_pass_has_exact_deterministic_schema_and_recommendation(self):
        evidence = _evidence()

        first = evaluate_ticket30_spatial_quality(MANIFEST_SHA256, evidence)
        second = evaluate_ticket30_spatial_quality(MANIFEST_SHA256, copy.deepcopy(evidence))

        self.assertEqual(first, second)
        self.assertEqual(
            first,
            {
                "schema": "ticket30-quality-gate.v1",
                "model": "spatial_mil_v4",
                "criteria": {
                    "defect_instances": {"minimum": 150},
                    "defect_coverage_recall": {"minimum": 1.0},
                    "grid_precision": {"minimum": 0.95},
                    "grid_recall": {"minimum": 0.95},
                    "normal_grid_leak_rate": {"maximum": 0.05},
                    "asserted_grid_occupancy_p95": {"maximum": 0.25},
                },
                "manifest_sha256": MANIFEST_SHA256,
                "per_seed": {
                    seed: {
                        code: {
                            "metrics": _metrics(),
                            "criteria": {key: True for key in METRIC_KEYS},
                            "overall": "PASS",
                        }
                        for code in CLASSES
                    }
                    for seed in SEEDS
                },
                "overall": "PASS",
                "recommendation": "spatial_mil_v4",
            },
        )
        self.assertEqual(
            json.dumps(first, sort_keys=True, separators=(",", ":")),
            json.dumps(second, sort_keys=True, separators=(",", ":")),
        )

    def test_one_seed_class_failure_cannot_be_hidden_by_other_seeds(self):
        evidence = _evidence({(42, "particle"): {"grid_precision": 0.94}})

        result = evaluate_ticket30_spatial_quality(MANIFEST_SHA256, evidence)

        self.assertEqual(result["per_seed"][17]["scratch"]["overall"], "PASS")
        self.assertEqual(result["per_seed"][42]["particle"]["overall"], "FAIL")
        self.assertFalse(result["per_seed"][42]["particle"]["criteria"]["grid_precision"])
        self.assertEqual(result["overall"], "FAIL")
        self.assertEqual(result["recommendation"], "cam_v2")

    def test_nullable_precision_is_valid_evidence_but_fails_its_criterion(self):
        evidence = _evidence({(91, "scratch"): {"grid_precision": None}})

        result = evaluate_ticket30_spatial_quality(MANIFEST_SHA256, evidence)

        entry = result["per_seed"][91]["scratch"]
        self.assertIsNone(entry["metrics"]["grid_precision"])
        self.assertFalse(entry["criteria"]["grid_precision"])
        self.assertEqual(entry["overall"], "FAIL")
        self.assertEqual(result["overall"], "FAIL")

    def test_rejects_membership_field_type_and_manifest_tamper(self):
        valid = _evidence()
        mutations = (
            (lambda value: value.pop(17), "seed membership"),
            (lambda value: value.__setitem__(99, copy.deepcopy(value[17])), "seed membership"),
            (lambda value: value[17].pop("scratch"), "class membership"),
            (lambda value: value[17].__setitem__("other", copy.deepcopy(value[17]["particle"])), "class membership"),
            (lambda value: value[17]["scratch"].pop("grid_recall"), "metric fields"),
            (lambda value: value[17]["scratch"].__setitem__("extra", 0.0), "metric fields"),
            (lambda value: value[17]["scratch"].__setitem__("defect_instances", True), "boolean"),
            (lambda value: value[17]["scratch"].__setitem__("grid_precision", False), "boolean"),
            (lambda value: value[17]["scratch"].__setitem__("grid_precision", math.nan), "finite"),
            (lambda value: value[17]["scratch"].__setitem__("grid_precision", math.inf), "finite"),
        )
        for mutate, message in mutations:
            evidence = copy.deepcopy(valid)
            mutate(evidence)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                evaluate_ticket30_spatial_quality(MANIFEST_SHA256, evidence)

        for manifest_sha256 in ("A" * 64, "a" * 63, "g" * 64, None):
            with self.subTest(manifest_sha256=manifest_sha256), self.assertRaisesRegex(ValueError, "manifest SHA-256"):
                evaluate_ticket30_spatial_quality(manifest_sha256, valid)


if __name__ == "__main__":
    unittest.main()
