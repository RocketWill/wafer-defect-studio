import tempfile
import json
import unittest
from pathlib import Path

from docs.demo.ticket31_development_corpus import build_ticket31_development_corpus
from docs.demo.ticket31_development_gate import (
    _materialize_bundle,
    _row_result,
    _training_config,
)


class Ticket31DevelopmentGateTest(unittest.TestCase):
    def test_published_report_preserves_failed_five_seed_decision(self) -> None:
        report = json.loads(
            Path("docs/demo/ticket31-development-gate.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["overall"], "FAIL")
        self.assertEqual(report["recommendation"], "cam_v2")
        self.assertEqual(tuple(report["per_seed"]), ("101", "211", "307", "401", "503"))
        rows = [row for seed in report["per_seed"].values() for row in seed["classes"].values()]
        self.assertEqual(sum(row["overall"] == "PASS" for row in rows), 3)
        self.assertEqual(sum(row["overall"] == "FAIL" for row in rows), 7)
        artifacts = [artifact for seed in report["per_seed"].values()
                     for artifact in seed["artifacts"].values()]
        self.assertEqual(len(artifacts), 30)
        self.assertTrue(all(len(artifact["sha256"]) == 64 for artifact in artifacts))

    def test_materialized_seed_bundle_contains_train_and_validation_but_no_test(self) -> None:
        cases = tuple(case for case in build_ticket31_development_corpus() if case.seed == 101)
        with tempfile.TemporaryDirectory() as temporary_directory:
            bundle = _materialize_bundle(
                cases, _training_config(101), Path(temporary_directory)
            ).bundle
        self.assertEqual({source.split for source in bundle.sources}, {"train", "validation"})
        self.assertEqual(len(bundle.sources), 18)
        self.assertEqual(len(bundle.patch_bags), 18 * 9)
        self.assertTrue(all(len(bag.patches) == 49 for bag in bundle.patch_bags))

    def test_row_requires_every_frozen_target(self) -> None:
        passing = {
            "defect_coverage_recall": 1.0,
            "grid_precision": 0.97,
            "grid_recall": 0.97,
            "normal_grid_leak_rate": 0.03,
            "asserted_grid_occupancy_p95": 0.20,
        }
        self.assertEqual(_row_result(passing), "PASS")
        self.assertEqual(_row_result({**passing, "grid_precision": 0.969}), "FAIL")


if __name__ == "__main__":
    unittest.main()
