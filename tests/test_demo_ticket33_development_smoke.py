import json
import unittest
from pathlib import Path


class Ticket33DevelopmentSmokeTest(unittest.TestCase):
    def test_seed_101_passes_both_validation_rows_with_score_separation(self) -> None:
        report = json.loads(
            Path("docs/demo/ticket33-development-smoke.json").read_text(encoding="utf-8")
        )

        self.assertEqual(report["schema"], "ticket33-development-smoke.v1")
        self.assertEqual(report["seed"], 101)
        self.assertEqual(report["selection_source"], "validation")
        self.assertEqual(report["overall"], "PASS")
        self.assertEqual(set(report["per_class"]), {"scratch", "particle"})
        for row in report["per_class"].values():
            self.assertEqual(row["overall"], "PASS")
            self.assertGreater(row["score_separation_margin"], 0.0)
            self.assertEqual(row["defect_coverage_recall"], 1.0)
            self.assertGreaterEqual(row["grid_precision"], 0.97)
            self.assertGreaterEqual(row["grid_recall"], 0.97)
            self.assertLessEqual(row["normal_grid_leak_rate"], 0.03)
            self.assertLessEqual(row["asserted_grid_occupancy_p95"], 0.20)


if __name__ == "__main__":
    unittest.main()
