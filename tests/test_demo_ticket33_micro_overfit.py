import json
import unittest
from pathlib import Path


class Ticket33MicroOverfitTest(unittest.TestCase):
    def test_four_compositions_overfit_with_positive_grid_separation(self) -> None:
        report = json.loads(
            Path("docs/demo/ticket33-micro-overfit.json").read_text(encoding="utf-8")
        )

        self.assertEqual(report["schema"], "ticket33-micro-overfit.v1")
        self.assertEqual(report["seed"], 101)
        self.assertEqual(
            tuple(report["compositions"]),
            ("normal", "scratch", "particle", "both"),
        )
        self.assertEqual(report["source_split"], "train")
        self.assertEqual(set(report["per_class"]), {"scratch", "particle"})
        for row in report["per_class"].values():
            self.assertGreater(row["weakest_asserted_grid_top_score"], row["hardest_normal_grid_top_score"])
            self.assertGreater(row["margin"], 0.0)
            self.assertLessEqual(row["asserted_grid_occupancy_p95"], 0.20)


if __name__ == "__main__":
    unittest.main()
