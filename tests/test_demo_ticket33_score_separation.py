import unittest
from pathlib import Path

from docs.demo.ticket33_score_separation import analyze_ticket32_score_separation


class Ticket33ScoreSeparationTest(unittest.TestCase):
    def test_ticket32_maps_expose_failed_asserted_vs_normal_grid_separation(self) -> None:
        report = analyze_ticket32_score_separation(
            Path("docs/demo/ticket32-development-gate.json"),
            Path("artifacts/ticket32-development-evidence"),
        )

        self.assertEqual(report["schema"], "ticket33-score-separation.v1")
        self.assertEqual(tuple(report["per_seed"]), ("101", "211", "307", "401", "503"))
        rows = [
            row
            for seed in report["per_seed"].values()
            for row in seed.values()
        ]
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["top_fraction"] == 0.01 for row in rows))
        self.assertTrue(all(
            abs(
                row["margin"]
                - (row["weakest_asserted_grid_top_score"] - row["hardest_normal_grid_top_score"])
            ) < 1e-12
            for row in rows
        ))
        self.assertGreaterEqual(sum(row["collapsed"] for row in rows), 8)


if __name__ == "__main__":
    unittest.main()
