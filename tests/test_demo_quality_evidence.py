import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "demo"))

from quality_evidence import compute_grid_quality_evidence


class DemoQualityEvidenceTest(unittest.TestCase):
    def test_grid_metrics_and_coarse_localization_are_truthful_and_insufficient(self):
        rows = (
            {
                "image_id": "validation-image",
                "split": "validation",
                "grid": (0, 0),
                "asserted": ("scratch",),
                "predicted": ("scratch",),
                "retained": ("scratch",),
            },
            {
                "image_id": "validation-image",
                "split": "validation",
                "grid": (0, 1),
                "asserted": ("particle",),
                "predicted": ("scratch", "particle"),
                "retained": ("scratch", "particle"),
            },
            {
                "image_id": "test-image",
                "split": "test",
                "grid": (0, 0),
                "asserted": ("scratch", "particle"),
                "predicted": ("scratch",),
                "retained": ("particle",),
            },
            {
                "image_id": "test-image",
                "split": "test",
                "grid": (0, 1),
                "asserted": (),
                "predicted": ("particle",),
                "retained": ("particle",),
            },
        )

        evidence = compute_grid_quality_evidence(("scratch", "particle"), rows)

        self.assertEqual(evidence["evidence_status"], "insufficient_evidence")
        self.assertEqual(evidence["exact_grid_match"], {"count": 1, "total": 4, "rate": 0.25})
        self.assertEqual(
            evidence["per_class"]["scratch"]["classification"],
            {"tp": 2, "fp": 1, "fn": 0, "f1": 0.8},
        )
        self.assertEqual(
            evidence["per_class"]["particle"]["classification"],
            {"tp": 1, "fp": 1, "fn": 1, "f1": 0.5},
        )
        self.assertEqual(
            evidence["per_class"]["scratch"]["localization"],
            {
                "asserted_grid_intersections": 1,
                "asserted_grids": 2,
                "intersection_rate": 0.5,
                "normal_grid_leaks": 1,
                "normal_grids": 2,
                "normal_grid_leak_rate": 0.5,
            },
        )
        self.assertEqual(
            evidence["per_class"]["particle"]["localization"],
            {
                "asserted_grid_intersections": 2,
                "asserted_grids": 2,
                "intersection_rate": 1.0,
                "normal_grid_leaks": 1,
                "normal_grids": 2,
                "normal_grid_leak_rate": 0.5,
            },
        )
        self.assertEqual(
            evidence["asserted_image_support"],
            {
                "scratch": {"validation": 1, "test": 1},
                "particle": {"validation": 1, "test": 1},
            },
        )


if __name__ == "__main__":
    unittest.main()
