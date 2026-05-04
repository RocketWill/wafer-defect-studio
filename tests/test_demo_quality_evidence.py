import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "demo"))

from quality_evidence import build_comparison_report, compute_grid_quality_evidence


class DemoQualityEvidenceTest(unittest.TestCase):
    def test_comparison_report_preserves_matched_split_runtime_and_v3_geometry(self):
        evidence = {
            "evidence_status": "measured",
            "grid_evaluation": {
                "per_class": {"scratch": {}, "particle": {}},
                "exact_grid_match": {"count": 1, "total": 2, "rate": 0.5},
            },
            "coarse_localization": {
                "per_class": {"scratch": {}, "particle": {}},
                "asserted_image_support": {
                    "scratch": {"validation": 2, "test": 2},
                    "particle": {"validation": 2, "test": 2},
                },
            },
        }
        cam = {
            "checkpoint_version": "wafer_defect_studio.resnet18.v2",
            "runtime_seconds": {"train": 1.0, "evaluation": 2.0, "detection": 3.0},
            **evidence,
        }
        patch = {
            "checkpoint_version": "wafer_defect_studio.resnet18.v3",
            "runtime_seconds": {"train": 4.0, "evaluation": 5.0, "detection": 6.0},
            **evidence,
            "patch_size": 128,
            "patch_stride": 64,
            "bag_pooling": "max",
        }

        report = build_comparison_report("split-1", cam, patch)

        self.assertEqual(report["split_id"], "split-1")
        self.assertEqual(tuple(report["models"]), ("cam_v2", "patch_v3"))
        self.assertEqual(
            report["models"]["patch_v3"],
            {"split_id": "split-1", **patch},
        )
        self.assertNotIn("approval", report)
        self.assertNotIn("pixel_iou", report)

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

        evidence = compute_grid_quality_evidence(
            ("scratch", "particle"),
            {
                "per_class": {
                    "scratch": {"tp": 2, "fp": 1, "fn": 0, "f1": 0.8},
                    "particle": {"tp": 1, "fp": 1, "fn": 1, "f1": 0.5},
                },
                "exact_grid_match": {"count": 1, "total": 4, "rate": 0.25},
            },
            rows,
        )

        self.assertEqual(evidence["evidence_status"], "insufficient_evidence")
        self.assertEqual(
            evidence["grid_evaluation"]["exact_grid_match"],
            {"count": 1, "total": 4, "rate": 0.25},
        )
        self.assertEqual(
            evidence["grid_evaluation"]["per_class"]["scratch"],
            {"tp": 2, "fp": 1, "fn": 0, "f1": 0.8},
        )
        self.assertEqual(
            evidence["grid_evaluation"]["per_class"]["particle"],
            {"tp": 1, "fp": 1, "fn": 1, "f1": 0.5},
        )
        self.assertEqual(
            evidence["coarse_localization"]["per_class"]["scratch"],
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
            evidence["coarse_localization"]["per_class"]["particle"],
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
            evidence["coarse_localization"]["asserted_image_support"],
            {
                "scratch": {"validation": 1, "test": 1},
                "particle": {"validation": 1, "test": 1},
            },
        )


if __name__ == "__main__":
    unittest.main()
