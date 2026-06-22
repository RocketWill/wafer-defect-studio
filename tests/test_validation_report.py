import json
import unittest
from pathlib import Path

from wafer_defect_studio.validation_report import (
    ValidationCheck,
    ValidationReport,
    render_markdown,
)


class ValidationReportTest(unittest.TestCase):
    def test_value_based_checks_render_deterministic_markdown(self):
        checks = (
            ValidationCheck(
                "SQLite capacity",
                True,
                "synthetic metadata stayed within the supported project limit",
                category="capacity",
            ),
            ValidationCheck(
                "CAM localization",
                False,
                "approximate heatmap only; no pixel-accurate segmentation claim",
                category="known-limit",
            ),
        )
        report = ValidationReport(
            environment={"cuda": "available", "python": "3.11"},
            checks=checks,
            known_limits=(
                "Representative hardware timing is not a release guarantee.",
            ),
        )
        markdown = report.to_markdown()
        self.assertEqual(markdown, render_markdown(report))
        self.assertIn("## Environment", markdown)
        self.assertIn("`cuda`: available", markdown)
        self.assertIn("PASS — capacity — SQLite capacity", markdown)
        self.assertIn("FAIL — known-limit — CAM localization", markdown)
        self.assertIn("approximate heatmap only", markdown)
        self.assertIn("## Known limitations", markdown)
        self.assertNotIn("pixel-accurate localization", markdown)

    def test_ticket_29_measurement_artifacts_are_self_consistent(self):
        root = Path(__file__).resolve().parents[1]
        summary = json.loads(
            (
                root / "docs/demo/screenshots/realistic-20mp/demo-summary.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            summary["corpus"],
            {"images": 20, "train": 16, "validation": 2, "test": 2},
        )

        v2 = summary["models"]["cam_v2"]
        v3 = summary["models"]["patch_v3"]
        self.assertEqual(v2["split_id"], summary["split_id"])
        self.assertEqual(v3["split_id"], summary["split_id"])
        self.assertEqual(v2["evidence_status"], "measured")
        self.assertEqual(v3["evidence_status"], "measured")
        self.assertEqual(v3["patch_size"], 128)
        self.assertEqual(v3["patch_stride"], 64)
        self.assertEqual(v3["bag_pooling"], "max")

        adr = (
            root / "docs/adr/0004-grid-supervised-patch-classification.md"
        ).read_text(encoding="utf-8")
        self.assertIn("status: accepted", adr)
        self.assertIn("does not promote v3 to the default", adr)

        screenshot_dir = root / "docs/demo/screenshots/realistic-20mp"
        for name in (
            "01-import.png",
            "02-annotation-two-classes.png",
            "03-dataset-snapshot.png",
            "04-cam-v2-training.png",
            "05-cam-v2-grid-evaluation.png",
            "06-patch-v3-training.png",
            "07-patch-v3-grid-evaluation.png",
            "08-patch-v3-confidence-map.png",
            "09-patch-v3-heatmap-export.png",
            "demo-summary.json",
        ):
            self.assertTrue((screenshot_dir / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
