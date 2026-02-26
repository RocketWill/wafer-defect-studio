import unittest

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
            known_limits=("Representative hardware timing is not a release guarantee.",),
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


if __name__ == "__main__":
    unittest.main()
