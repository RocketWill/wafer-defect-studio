from pathlib import Path
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

    def test_published_report_distinguishes_gui_and_service_evidence(self):
        report_path = (
            Path(__file__).resolve().parents[1]
            / ".scratch"
            / "wafer-defect-classification"
            / "mvp-validation-report.md"
        )
        markdown = report_path.read_text(encoding="utf-8")
        self.assertIn("### GUI-driven smoke validation", markdown)
        self.assertIn("### Service-pipeline validation", markdown)
        for command in (
            "tests.test_gui_validation_smoke.GuiValidationSmokeTest.test_file_actions_create_import_and_display_wafer_image",
            "tests.test_gui_validation_smoke.GuiValidationSmokeTest.test_grid_profile_origin_and_area_controls_persist_through_visible_actions",
            "tests.test_gui_validation_smoke.GuiValidationSmokeTest.test_annotation_and_review_controls_persist_visible_multilabel_workflow",
        ):
            self.assertIn(command, markdown)
        lowered = markdown.lower()
        self.assertNotIn("service-only smoke", lowered)
        self.assertNotIn("complete end-to-end", lowered)


if __name__ == "__main__":
    unittest.main()
