from pathlib import Path
import json
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

    def test_ticket_29_closeout_documents_the_measured_v2_v3_contract(self):
        root = Path(__file__).resolve().parents[1]
        summary = json.loads(
            (root / "docs/demo/screenshots/realistic-20mp/demo-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["corpus"], {"images": 20, "train": 16, "validation": 2, "test": 2})

        v2 = summary["models"]["cam_v2"]
        v3 = summary["models"]["patch_v3"]

        def grid_line(label, report):
            per_class = report["grid_evaluation"]["per_class"]
            exact = report["grid_evaluation"]["exact_grid_match"]
            return (
                f"{label}: scratch F1 {per_class['scratch']['f1']:.4f}, "
                f"particle F1 {per_class['particle']['f1']:.4f}, "
                f"exact Grid match {exact['count']}/{exact['total']}"
            )

        def localization_line(label, report):
            per_class = report["coarse_localization"]["per_class"]
            return (
                f"{label}: scratch intersection/leak "
                f"{per_class['scratch']['intersection_rate']:.4f}/"
                f"{per_class['scratch']['normal_grid_leak_rate']:.4f}; "
                f"particle intersection/leak "
                f"{per_class['particle']['intersection_rate']:.4f}/"
                f"{per_class['particle']['normal_grid_leak_rate']:.4f}"
            )

        required_closeout = (
            "20 images; image-level split 16/2/2",
            "CAM v2 remains the default; Patch Classification v3 is optional.",
            grid_line("v2", v2),
            grid_line("v3", v3),
            localization_line("v2", v2),
            localization_line("v3", v3),
            "Patch geometry: 128 px size, 64 px stride, max pooling.",
            "does not establish segmentation, Neurocle equivalence, or production accuracy.",
        )
        for relative_path in ("README.md", "docs/demo/phase2-end-to-end-tutorial.md"):
            markdown = (root / relative_path).read_text(encoding="utf-8")
            normalized = " ".join(markdown.split())
            for statement in required_closeout:
                self.assertIn(statement, normalized, relative_path)

        tutorial = (root / "docs/demo/phase2-end-to-end-tutorial.md").read_text(
            encoding="utf-8"
        )
        for legacy_name in (
            "04-training-complete.png",
            "05-evaluation-approved.png",
            "06-detection-controls.png",
            "06-heatmap.png",
            "07-regions.png",
            "08-both.png",
        ):
            self.assertNotIn(legacy_name, tutorial)
        self.assertIn("evidence-patch_v3-test-1", tutorial)
        self.assertIn("value-only confidence viewer", tutorial)

        adr = (root / "docs/adr/0004-grid-supervised-patch-classification.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("status: accepted", adr)
        self.assertIn("does not promote v3 to the default", adr)
        conclusion = (
            "v3 reduced particle Normal Grid leakage to "
            f"{v3['coarse_localization']['per_class']['particle']['normal_grid_leak_rate']:.4f}, "
            "but scratch retained "
            f"{v3['coarse_localization']['per_class']['scratch']['normal_grid_leak_rate']:.4f} leakage"
        )
        self.assertIn(conclusion, adr)
        self.assertNotIn("v3 improved scratch Grid F1", adr)

        normalized_tutorial = " ".join(tutorial.split())
        self.assertIn(
            "v3 把 particle Normal Grid leakage "
            f"降到 {v3['coarse_localization']['per_class']['particle']['normal_grid_leak_rate']:.4f}",
            normalized_tutorial,
        )
        self.assertIn(
            "scratch 仍有 "
            f"{v3['coarse_localization']['per_class']['scratch']['normal_grid_leak_rate']:.4f} leakage",
            normalized_tutorial,
        )

        issue = (
            root
            / ".scratch/wafer-defect-classification/issues/29-grid-supervised-patch-classification.md"
        ).read_text(encoding="utf-8")
        self.assertIn("### Slice 29.16", issue)
        self.assertIn("Status: passed", issue[issue.index("### Slice 29.16") :])

        delivery_map = (root / ".scratch/wafer-defect-classification/map.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("- [x] 29 — Grid-supervised Patch Classification", delivery_map)
        self.assertIn("Ticket 29 resolved:", delivery_map)

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
