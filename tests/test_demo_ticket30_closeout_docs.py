import json
import unittest
from pathlib import Path


class Ticket30CloseoutDocsTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.gate = json.loads(
            (self.root / "docs/demo/ticket30-quality-gate.json").read_text(
                encoding="utf-8"
            )
        )

    def test_readme_and_tutorial_publish_truthful_gate_and_metrics(self):
        documents = (
            self.root / "README.md",
            self.root / "docs/demo/phase2-end-to-end-tutorial.md",
        )
        required = (
            "real generated held-out RTX 3090 evidence",
            "Spatial MIL v4",
            "gate FAIL",
            "CAM v2 remains the default",
            "v4 is experimental",
        )
        for document in documents:
            markdown = document.read_text(encoding="utf-8")
            normalized = " ".join(markdown.split())
            for statement in required:
                self.assertIn(statement, normalized, document.name)

            gate_link = "docs/demo/ticket30-quality-gate.json"
            if document.name == "phase2-end-to-end-tutorial.md":
                gate_link = "ticket30-quality-gate.json"
            self.assertIn(gate_link, markdown, document.name)

            for seed in ("17", "42", "91"):
                for class_code in ("scratch", "particle"):
                    metrics = self.gate["per_seed"][seed][class_code]["metrics"]
                    row = (
                        f"{seed} | {class_code} | {metrics['defect_coverage_recall']} | "
                        f"{metrics['grid_precision']} | {metrics['grid_recall']} | "
                        f"{metrics['normal_grid_leak_rate']} | "
                        f"{metrics['asserted_grid_occupancy_p95']} | "
                        f"{metrics['defect_instances']}"
                    )
                    self.assertIn(row, markdown, document.name)

    def test_ticket29_screenshots_are_not_presented_as_ticket30_and_10_to_13_absent(self):
        root_documents = (
            (self.root / "README.md").read_text(encoding="utf-8"),
            (self.root / "docs/demo/phase2-end-to-end-tutorial.md").read_text(
                encoding="utf-8"
            ),
        )
        for markdown in root_documents:
            self.assertIn("Ticket 29", markdown)
            for screenshot_number in ("10", "11", "12", "13"):
                self.assertNotIn(f"{screenshot_number}-", markdown)

        screenshot_dir = self.root / "docs/demo/screenshots/realistic-20mp"
        for screenshot_number in ("10", "11", "12", "13"):
            self.assertFalse(
                any(screenshot_dir.glob(f"{screenshot_number}-*")), screenshot_number
            )


if __name__ == "__main__":
    unittest.main()
