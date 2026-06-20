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

    def test_evidence_index_publishes_truthful_gate_decision(self):
        self.assertEqual(self.gate["overall"], "FAIL")
        for seed in ("17", "42", "91"):
            for class_code in ("scratch", "particle"):
                metrics = self.gate["per_seed"][seed][class_code]["metrics"]
                self.assertGreaterEqual(metrics["defect_instances"], 150)

        evidence_index = (self.root / "docs/demo/README.md").read_text(
            encoding="utf-8"
        )
        for statement in (
            "Spatial MIL v4",
            "Real generated held-out RTX 3090 gate",
            "FAIL",
            "ticket30-quality-gate.json",
            "CAM v2 remains the default",
        ):
            self.assertIn(statement, evidence_index)

    def test_ticket29_screenshots_are_not_presented_as_ticket30_and_10_to_13_absent(self):
        root_documents = (
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
