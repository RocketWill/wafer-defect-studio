import json
import unittest
from pathlib import Path


class Ticket34CloseoutDocsTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.report = json.loads(
            (self.root / "docs/demo/ticket34-final-gate.json").read_text(
                encoding="utf-8"
            )
        )
        self.seal = json.loads(
            (self.root / "docs/demo/ticket34-final-seal.json").read_text(
                encoding="utf-8"
            )
        )

    def test_documents_publish_frozen_final_decision_from_report(self):
        self.assertEqual(self.report["overall"], "FAIL")
        self.assertEqual(self.report["final_members"], [
            "ticket30-evidence-17.png",
            "ticket30-evidence-42.png",
            "ticket30-evidence-91.png",
        ])
        rows = [
            self.report["per_seed"][str(seed)]["per_class"][class_code]
            for seed in (17, 42, 91)
            for class_code in ("scratch", "particle")
        ]
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["defect_instances"] >= 150 for row in rows))
        self.assertTrue(all(row["score_separation_margin"] > 0 for row in rows))
        scratch = [
            self.report["per_seed"][str(seed)]["per_class"]["scratch"]
            for seed in (17, 42, 91)
        ]
        particle = [
            self.report["per_seed"][str(seed)]["per_class"]["particle"]
            for seed in (17, 42, 91)
        ]
        scratch_occupancy = (
            min(row["asserted_grid_occupancy_p95"] for row in scratch),
            max(row["asserted_grid_occupancy_p95"] for row in scratch),
        )
        particle_occupancy = (
            min(row["asserted_grid_occupancy_p95"] for row in particle),
            max(row["asserted_grid_occupancy_p95"] for row in particle),
        )

        documents = (
            self.root / "README.md",
            self.root / "docs/demo/phase2-end-to-end-tutorial.md",
        )
        for document in documents:
            markdown = " ".join(document.read_text(encoding="utf-8").split())
            self.assertIn("Ticket 34", markdown, document.name)
            self.assertIn("frozen final held-out gate", markdown, document.name)
            self.assertIn("3 × 2", markdown, document.name)
            self.assertIn("RTX 3090", markdown, document.name)
            self.assertIn("150 defect instances", markdown, document.name)
            self.assertIn("ticket34-final-gate.json", markdown, document.name)
            self.assertIn("ticket34-final-seal.json", markdown, document.name)
            self.assertIn("scratch", markdown, document.name)
            self.assertIn("0.9", markdown, document.name)
            self.assertIn(
                f"{scratch_occupancy[0]}–{scratch_occupancy[1]}",
                markdown,
                document.name,
            )
            self.assertIn("particle", markdown, document.name)
            self.assertIn("0.5", markdown, document.name)
            self.assertIn("0.125", markdown, document.name)
            self.assertIn(
                f"{particle_occupancy[0]}–{particle_occupancy[1]}",
                markdown,
                document.name,
            )
            self.assertIn("positive score-separation margins", markdown, document.name)
            self.assertIn("CAM v2 remains the default", markdown, document.name)
            self.assertIn("v6 remains experimental", markdown, document.name)
            self.assertIn("No 10–13 screenshots were added", markdown, document.name)
            self.assertIn("01–09", markdown, document.name)
            self.assertIn("Ticket 29", markdown, document.name)
            for claim in ("Neurocle equivalence", "segmentation", "production accuracy"):
                self.assertIn(claim, markdown, document.name)

        seal_reference = self.seal["seal_sha256"]
        self.assertEqual(seal_reference, self.report["seal_sha256"])


if __name__ == "__main__":
    unittest.main()
