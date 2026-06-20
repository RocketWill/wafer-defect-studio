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

        self.assertEqual(scratch_occupancy, (
            0.5312423706054688,
            0.5492210388183594,
        ))
        self.assertEqual(particle_occupancy, (
            0.8409576416015625,
            0.8425254821777344,
        ))

        evidence_index = (self.root / "docs/demo/README.md").read_text(
            encoding="utf-8"
        )
        for statement in (
            "Grid-contrastive Spatial MIL v6",
            "Frozen final held-out gate",
            "FAIL",
            "ticket34-final-gate.json",
            "CAM v2 remains the default",
            "Spatial MIL remains experimental",
        ):
            self.assertIn(statement, evidence_index)

        seal_reference = self.seal["seal_sha256"]
        self.assertEqual(seal_reference, self.report["seal_sha256"])


if __name__ == "__main__":
    unittest.main()
