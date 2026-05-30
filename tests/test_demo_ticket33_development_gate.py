import hashlib
import json
import unittest
from pathlib import Path


class Ticket33DevelopmentGateTest(unittest.TestCase):
    def test_published_five_seed_decision_and_artifacts_are_complete(self) -> None:
        report = json.loads(
            Path("docs/demo/ticket33-development-gate.json").read_text(encoding="utf-8")
        )

        self.assertEqual(report["schema"], "ticket33-development-gate.v1")
        self.assertEqual(tuple(report["per_seed"]), ("101", "211", "307", "401", "503"))
        rows = [row for seed in report["per_seed"].values() for row in seed["per_class"].values()]
        self.assertEqual(len(rows), 10)
        expected = "PASS" if all(row["overall"] == "PASS" for row in rows) else "FAIL"
        self.assertEqual(report["overall"], expected)
        self.assertEqual(
            report["recommendation"],
            "run_final_held_out_gate" if expected == "PASS" else "cam_v2",
        )
        self.assertEqual(
            report["spatial_mil_v6_status"],
            "development_pass" if expected == "PASS" else "experimental",
        )

        root = Path("artifacts/ticket33-development-evidence")
        artifacts = [artifact for seed in report["per_seed"].values() for artifact in seed["artifacts"].values()]
        self.assertEqual(len(artifacts), 15)
        for artifact in artifacts:
            path = root / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
