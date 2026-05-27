import hashlib
import json
import unittest
from pathlib import Path


class Ticket32DevelopmentGateTest(unittest.TestCase):
    def test_published_report_preserves_failed_repaired_decision_and_artifacts(self) -> None:
        report_path = Path("docs/demo/ticket32-development-gate.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["schema"], "ticket32-development-gate.v1")
        self.assertEqual(report["overall"], "FAIL")
        self.assertEqual(report["recommendation"], "cam_v2")
        self.assertEqual(report["spatial_mil_v5_status"], "experimental")
        self.assertEqual(tuple(report["per_seed"]), ("101", "211", "307", "401", "503"))

        rows = [
            row
            for seed in report["per_seed"].values()
            for row in seed["classes"].values()
        ]
        self.assertEqual(sum(row["overall"] == "PASS" for row in rows), 2)
        self.assertEqual(sum(row["overall"] == "FAIL" for row in rows), 8)
        self.assertEqual(
            {seed["selected_epoch"] for seed in report["per_seed"].values()},
            {1},
        )

        evidence_root = Path("artifacts/ticket32-development-evidence")
        artifacts = [
            artifact
            for seed in report["per_seed"].values()
            for artifact in seed["artifacts"].values()
        ]
        self.assertEqual(len(artifacts), 35)
        for artifact in artifacts:
            path = evidence_root / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
