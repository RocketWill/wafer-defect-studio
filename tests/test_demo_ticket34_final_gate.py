import hashlib
import json
import unittest
from pathlib import Path

from docs.demo.ticket34_contract import FINAL_MEMBERS, FINAL_TARGETS


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "artifacts/ticket34-final-evidence"
REPORT_PATH = REPO_ROOT / "docs/demo/ticket34-final-gate.json"


class Ticket34FinalGateArtifactTest(unittest.TestCase):
    def test_report_seal_maps_rows_targets_and_decision(self) -> None:
        report_bytes = REPORT_PATH.read_bytes()
        report = json.loads(report_bytes.decode("utf-8"))
        generated_report = ARTIFACT_ROOT / "ticket34-final-gate.json"
        self.assertEqual(REPORT_PATH.read_text(encoding="utf-8"), generated_report.read_text(encoding="utf-8"))
        self.assertEqual(
            REPORT_PATH.read_text(encoding="utf-8"),
            json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        )

        seal = json.loads(
            (REPO_ROOT / "docs/demo/ticket34-final-seal.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["seal_sha256"], seal["seal_sha256"])
        self.assertEqual(report["final_members"], list(FINAL_MEMBERS))
        self.assertEqual(report["class_codes"], ["scratch", "particle"])
        self.assertEqual(
            FINAL_TARGETS,
            {
                "defect_instances": 150,
                "defect_coverage_recall": 1.0,
                "grid_precision": 0.95,
                "grid_recall": 0.95,
                "normal_grid_leak_rate": 0.05,
                "asserted_grid_occupancy_p95": 0.25,
            },
        )

        rows = [
            report["per_seed"][str(seed)]["per_class"][code]
            for seed in (17, 42, 91)
            for code in ("scratch", "particle")
        ]
        self.assertEqual(len(rows), 6)

        for seed in (17, 42, 91):
            map_entry = report["map_artifacts"][str(seed)]
            map_path = ARTIFACT_ROOT / map_entry["path"]
            payload = map_path.read_bytes()
            self.assertEqual(map_entry["bytes"], len(payload))
            self.assertEqual(map_entry["sha256"], hashlib.sha256(payload).hexdigest())

        def passes(row: dict[str, object]) -> bool:
            return (
                row["defect_instances"] >= FINAL_TARGETS["defect_instances"]
                and row["defect_coverage_recall"] >= FINAL_TARGETS["defect_coverage_recall"]
                and row["grid_precision"] >= FINAL_TARGETS["grid_precision"]
                and row["grid_recall"] >= FINAL_TARGETS["grid_recall"]
                and row["normal_grid_leak_rate"] <= FINAL_TARGETS["normal_grid_leak_rate"]
                and row["asserted_grid_occupancy_p95"] <= FINAL_TARGETS["asserted_grid_occupancy_p95"]
                and row["score_separation_margin"] > 0.0
            )

        self.assertEqual(report["overall"], "PASS" if all(map(passes, rows)) else "FAIL")
        self.assertEqual(report["overall"], "FAIL")
        self.assertEqual(report["recommendation"], "cam_v2")
        self.assertEqual(report["threshold_source"], "sealed_validation_artifact")
        self.assertEqual(report["final_calibration"], "forbidden")


if __name__ == "__main__":
    unittest.main()
