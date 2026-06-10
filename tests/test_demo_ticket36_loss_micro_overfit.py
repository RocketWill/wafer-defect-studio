import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from docs.demo.ticket36_loss_micro_overfit import (
    CASE_IDS,
    REPORT_PATH,
    build_ticket36_loss_micro_overfit_report,
    canonical_ticket36_loss_micro_overfit_json,
    validate_ticket36_loss_micro_overfit_report,
)


class Ticket36LossMicroOverfitReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generated_report = build_ticket36_loss_micro_overfit_report()

    def test_tracked_report_is_canonical_and_all_fixed_cases_pass(self) -> None:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        validate_ticket36_loss_micro_overfit_report(report)
        self.assertEqual(report["schema"], "ticket36-loss-micro-overfit.v1")
        self.assertEqual(tuple(row["case_id"] for row in report["per_case"]), CASE_IDS)
        self.assertEqual(report["overall"], "PASS")
        self.assertTrue(all(row["gates"]["overall"] == "PASS" for row in report["per_case"]))
        self.assertEqual(
            canonical_ticket36_loss_micro_overfit_json(
                self.generated_report
            ),
            REPORT_PATH.read_text(encoding="utf-8").rstrip("\n"),
        )

    def test_cases_cover_sparse_dense_boundary_and_close_particles(self) -> None:
        report = self.generated_report
        self.assertGreaterEqual(len(report["per_case"]), 6)
        self.assertEqual(
            set(CASE_IDS),
            {
                "single_particle",
                "dense150_particle",
                "single_scratch",
                "dense150_scratch",
                "boundary_particle",
                "boundary_scratch",
                "close_two_particles",
            },
        )
        cases = {row["case_id"]: row for row in report["per_case"]}
        self.assertGreater(cases["boundary_scratch"]["geometry"]["core_cell_count"], 1)
        self.assertEqual(cases["boundary_particle"]["geometry"]["min_xy"], [0, 0])
        self.assertEqual(cases["close_two_particles"]["instance_count"], 2)

    def test_contract_discloses_cpu_logits_only_and_claim_boundaries(self) -> None:
        report = self.generated_report
        self.assertEqual(report["execution"], {"device": "cpu", "cnn": False, "gpu": False, "images": False})
        self.assertEqual(report["claims"], ["loss-level micro-overfit only", "not segmentation", "not Neurocle equivalence", "not production accuracy"])
        self.assertEqual(report["boundary"]["cam_default"], "cam_v2")
        self.assertFalse(report["artifact_contract"]["seal_status"] == "sealed")

    def test_cli_rewrites_one_canonical_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            completed = subprocess.run(
                [sys.executable, "docs/demo/ticket36_loss_micro_overfit.py", "--output", str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.strip(), "PASS")
            self.assertEqual(output.read_bytes(), REPORT_PATH.read_bytes())
            validate_ticket36_loss_micro_overfit_report(
                json.loads(output.read_text(encoding="utf-8"))
            )


if __name__ == "__main__":
    unittest.main()
