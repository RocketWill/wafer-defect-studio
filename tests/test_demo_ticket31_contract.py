import json
import unittest
from pathlib import Path

from docs.demo.ticket31_contract import (
    DEVELOPMENT_SEEDS,
    FINAL_TARGETS,
    build_ticket31_contract,
    canonical_contract_json,
    validate_ticket30_baseline,
)


class Ticket31ContractTest(unittest.TestCase):
    def test_freezes_v4_baseline_and_disjoint_evidence_boundary(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report_text = (root / "docs/demo/ticket30-quality-gate.json").read_text(
            encoding="utf-8"
        )

        baseline = validate_ticket30_baseline(report_text)
        contract = build_ticket31_contract(
            ("development-train-a",),
            ("development-validation-b",),
        )

        self.assertEqual(DEVELOPMENT_SEEDS, (101, 211, 307, 401, 503))
        self.assertEqual(baseline["overall"], "FAIL")
        self.assertEqual(baseline["recommendation"], "cam_v2")
        self.assertEqual(len(baseline["failed_rows"]), 6)
        self.assertEqual(
            contract["ticket30_report_sha256"],
            "83ed9fb3da660789bc30be984bf016ff2c0370dc997a368f519cd81280cfb4f4",
        )
        self.assertEqual(contract["checkpoint_selection_sources"], ["validation"])
        self.assertEqual(
            contract["final_corpus_sha256"],
            "e733096ecf3bc52970ea88bb20d1d2dbeea36c6a64ed3fcf15ee98467c6cc0b8",
        )
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
        self.assertEqual(
            canonical_contract_json(contract),
            json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )

    def test_rejects_development_overlap_or_final_membership(self) -> None:
        with self.assertRaisesRegex(ValueError, "train and validation members overlap"):
            build_ticket31_contract(("shared",), ("shared",))
        with self.assertRaisesRegex(ValueError, "final evidence member"):
            build_ticket31_contract(("ticket30-evidence-17.png",), ("validation",))


if __name__ == "__main__":
    unittest.main()
