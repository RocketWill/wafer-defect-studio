import json
import unittest

from docs.demo.ticket34_contract import (
    FINAL_MEMBERS,
    FINAL_TARGETS,
    FROZEN_CORPUS_SHA256,
    SCORE_SEPARATION_MARGIN_MINIMUM,
    TICKET30_MANIFEST_SHA256,
    TICKET33_REPORT_SHA256,
    V6_POLICY,
    build_ticket34_contract,
    canonical_contract_json,
    validate_ticket33_development_report,
)


class Ticket34ContractTest(unittest.TestCase):
    def test_freezes_final_boundary_and_v6_policy(self) -> None:
        contract = build_ticket34_contract(
            ("development-train-a",),
            ("development-validation-b",),
        )

        self.assertEqual(
            FINAL_MEMBERS,
            (
                "ticket30-evidence-17.png",
                "ticket30-evidence-42.png",
                "ticket30-evidence-91.png",
            ),
        )
        self.assertEqual(contract["final_members"], list(FINAL_MEMBERS))
        self.assertEqual(
            TICKET30_MANIFEST_SHA256,
            "c726559838cffde843e28662528c0f5ffff175dac39b6ac9b884891ef594c922",
        )
        self.assertEqual(contract["ticket30_manifest_sha256"], TICKET30_MANIFEST_SHA256)
        self.assertEqual(contract["final_corpus_sha256"], FROZEN_CORPUS_SHA256)
        self.assertEqual(
            TICKET33_REPORT_SHA256,
            "c8c54e09148974cce683c47087da5b328af55e98b8fac612920515c3e704d2d2",
        )
        self.assertEqual(contract["ticket33_report_sha256"], TICKET33_REPORT_SHA256)
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
        self.assertEqual(contract["final_targets"], FINAL_TARGETS)
        self.assertEqual(contract["checkpoint_selection_sources"], ["validation"])
        self.assertEqual(
            contract["development_members"],
            {"train": ["development-train-a"], "validation": ["development-validation-b"]},
        )
        self.assertEqual(contract["default_until_final_pass"], "cam_v2")
        self.assertEqual(
            contract["score_separation_margin"],
            {"operator": ">", "minimum": SCORE_SEPARATION_MARGIN_MINIMUM},
        )
        self.assertEqual(
            V6_POLICY,
            {
                "positive_spatial_evidence": "top_1_percent",
                "absent_class_suppression": "dense",
                "normal_grid_ranking": "same_image",
            },
        )
        self.assertEqual(contract["v6_policy"], V6_POLICY)
        self.assertEqual(
            canonical_contract_json(contract),
            json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )

    def test_requires_the_published_ticket33_pass_report(self) -> None:
        from pathlib import Path

        report_text = Path("docs/demo/ticket33-development-gate.json").read_bytes().decode("utf-8")
        self.assertEqual(
            validate_ticket33_development_report(report_text)["overall"],
            "PASS",
        )
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_ticket33_development_report(report_text + " ")

    def test_rejects_development_or_final_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "train and validation members overlap"):
            build_ticket34_contract(("shared",), ("shared",))
        with self.assertRaisesRegex(ValueError, "final evidence member"):
            build_ticket34_contract(("ticket30-evidence-17.png",), ("validation",))
        with self.assertRaisesRegex(ValueError, "final evidence member"):
            build_ticket34_contract(("train",), ("ticket30-evidence-42.png",))


if __name__ == "__main__":
    unittest.main()
