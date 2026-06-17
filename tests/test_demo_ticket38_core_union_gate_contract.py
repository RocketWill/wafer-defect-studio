import copy
import json
import tempfile
import unittest
from pathlib import Path

from docs.demo.ticket38_core_union_gate_contract import (
    ARTIFACT_ROLES,
    CASE_IDS,
    CASE_SPECS,
    CANDIDATE_ARTIFACT_ROOT,
    SCHEMA,
    SOURCE_DEPENDENCIES,
    build_ticket38_core_union_gate_contract,
    canonical_ticket38_core_union_gate_contract_json,
    main,
    validate_ticket38_core_union_gate_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "docs/demo/ticket38-core-union-gate-contract.json"


class Ticket38CoreUnionGateContractTest(unittest.TestCase):
    def test_builder_freezes_dense_gate_contract(self) -> None:
        contract = build_ticket38_core_union_gate_contract()

        self.assertEqual(contract["schema"], SCHEMA)
        self.assertEqual(
            contract["source_geometry"],
            {"coordinate_system": "source_pixels", "width": 1536, "height": 1536},
        )
        self.assertEqual(contract["class_order"], ["scratch", "particle"])

        membership = contract["membership"]
        self.assertEqual(tuple(membership["case_ids"]), CASE_IDS)
        self.assertEqual(tuple(tuple(item) for item in membership["case_specs"]), CASE_SPECS)
        self.assertEqual((membership["seed"], membership["split"]), (101, "train"))

        self.assertEqual(
            contract["recipe"],
            {
                "architecture": "resnet18_spatial_logits_v5",
                "weights_policy": "imagenet",
                "weights_id": "ResNet18_Weights.IMAGENET1K_V1",
                "weights_url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
                "weights_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
                "batch_norm_policy": "frozen_running_stats",
                "epochs": 30,
                "optimizer": "adamw",
                "learning_rate": 0.0003,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 5.0,
                "patch_size": 128,
                "patch_stride": 64,
                "feature_stride": 2,
                "threshold": 0.5,
                "tolerance_pixels": 8,
                "calibration": "forbidden",
            },
        )
        self.assertEqual(
            contract["objective"],
            {
                "loss": "compute_ticket38_objective",
                "terms": [
                    "core_union_coverage_loss",
                    "all_row_hardest_tail_far",
                    "local_full_far",
                ],
                "hardest_fraction": 0.01,
                "context_weights": {
                    "dense_absent_class_loss": 0.5,
                    "same_image_grid_ranking_loss": 0.25,
                    "overlap_consistency_loss": 0.10,
                },
                "forbidden_terms": [
                    "per_instance_coverage_loss",
                    "positive_spatial_topk_loss",
                    "present_sparse_budget_loss",
                ],
            },
        )
        self.assertEqual(contract["evaluation"]["method"], "rendered_truth_components")
        self.assertEqual(
            contract["evaluation"]["cases"],
            [
                {
                    "case_id": CASE_IDS[0],
                    "class_code": "scratch",
                    "truth_components": 1,
                    "proposal_count": 1,
                    "matched_count": 1,
                },
                {
                    "case_id": CASE_IDS[1],
                    "class_code": "scratch",
                    "truth_components": 8,
                    "proposal_count": 8,
                    "matched_count": 8,
                },
                {
                    "case_id": CASE_IDS[2],
                    "class_code": "particle",
                    "truth_components": 1,
                    "proposal_count": 1,
                    "matched_count": 1,
                },
                {
                    "case_id": CASE_IDS[3],
                    "class_code": "particle",
                    "truth_components": 42,
                    "proposal_count": 42,
                    "matched_count": 42,
                },
            ],
        )
        self.assertEqual(
            contract["evaluation"]["gates"],
            {
                "proposal_precision_min": 1.0,
                "proposal_recall_min": 1.0,
                "unique_truth_touch_recall_min": 1.0,
                "unmatched_truth_max": 0,
                "unmatched_proposal_max": 0,
                "cross_component_merge_max": 0,
                "normal_grid_leaks_max": 0,
                "asserted_grid_occupancy_p95_max": 0.25,
            },
        )
        self.assertEqual(
            contract["execution"],
            {
                "gate_run_count": 0,
                "training": "forbidden_until_contract_commit",
                "inference": "forbidden_until_contract_commit",
                "cuda": "forbidden_until_contract_commit",
                "pixel_materialization": "forbidden_until_contract_commit",
            },
        )
        self.assertNotIn("probe_run_count", contract["execution"])
        self.assertNotIn("formal_run_count", contract["execution"])
        self.assertEqual(contract["candidate_artifacts"]["root"], CANDIDATE_ARTIFACT_ROOT)
        self.assertEqual(
            tuple(item["role"] for item in contract["candidate_artifacts"]["roles"]),
            ARTIFACT_ROLES,
        )
        self.assertTrue(all(item["bytes"] is None for item in contract["candidate_artifacts"]["roles"]))
        self.assertTrue(all(item["sha256"] is None for item in contract["candidate_artifacts"]["roles"]))
        paths = {item["path"] for item in contract["source_dependencies"]}
        self.assertIn("docs/demo/ticket38_core_union_probe.py", paths)
        self.assertIn("docs/demo/ticket38-core-union-probe.json", paths)

        validate_ticket38_core_union_gate_contract(contract)
        self.assertEqual(
            canonical_ticket38_core_union_gate_contract_json(contract) + "\n",
            ARTIFACT_PATH.read_text(encoding="utf-8"),
        )

    def test_rejects_gate_membership_path_source_and_execution_drift(self) -> None:
        base = build_ticket38_core_union_gate_contract()
        mutations = (
            (lambda value: value["objective"]["terms"].append("per_instance_coverage_loss"), "objective"),
            (lambda value: value["evaluation"]["gates"].update(proposal_recall_min=0.9), "evaluation"),
            (lambda value: value["membership"]["case_ids"].reverse(), "membership"),
            (lambda value: value["execution"].update(gate_run_count=1), "execution"),
            (
                lambda value: value["candidate_artifacts"]["roles"][0].update(
                    path="artifacts/ticket38-core-union-gate/../escape.pt"
                ),
                "path",
            ),
            (
                lambda value: value["candidate_artifacts"]["roles"][0].update(path="C:/outside.pt"),
                "absolute",
            ),
            (
                lambda value: value["candidate_artifacts"]["roles"][1].update(
                    role=value["candidate_artifacts"]["roles"][0]["role"]
                ),
                "unique",
            ),
            (lambda value: value["candidate_artifacts"]["roles"][0].update(bytes=1), "hashes"),
        )
        for mutate, message in mutations:
            tampered = copy.deepcopy(base)
            mutate(tampered)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_ticket38_core_union_gate_contract(tampered)

        dependency_path = SOURCE_DEPENDENCIES[0][0]
        source = (REPO_ROOT / dependency_path).read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source dependency SHA-256 mismatch"):
            validate_ticket38_core_union_gate_contract(
                base,
                source_texts={dependency_path: source + "\n# tampered"},
            )

    def test_cli_writes_canonical_contract_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "contract.json"
            self.assertEqual(main(["--output", str(output)]), 0)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                ARTIFACT_PATH.read_text(encoding="utf-8"),
            )
            self.assertEqual(main(["--output", str(output)]), 1)

    def test_canonical_json_is_strict(self) -> None:
        contract = build_ticket38_core_union_gate_contract()
        self.assertEqual(
            canonical_ticket38_core_union_gate_contract_json(contract),
            json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )


if __name__ == "__main__":
    unittest.main()
