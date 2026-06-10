import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from docs.demo.ticket36_single_image_cnn_micro_overfit import (
    CASE_IDS,
    CASE_SPECS,
    FEATURE_STRIDE,
    PATCH_SIZE,
    PATCH_STRIDE,
    SCHEMA,
    build_ticket36_cases,
    build_patch_supervision,
    canonical_ticket36_single_image_report_json,
    validate_ticket36_single_image_report,
    verify_ticket36_artifact_seal,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "docs/demo/ticket36-single-image-cnn-micro-overfit.json"


class Ticket36SingleImageCnnMicroOverfitTest(unittest.TestCase):
    def test_case_membership_and_frozen_recipe(self) -> None:
        self.assertEqual(
            tuple((case.composition, case.scratch_count, case.particle_count) for case in build_ticket36_cases()),
            CASE_SPECS,
        )
        self.assertEqual(tuple(case.case_id for case in build_ticket36_cases()), CASE_IDS)
        self.assertTrue(all(case.seed == 101 and case.split == "train" for case in build_ticket36_cases()))

    def test_source_masks_pool_to_feature_masks_and_far_never_hits_same_class_core(self) -> None:
        case = build_ticket36_cases()[0]
        instance = case.instances[0].defect
        point = instance.center if hasattr(instance, "center") else instance.points[0]
        left = max(0, min(case.oracle.image_width - PATCH_SIZE, point[0] - PATCH_SIZE // 2))
        top = max(0, min(case.oracle.image_height - PATCH_SIZE, point[1] - PATCH_SIZE // 2))
        supervision = build_patch_supervision(
            case,
            ((left, top, PATCH_SIZE, PATCH_SIZE),),
            is_normal_grid=False,
        )
        self.assertEqual(supervision.source_core_masks.shape[-2:], (PATCH_SIZE, PATCH_SIZE))
        self.assertEqual(supervision.feature_core_masks.shape[-2:], (PATCH_SIZE // FEATURE_STRIDE, PATCH_SIZE // FEATURE_STRIDE))
        self.assertEqual(supervision.source_tolerance_union.shape, (2, PATCH_SIZE, PATCH_SIZE))
        self.assertEqual(supervision.far_negative_masks.shape, (2, PATCH_SIZE // FEATURE_STRIDE, PATCH_SIZE // FEATURE_STRIDE))
        self.assertTrue(torch.equal(
            supervision.feature_core_masks.any(dim=0),
            supervision.feature_core_masks.any(dim=0) & ~supervision.far_negative_masks[0],
        ))
        self.assertTrue(not bool((supervision.feature_core_masks.any(dim=0) & supervision.far_negative_masks[0]).any().item()))

    def test_normal_grid_far_mask_ignores_adjacent_class_tolerance(self) -> None:
        case = build_ticket36_cases()[-1]
        rect = (1024, 0, PATCH_SIZE, PATCH_SIZE)
        asserted_semantics = build_patch_supervision(
            case, (rect,), is_normal_grid=False
        )
        normal_semantics = build_patch_supervision(
            case, (rect,), is_normal_grid=True
        )
        self.assertEqual(int((~asserted_semantics.far_negative_masks[1]).sum()), 19)
        self.assertTrue(bool(normal_semantics.far_negative_masks.all().item()))

    def test_tracked_report_is_canonical_and_gate_validator_is_fail_closed(self) -> None:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        validate_ticket36_single_image_report(report, repo_root=REPO_ROOT)
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(tuple(row["case_id"] for row in report["per_case"]), CASE_IDS)
        self.assertIn(report["overall"], {"PASS", "FAIL"})
        self.assertEqual(
            canonical_ticket36_single_image_report_json(report) + "\n",
            REPORT_PATH.read_text(encoding="utf-8"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            temp_repo = Path(temporary)
            root = temp_repo / "artifacts/ticket36-repair-candidate"
            root.mkdir(parents=True)
            for role in report["artifact_seal"]["roles"]:
                source = REPO_ROOT / role["path"]
                target = root / Path(role["path"]).name
                shutil.copyfile(source, target)
            seal = dict(report["artifact_seal"])
            verify_ticket36_artifact_seal(seal, repo_root=temp_repo)
            tampered = temp_repo / seal["roles"][0]["path"]
            original = tampered.read_bytes()
            try:
                tampered.write_bytes(original + b"tamper")
                with self.assertRaises(ValueError):
                    verify_ticket36_artifact_seal(seal, repo_root=temp_repo)
            finally:
                tampered.write_bytes(original)

    def test_validator_rejects_fabricated_all_pass_rows_against_sealed_matching(self) -> None:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        fabricated = copy.deepcopy(report)
        for row in fabricated["per_case"]:
            row.update(
                {
                    "instance_recall": 1.0,
                    "unmatched_instance_count": 0,
                    "unmatched_component_count": 0,
                    "merged_component_count": 0,
                    "normal_grid_leak_rate": 0.0,
                    "asserted_grid_occupancy_p95": 0.0,
                    "remote_response": {"numerator": 0, "denominator": 1, "ratio": 0.0},
                    "overall": "PASS",
                }
            )
        fabricated["overall"] = "PASS"
        fabricated["recommendation"] = "spatial_mil_v7"
        with self.assertRaises(ValueError):
            validate_ticket36_single_image_report(fabricated, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
