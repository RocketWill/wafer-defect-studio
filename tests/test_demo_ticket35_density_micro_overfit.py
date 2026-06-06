import json
import unittest
from pathlib import Path

import torch

from docs.demo.ticket35_density_micro_overfit import (
    CLASS_CODES,
    FEATURE_STRIDE,
    FIXED_THRESHOLD,
    MICRO_CASE_COUNTS,
    SCHEMA,
    build_feature_instance_supervision,
    build_ticket35_density_micro_cases,
    canonical_ticket35_density_micro_overfit_json,
    validate_ticket35_density_micro_overfit_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "docs/demo/ticket35-density-micro-overfit.json"


class Ticket35DensityMicroOverfitTest(unittest.TestCase):
    def test_freezes_four_density_extremes_and_sparse_adapter_contract(self) -> None:
        cases = build_ticket35_density_micro_cases()
        self.assertEqual(len(cases), 4)
        self.assertEqual(
            tuple((case.composition, case.scratch_count, case.particle_count) for case in cases),
            MICRO_CASE_COUNTS,
        )
        self.assertTrue(all(case.seed == 101 and case.split == "train" for case in cases))
        names = {
            value
            for case in cases
            for value in (case.case_id, case.filename, *(instance.instance_id for instance in case.instances))
        }
        self.assertTrue(all("ticket30-evidence-" not in value for value in names))
        self.assertTrue(all("ticket35-final-member-" not in value for value in names))

        case = cases[0]
        defect = case.instances[0].defect
        point = defect.center if hasattr(defect, "center") else defect.points[0]
        rect = (max(0, min(case.oracle.image_width - 128, point[0] - 64)),
                max(0, min(case.oracle.image_height - 128, point[1] - 64)), 128, 128)
        supervision = build_feature_instance_supervision(
            case,
            (rect,),
            feature_stride=FEATURE_STRIDE,
        )
        self.assertEqual(supervision.masks.ndim, 3)
        self.assertEqual(supervision.masks.shape[-2:], (64, 64))
        self.assertEqual(supervision.batch_indices.shape, (supervision.masks.shape[0],))
        self.assertEqual(supervision.class_indices.shape, (supervision.masks.shape[0],))
        self.assertEqual(len(supervision.instance_ids), supervision.masks.shape[0])
        self.assertTrue(bool(supervision.masks.any().item()))
        self.assertTrue(torch.equal(supervision.batch_indices, torch.zeros_like(supervision.batch_indices)))

    def test_tracked_report_is_canonical_and_meets_frozen_gate(self) -> None:
        report = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["seed"], 101)
        self.assertEqual(report["source_split"], "train")
        self.assertEqual(report["threshold"], FIXED_THRESHOLD)
        self.assertEqual(report["calibration"], "forbidden")
        self.assertEqual(set(report["class_codes"]), set(CLASS_CODES))
        self.assertEqual(len(report["per_case"]), 4)
        self.assertEqual(report["boundary"]["ticket34_final_use"], "forbidden")
        self.assertEqual(report["boundary"]["ticket35_reserved_final_use"], "forbidden")
        self.assertEqual(report["boundary"]["final_pixels_materialized"], False)
        self.assertEqual(
            tuple(report["boundary"]["ticket34_final_members"]),
            ("ticket30-evidence-17.png", "ticket30-evidence-42.png", "ticket30-evidence-91.png"),
        )
        self.assertEqual(
            tuple(report["boundary"]["ticket35_reserved_final_members"]),
            (
                "ticket35-final-member-01",
                "ticket35-final-member-02",
                "ticket35-final-member-03",
            ),
        )
        self.assertEqual(report["overall"], "FAIL")
        rows = report["per_case"]
        self.assertTrue(all(row["overall"] == "FAIL" for row in rows.values()))
        self.assertLess(
            rows["ticket35-density-101-train-particle-000-150-10"]["instance_recall"],
            1.0,
        )
        self.assertGreater(
            rows["ticket35-density-101-train-scratch-001-000-01"][
                "outside_declared_extent_response"
            ],
            0.0,
        )
        self.assertGreater(
            rows["ticket35-density-101-train-scratch-150-000-05"][
                "outside_declared_extent_response"
            ],
            0.0,
        )
        validate_ticket35_density_micro_overfit_report(report)
        self.assertEqual(
            canonical_ticket35_density_micro_overfit_json(report) + "\n",
            ARTIFACT_PATH.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
