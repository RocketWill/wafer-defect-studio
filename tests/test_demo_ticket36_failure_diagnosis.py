import hashlib
import json
import unittest
from pathlib import Path

from docs.demo.ticket36_failure_diagnosis import (
    ARTIFACT_INVENTORY_SCHEMA,
    build_ticket36_failure_diagnosis,
    canonical_ticket36_failure_diagnosis_json,
    read_ticket35_density_micro_overfit_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "docs/demo/ticket35-density-micro-overfit.json"
ARTIFACT_PATH = REPO_ROOT / "docs/demo/ticket36-failure-diagnosis.json"


def _inventory() -> dict[str, object]:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    return {
        "schema": ARTIFACT_INVENTORY_SCHEMA,
        "root": "temp/ticket35-density-micro-overfit",
        "source_pngs": [
            {
                "path": f"sources/{row['filename']}",
                "case_id": row["case_id"],
                "status": "present",
            }
            for row in report["per_case"].values()
        ],
        "missing": ["checkpoint", "maps", "per_instance_matching"],
    }


class Ticket36FailureDiagnosisTest(unittest.TestCase):
    def test_builds_blocked_diagnosis_without_inventing_spatial_causes(self) -> None:
        report_text = REPORT_PATH.read_text(encoding="utf-8")
        inventory = _inventory()
        diagnosis = build_ticket36_failure_diagnosis(
            report_text,
            inventory,
        )

        self.assertEqual(diagnosis["schema"], "ticket36-failure-diagnosis.v1")
        self.assertEqual(
            diagnosis["sources"]["ticket35_density_micro_overfit_report"]["sha256"],
            hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(len(diagnosis["per_case"]), 4)
        particle150 = next(
            row for row in diagnosis["per_case"]
            if row["case_id"] == "ticket35-density-101-train-particle-000-150-10"
        )
        self.assertEqual(particle150["covered_defect_instances"], 143)
        self.assertEqual(particle150["missed_defect_instances"], 7)
        self.assertEqual(particle150["instance_recall"], 0.9533333333333334)
        outside = diagnosis["observed_failure"]["outside_declared_extent_response"]
        self.assertEqual(outside["min"], 0.9273584905660377)
        self.assertEqual(outside["max"], 0.9737548147164298)
        self.assertEqual(diagnosis["root_symptom"]["far_activation"], "dominant")
        self.assertEqual(
            diagnosis["root_symptom"]["particle150_recall_gap"],
            {"covered": 143, "total": 150, "missed": 7},
        )

        for name in (
            "component_analysis",
            "miss_location",
            "boundary_classification",
            "merge_classification",
        ):
            analysis = diagnosis["spatial_analysis"][name]
            self.assertEqual(analysis["status"], "unavailable_missing_spatial_artifacts")
            self.assertEqual(
                analysis["missing_artifacts"],
                ["checkpoint", "maps", "per_instance_matching"],
            )

        self.assertEqual(diagnosis["decision"]["overall"], "BLOCKED")
        self.assertEqual(diagnosis["decision"]["recommendation"], "cam_v2")
        self.assertEqual(
            diagnosis["decision"]["next_slice"],
            "freeze_localization_and_artifact_contract",
        )
        self.assertEqual(diagnosis["decision"]["ticket35_06"], "blocked")
        self.assertFalse(diagnosis["boundary"]["final_members_used"])
        self.assertEqual(diagnosis["claims"], [
            "aggregate diagnosis only",
            "not segmentation",
            "not Neurocle equivalence",
            "not production accuracy",
        ])

        self.assertEqual(
            canonical_ticket36_failure_diagnosis_json(diagnosis) + "\n",
            ARTIFACT_PATH.read_text(encoding="utf-8"),
        )

    def test_rejects_tampered_report_and_inventory_drift(self) -> None:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        report["per_case"]["ticket35-density-101-train-particle-000-150-10"][
            "instance_recall"
        ] = 1.0
        tampered = json.dumps(report, sort_keys=True, separators=(",", ":"))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            build_ticket36_failure_diagnosis(tampered, _inventory())

        inventory = _inventory()
        inventory["missing"] = ["checkpoint"]
        with self.assertRaisesRegex(ValueError, "artifact inventory missing set"):
            build_ticket36_failure_diagnosis(
                REPORT_PATH.read_text(encoding="utf-8"), inventory
            )

        for mutate in (
            lambda value: value.update(root="temp/ticket30-evidence-17"),
            lambda value: value["source_pngs"][0].update(
                path="sources/ticket30-evidence-17.png"
            ),
            lambda value: value["source_pngs"][0].update(
                case_id="ticket30-evidence-17.png"
            ),
        ):
            forbidden = _inventory()
            mutate(forbidden)
            with self.assertRaisesRegex(ValueError, "forbidden final token"):
                build_ticket36_failure_diagnosis(
                    REPORT_PATH.read_text(encoding="utf-8"), forbidden
                )

    def test_cli_source_reader_returns_verified_report(self) -> None:
        report = read_ticket35_density_micro_overfit_report(REPORT_PATH)
        self.assertEqual(report["overall"], "FAIL")
        self.assertEqual(report["recommendation"], "cam_v2")


if __name__ == "__main__":
    unittest.main()
