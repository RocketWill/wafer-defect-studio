import copy
import json
import unittest
from pathlib import Path

from docs.demo.ticket35_density_diagnosis import (
    build_ticket35_density_diagnosis,
    canonical_density_diagnosis_json,
    read_ticket31_development_metadata,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "docs/demo/ticket34-final-gate.json"
CORPUS_SOURCE_PATH = REPO_ROOT / "docs/demo/ticket31_development_corpus.py"
ARTIFACT_PATH = REPO_ROOT / "docs/demo/ticket35-density-diagnosis.json"


class Ticket35DensityDiagnosisTest(unittest.TestCase):
    def test_freezes_literal_final_evidence_and_density_hypothesis(self) -> None:
        report_text = REPORT_PATH.read_text(encoding="utf-8")
        metadata = read_ticket31_development_metadata(CORPUS_SOURCE_PATH)
        diagnosis = build_ticket35_density_diagnosis(report_text, metadata)

        self.assertEqual(diagnosis["schema"], "ticket35-density-diagnosis.v1")
        self.assertEqual(diagnosis["density_shift"]["ratio_final_to_development"], 75)
        self.assertEqual(
            diagnosis["final_boundary"]["status"],
            "diagnostic_only_consumed_final",
        )
        self.assertEqual(
            diagnosis["hypothesis"]["density_composition_shift"]["status"],
            "primary_hypothesis",
        )
        self.assertFalse(
            diagnosis["hypothesis"]["density_composition_shift"]["causal_proof"]
        )

        for seed in (17, 42, 91):
            scratch = diagnosis["per_seed"][str(seed)]["per_class"]["scratch"]
            particle = diagnosis["per_seed"][str(seed)]["per_class"]["particle"]
            self.assertEqual(scratch["defect_instances"], 150)
            self.assertEqual(scratch["covered_defect_instances"], 135)
            self.assertEqual(scratch["defect_coverage_recall"], 0.9)
            self.assertEqual(scratch["asserted_grid_occupancy_p95"] > 0.53, True)
            self.assertEqual(scratch["grid_precision"], 1.0)
            self.assertEqual(scratch["normal_grid_leak_rate"], 0.0)
            self.assertGreater(scratch["score_separation_margin"], 0.0)
            self.assertEqual(scratch["diagnostic"]["missed_defect_instances"], 15)
            self.assertTrue(scratch["diagnostic"]["extent_failure"])

            self.assertEqual(particle["defect_instances"], 150)
            self.assertEqual(particle["covered_defect_instances"], 150)
            self.assertEqual(particle["defect_coverage_recall"], 1.0)
            self.assertEqual(particle["grid_precision"], 0.5)
            self.assertEqual(particle["normal_grid_leak_rate"], 0.125)
            self.assertGreater(particle["score_separation_margin"], 0.0)
            self.assertEqual(particle["diagnostic"]["grid_false_positives"], 1)
            self.assertEqual(particle["diagnostic"]["normal_grid_leaks"], 1)
            self.assertTrue(particle["diagnostic"]["extent_failure"])

        self.assertEqual(
            canonical_density_diagnosis_json(diagnosis) + "\n",
            ARTIFACT_PATH.read_text(encoding="utf-8"),
        )

    def test_tampered_ticket34_report_fails_before_diagnosis(self) -> None:
        payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        payload = copy.deepcopy(payload)
        payload["per_seed"]["17"]["per_class"]["scratch"][
            "defect_coverage_recall"
        ] = 1.0
        tampered = json.dumps(payload, sort_keys=True, separators=(",", ":"))

        with self.assertRaisesRegex(ValueError, "Ticket 34 final report SHA-256 mismatch"):
            build_ticket35_density_diagnosis(
                tampered,
                read_ticket31_development_metadata(CORPUS_SOURCE_PATH),
            )


if __name__ == "__main__":
    unittest.main()
