import json
import copy
import tempfile
import unittest
from pathlib import Path

from docs.demo.ticket37_existing_artifact_report import (
    FROZEN_ARTIFACT_SEAL,
    normalize_ticket37_configuration_key_order,
    validate_ticket37_frozen_artifact_seal,
    validate_ticket37_matching_against_recomputed,
    write_ticket37_recovered_report,
)


class Ticket37ExistingArtifactReportTest(unittest.TestCase):
    def test_matching_validation_ignores_canonical_object_key_order(self) -> None:
        case_order = ("scratch-case", "particle-case")
        recomputed = {
            "schema": "matching.v1",
            "case_order": list(case_order),
            "per_case": {
                "scratch-case": {"scratch": {"proposal_precision": 1.0}},
                "particle-case": {"particle": {"proposal_precision": 0.5}},
            },
        }
        stored = json.loads(json.dumps(recomputed, sort_keys=True))

        validate_ticket37_matching_against_recomputed(stored, recomputed, case_order=case_order)

        stored["per_case"]["particle-case"]["particle"]["proposal_precision"] = 1.0
        with self.assertRaises(ValueError):
            validate_ticket37_matching_against_recomputed(stored, recomputed, case_order=case_order)

    def test_recovered_report_write_is_exclusive_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            report = {"schema": "report.v1", "overall": "FAIL"}

            write_ticket37_recovered_report(path, report)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), report)
            with self.assertRaises(FileExistsError):
                write_ticket37_recovered_report(path, report)

    def test_configuration_normalization_only_restores_case_key_order(self) -> None:
        value = {
            "loss_history": {"particle": [2.0], "scratch": [1.0]},
            "unchanged": {"z": 1},
        }

        normalized = normalize_ticket37_configuration_key_order(
            value,
            case_order=("scratch", "particle"),
        )

        self.assertEqual(tuple(normalized["loss_history"]), ("scratch", "particle"))
        self.assertEqual(normalized, value)
        self.assertIsNot(normalized, value)

    def test_frozen_artifact_receipt_rejects_any_role_drift(self) -> None:
        validate_ticket37_frozen_artifact_seal(copy.deepcopy(FROZEN_ARTIFACT_SEAL))
        for field, replacement in (("bytes", 1), ("sha256", "0" * 64)):
            changed = copy.deepcopy(FROZEN_ARTIFACT_SEAL)
            changed["roles"][1][field] = replacement
            with self.assertRaises(ValueError):
                validate_ticket37_frozen_artifact_seal(changed)


if __name__ == "__main__":
    unittest.main()
