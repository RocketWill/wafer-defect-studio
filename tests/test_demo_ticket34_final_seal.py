import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from docs.demo.ticket34_contract import FINAL_MEMBERS
from docs.demo.ticket34_final_seal import (
    EXPECTED_DEVELOPMENT_MEMBERS,
    FINAL_GATE_CLASS_CODES,
    build_ticket34_final_contract,
    canonical_final_contract_json,
    run_final_gate,
    validate_ticket34_final_seal,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SEAL_PATH = REPO_ROOT / "docs/demo/ticket34-final-seal.json"


class _Case:
    def __init__(self, seed, filename, oracle):
        self.seed = seed
        self.filename = filename
        self.split = "test"
        self.oracle = oracle


class _Oracle:
    image_width = 4
    image_height = 4
    defects = ()

    def grid_truth(self, grids):
        return {(grid.row, grid.column): () for grid in grids}


class Ticket34FinalSealTest(unittest.TestCase):
    def test_actual_contract_has_exact_120_member_development_boundary(self):
        contract = build_ticket34_final_contract()
        self.assertEqual(contract["development_members"], EXPECTED_DEVELOPMENT_MEMBERS)
        self.assertEqual(len(contract["development_members"]["train"]), 120)
        self.assertEqual(len(contract["development_members"]["validation"]), 120)
        self.assertEqual(
            canonical_final_contract_json(contract),
            json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )

    def test_tampered_seal_fails_before_provider(self):
        seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(seal)
        tampered["files"][0]["sha256"] = "0" * 64
        calls = []
        with self.assertRaisesRegex(ValueError, "seal manifest hash mismatch"):
            run_final_gate(
                tampered,
                seal_root=REPO_ROOT,
                output_root=Path(tempfile.mkdtemp()),
                corpus_provider=lambda: calls.append("provider"),
                model_loader=lambda path: object(),
                scorer=lambda model, bounds, source: np.zeros((4, 4, 2), dtype=np.float32),
                renderer=lambda case: np.zeros((4, 4), dtype=np.uint8),
            )
        self.assertEqual(calls, [])

    def test_failed_quality_is_reported_truthfully_without_final_calibration(self):
        seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
        cases = tuple(
            _Case(seed, filename, _Oracle())
            for seed, filename in zip((17, 42, 91), FINAL_MEMBERS, strict=True)
        )
        with tempfile.TemporaryDirectory() as temporary:
            report = run_final_gate(
                seal,
                seal_root=REPO_ROOT,
                output_root=Path(temporary),
                corpus_provider=lambda: cases,
                model_loader=lambda path: object(),
                scorer=lambda model, bounds, source: np.zeros((4, 4, 2), dtype=np.float32),
                renderer=lambda case: np.zeros((4, 4), dtype=np.uint8),
            )
        self.assertEqual(report["overall"], "FAIL")
        self.assertEqual(tuple(report["per_seed"]), ("17", "42", "91"))
        self.assertEqual(
            tuple(report["per_seed"]["17"]["per_class"]), FINAL_GATE_CLASS_CODES
        )
        self.assertEqual(report["map_artifacts"]["17"]["seed"], 17)
        self.assertEqual(report["map_artifacts"]["42"]["seed"], 42)
        self.assertEqual(report["map_artifacts"]["91"]["seed"], 91)


if __name__ == "__main__":
    unittest.main()
