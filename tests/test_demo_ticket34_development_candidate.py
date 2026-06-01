import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from docs.demo.ticket31_development_corpus import build_ticket31_development_corpus
from docs.demo.ticket34_contract import FINAL_MEMBERS
from docs.demo.ticket34_development_candidate import (
    CANONICAL_TRAINING_SEED,
    EPOCHS,
    run_development_candidate,
)
from wafer_defect_studio.grid_geometry import annotation_grids


class _FakeModel:
    def state_dict(self):
        return {"fake.weight": torch.ones(1)}


class Ticket34DevelopmentCandidateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        corpus = build_ticket31_development_corpus()
        cls.cases = tuple(
            next(
                case
                for case in corpus
                if case.seed == seed and case.split == split and case.composition == "both"
            )
            for seed in (101, 211, 307, 401, 503)
            for split in ("train", "validation")
        )

    def _renderer(self, case):
        maps = np.zeros((1536, 1536, 2), dtype=np.float32)
        grids = annotation_grids(1536, 1536, 512, 512)
        for defect in case.oracle.defects:
            class_index = ("scratch", "particle").index(defect.class_code)
            points = (
                defect.points
                if hasattr(defect, "points")
                else (defect.center,)
            )
            for x, y in points:
                maps[y, x, class_index] = 0.9
        return maps

    def test_one_validation_candidate_records_membership_and_integrity(self):
        calls = {"train": [], "validation": []}

        def train(cases, output_root, *, epochs, seed):
            calls["train"] = [case.filename for case in cases]
            self.assertEqual(seed, CANONICAL_TRAINING_SEED)
            self.assertEqual(epochs, EPOCHS)
            return _FakeModel(), [1.0] * 29 + [0.25]

        def score(model, bounds, source):
            del model, bounds
            calls["validation"].append(source)
            return source

        with tempfile.TemporaryDirectory() as temporary:
            report = run_development_candidate(
                Path(temporary),
                corpus=self.cases,
                trainer=train,
                scorer=score,
                renderer=self._renderer,
                environment={
                    "python": "3.11.9",
                    "torch": "2.3.1+cu121",
                    "cuda": "12.1",
                    "gpu": "fixture",
                },
            )

            self.assertEqual(report["overall"], "PASS")
            self.assertEqual(report["training_seed"], CANONICAL_TRAINING_SEED)
            self.assertEqual(report["epochs"], EPOCHS)
            self.assertEqual(report["candidate_count"], 1)
            self.assertEqual(report["candidate_epoch"], EPOCHS)
            self.assertEqual(report["selection_source"], "validation_spatial_metrics")
            self.assertEqual(len(calls["train"]), 5)
            self.assertEqual(len(calls["validation"]), 5)
            self.assertEqual(
                {case.split for case in self.cases}, {"train", "validation"}
            )
            self.assertEqual(
                report["membership"]["train"], calls["train"]
            )
            self.assertEqual(
                report["membership"]["validation"],
                [case.filename for case in self.cases if case.split == "validation"],
            )
            self.assertEqual(report["environment"]["gpu"], "fixture")
            self.assertIn("checkpoint", report["artifacts"])
            checkpoint = torch.load(
                Path(temporary) / report["artifacts"]["checkpoint"]["path"],
                map_location="cpu",
                weights_only=False,
            )
            self.assertIn("state_dict", checkpoint)
            self.assertEqual(
                report["validation_thresholds"]["source_split"], "validation"
            )
            self.assertEqual(report["validation_metrics"]["candidate_epoch"], EPOCHS)
            self.assertEqual(report["validation_metrics"]["candidate_count"], 1)
            self.assertEqual(
                report["validation_metrics"]["selection_source"],
                "validation_spatial_metrics",
            )

    def test_rejects_final_member_before_training_or_rendering(self):
        final_case = dataclasses.replace(
            self.cases[0], filename=FINAL_MEMBERS[0]
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "final evidence member"):
                run_development_candidate(
                    Path(temporary),
                    corpus=(final_case, *self.cases[1:]),
                    trainer=lambda *args, **kwargs: self.fail("training must not run"),
                    renderer=lambda case: self.fail("final renderer must not run"),
                    environment={"python": "fixture", "torch": "fixture", "cuda": "fixture", "gpu": "fixture"},
                )


if __name__ == "__main__":
    unittest.main()
