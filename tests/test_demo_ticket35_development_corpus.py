import copy
import json
import unittest
from pathlib import Path

from docs.demo.ticket31_contract import FINAL_MEMBERS
from docs.demo.ticket35_development_corpus import (
    COUNT_LEVELS,
    DEVELOPMENT_SEEDS,
    RESERVED_FINAL_MEMBER_IDS,
    SPLITS,
    build_ticket35_development_corpus,
    build_ticket35_development_metadata,
    canonical_ticket35_development_metadata_json,
    verify_ticket35_development_metadata,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "docs/demo/ticket35-development-corpus.json"


class Ticket35DevelopmentCorpusTest(unittest.TestCase):
    def test_builds_density_diverse_metadata_without_final_pixels(self) -> None:
        corpus = build_ticket35_development_corpus()
        metadata = build_ticket35_development_metadata(corpus)

        self.assertEqual(len(corpus), 120)
        self.assertEqual(tuple(dict.fromkeys(case.seed for case in corpus)), DEVELOPMENT_SEEDS)
        self.assertEqual(tuple(dict.fromkeys(case.split for case in corpus)), SPLITS)
        self.assertEqual(metadata["count_levels"], list(COUNT_LEVELS))
        self.assertEqual(metadata["case_count"], len(corpus))
        self.assertFalse(metadata["pixels_materialized"])
        self.assertEqual(
            metadata["reserved_final_member_ids"], list(RESERVED_FINAL_MEMBER_IDS)
        )
        self.assertEqual(
            metadata["reserved_final_status"], "unopened_no_pixels_materialized"
        )
        self.assertEqual(metadata["promotion_use"], "later_only")
        self.assertEqual(
            metadata["reserved_final"]["reserved_final_member_ids"],
            list(RESERVED_FINAL_MEMBER_IDS),
        )
        self.assertEqual(
            metadata["reserved_final"]["status"],
            "unopened_no_pixels_materialized",
        )
        self.assertEqual(metadata["reserved_final"]["promotion_use"], "later_only")

        instance_ids = {instance.instance_id for case in corpus for instance in case.instances}
        self.assertEqual(len(instance_ids), sum(len(case.instances) for case in corpus))
        train_families = {case.family for case in corpus if case.split == "train"}
        validation_families = {case.family for case in corpus if case.split == "validation"}
        self.assertTrue(train_families.isdisjoint(validation_families))
        train_ids = {
            instance.instance_id
            for case in corpus
            if case.split == "train"
            for instance in case.instances
        }
        validation_ids = {
            instance.instance_id
            for case in corpus
            if case.split == "validation"
            for instance in case.instances
        }
        self.assertTrue(train_ids.isdisjoint(validation_ids))

        for seed in DEVELOPMENT_SEEDS:
            for split in SPLITS:
                cases = tuple(case for case in corpus if case.seed == seed and case.split == split)
                self.assertEqual({case.composition for case in cases}, {"normal", "scratch", "particle", "both"})
                self.assertEqual(
                    {case.scratch_count for case in cases if case.composition == "scratch"},
                    set(COUNT_LEVELS),
                )
                self.assertEqual(
                    {case.particle_count for case in cases if case.composition == "particle"},
                    set(COUNT_LEVELS),
                )

        self.assertGreaterEqual(
            len({case.placement_stratum for case in corpus}), 3
        )
        self.assertTrue({"near_boundary", "cross_grid"}.issubset(
            {case.placement_stratum for case in corpus}
        ))
        self.assertGreaterEqual(
            len({case.scratch_orientation for case in corpus if case.scratch_orientation}), 3
        )
        self.assertGreaterEqual(
            len({case.scratch_length for case in corpus if case.scratch_length}), 3
        )
        self.assertGreaterEqual(
            len({case.particle_radius for case in corpus if case.particle_radius}), 3
        )

        names = {
            value
            for case in corpus
            for value in (case.case_id, case.family, case.filename)
        }
        names.update(instance.instance_id for case in corpus for instance in case.instances)
        for final_member in FINAL_MEMBERS:
            self.assertFalse(any(final_member in value for value in names))
        self.assertTrue(names.isdisjoint(RESERVED_FINAL_MEMBER_IDS))

        self.assertEqual(
            canonical_ticket35_development_metadata_json(metadata) + "\n",
            ARTIFACT_PATH.read_text(encoding="utf-8"),
        )
        verify_ticket35_development_metadata(json.loads(ARTIFACT_PATH.read_text(encoding="utf-8")), corpus)

    def test_metadata_hash_tamper_fails_closed(self) -> None:
        corpus = build_ticket35_development_corpus()
        metadata = copy.deepcopy(build_ticket35_development_metadata(corpus))
        metadata["corpus_sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "corpus SHA-256 mismatch"):
            verify_ticket35_development_metadata(metadata, corpus)


if __name__ == "__main__":
    unittest.main()
