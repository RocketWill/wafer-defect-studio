import hashlib
import json
import unittest

from docs.demo.ticket30_evidence_corpus import (
    CLASS_CODES,
    FROZEN_CORPUS_SHA256,
    SEEDS,
    build_ticket30_evidence_corpus,
    hash_ticket30_evidence_corpus,
    resolve_ticket30_evidence_corpus,
    serialize_ticket30_evidence_corpus,
)


class Ticket30EvidenceCorpusTest(unittest.TestCase):
    def test_builds_frozen_test_only_oracle_with_150_unique_instances_per_class(self):
        corpus = build_ticket30_evidence_corpus()

        self.assertEqual(tuple(case.seed for case in corpus), SEEDS)
        self.assertTrue(all(case.split == "test" for case in corpus))
        self.assertTrue(all((case.oracle.image_width, case.oracle.image_height) == (1536, 1536) for case in corpus))
        instances = tuple(instance for case in corpus for instance in case.instances)
        self.assertEqual(
            tuple(dict.fromkeys(instance.defect.class_code for instance in instances)),
            CLASS_CODES,
        )
        for case in corpus:
            self.assertEqual(
                {
                    code: sum(
                        instance.defect.class_code == code for instance in case.instances
                    )
                    for code in CLASS_CODES
                },
                {"scratch": 150, "particle": 150},
            )
        self.assertEqual(len({instance.instance_id for instance in instances}), len(instances))
        self.assertEqual(
            tuple(instance.defect for instance in corpus[0].instances),
            corpus[0].oracle.defects,
        )
        for case in corpus:
            for code in CLASS_CODES:
                defects = [
                    instance.defect
                    for instance in case.instances
                    if instance.defect.class_code == code
                ]
                boxes = [
                    (
                        min(point[0] for point in defect.points) - defect.radius,
                        min(point[1] for point in defect.points) - defect.radius,
                        max(point[0] for point in defect.points) + defect.radius,
                        max(point[1] for point in defect.points) + defect.radius,
                    )
                    if hasattr(defect, "points")
                    else (
                        defect.center[0] - defect.radius,
                        defect.center[1] - defect.radius,
                        defect.center[0] + defect.radius,
                        defect.center[1] + defect.radius,
                    )
                    for defect in defects
                ]
                self.assertTrue(all(
                    right_a < left_b or right_b < left_a or bottom_a < top_b or bottom_b < top_a
                    for index, (left_a, top_a, right_a, bottom_a) in enumerate(boxes)
                    for left_b, top_b, right_b, bottom_b in boxes[index + 1 :]
                ))

    def test_canonical_serialization_hash_and_resolver_reject_drift_with_context(self):
        corpus = build_ticket30_evidence_corpus()
        serialized = serialize_ticket30_evidence_corpus(corpus)
        self.assertEqual(serialized, serialize_ticket30_evidence_corpus(build_ticket30_evidence_corpus()))
        self.assertEqual(hash_ticket30_evidence_corpus(corpus), FROZEN_CORPUS_SHA256)
        self.assertEqual(resolve_ticket30_evidence_corpus(serialized, FROZEN_CORPUS_SHA256), corpus)

        with self.assertRaisesRegex(ValueError, "ticket30 evidence corpus SHA-256 mismatch"):
            resolve_ticket30_evidence_corpus(serialized, "0" * 64)

        for drift, mutate, message in (
            ("membership", lambda value: value["cases"][0].update(seed=18), "seed membership/order"),
            ("class", lambda value: value["cases"][0]["instances"][0]["defect"].update(class_code="crack"), "class order/membership"),
            ("count", lambda value: value["cases"][0]["instances"].pop(), "instance count"),
        ):
            payload = json.loads(serialized)
            mutate(payload)
            tampered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            tampered_hash = hashlib.sha256(tampered.encode("utf-8")).hexdigest()
            with self.subTest(drift=drift), self.assertRaisesRegex(ValueError, message):
                resolve_ticket30_evidence_corpus(tampered, tampered_hash)


if __name__ == "__main__":
    unittest.main()
