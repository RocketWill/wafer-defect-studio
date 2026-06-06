import hashlib
import json
import unittest
from dataclasses import replace

from docs.demo.ticket35_sparse_localization import (
    BUNDLE_SCHEMA,
    CORPUS_SHA256,
    MANIFEST_PATH,
    build_ticket35_sparse_localization_bundle,
    canonical_ticket35_sparse_localization_json,
    hash_ticket35_sparse_localization_bundle,
    resolve_ticket35_sparse_localization_bundle,
    validate_ticket35_sparse_localization_bundle,
    verify_ticket35_sparse_localization_manifest,
)


class Ticket35SparseLocalizationTest(unittest.TestCase):
    def test_bundle_has_one_truth_row_per_instance_and_manifest_is_reproducible(self) -> None:
        bundle = build_ticket35_sparse_localization_bundle()
        validate_ticket35_sparse_localization_bundle(bundle)

        self.assertEqual(bundle.schema, BUNDLE_SCHEMA)
        self.assertEqual(bundle.corpus_sha256, CORPUS_SHA256)
        self.assertEqual(len(bundle.instances), 4020)
        self.assertEqual(
            tuple(row.instance_id for row in bundle.instances),
            tuple(dict.fromkeys(row.instance_id for row in bundle.instances)),
        )
        self.assertEqual(
            {row.kind for row in bundle.instances},
            {"point", "scribble"},
        )
        self.assertEqual(
            {row.provenance for row in bundle.instances},
            {"synthetic_defect_oracle"},
        )

        encoded = canonical_ticket35_sparse_localization_json(bundle)
        payload = json.loads(encoded)
        self.assertNotIn("mask", encoded)
        self.assertNotIn("pixel_mask", encoded)
        self.assertNotIn("filled_region", encoded)
        self.assertEqual(hash_ticket35_sparse_localization_bundle(bundle), hash_ticket35_sparse_localization_bundle(bundle))
        verify_ticket35_sparse_localization_manifest(
            json.loads(MANIFEST_PATH.read_text(encoding="utf-8")), bundle
        )
        self.assertEqual(len(payload["instances"][0]["points"]), 2)

    def test_malformed_or_extra_truth_fails_closed_with_instance_context(self) -> None:
        bundle = build_ticket35_sparse_localization_bundle()

        duplicate = replace(bundle, instances=bundle.instances + (bundle.instances[0],))
        with self.assertRaisesRegex(ValueError, "duplicate instance_id.*ticket35-density"):
            validate_ticket35_sparse_localization_bundle(duplicate)

        missing = replace(bundle, instances=bundle.instances[:-1])
        with self.assertRaisesRegex(ValueError, "instance coverage.*expected=4020 actual=4019"):
            validate_ticket35_sparse_localization_bundle(missing)

        tampered = replace(bundle)
        row = bundle.instances[0]
        tampered = replace(tampered, instances=(
            type(row)(
                row.instance_id,
                row.case_id,
                row.filename,
                row.split,
                "particle",
                "point",
                row.points,
                row.provenance,
            ),
        ) + bundle.instances[1:])
        with self.assertRaisesRegex(ValueError, "class/kind mismatch.*ticket35-density"):
            validate_ticket35_sparse_localization_bundle(tampered)

        payload = json.loads(canonical_ticket35_sparse_localization_json(bundle))
        payload["instances"][0]["mask"] = []
        malformed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self.assertRaisesRegex(ValueError, "row fields drift.*row_index=0"):
            resolve_ticket35_sparse_localization_bundle(
                malformed,
                hashlib.sha256(malformed.encode("utf-8")).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
