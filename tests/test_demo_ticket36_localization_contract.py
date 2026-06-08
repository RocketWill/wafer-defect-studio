import copy
import json
import tempfile
import unittest
from pathlib import Path

from docs.demo.ticket36_localization_contract import (
    ARTIFACT_ROLES,
    CANDIDATE_ARTIFACT_ROOT,
    CLASS_ORDER,
    FEATURE_STRIDE,
    SCHEMA,
    SPARSE_BUNDLE_SHA256,
    build_ticket36_localization_contract,
    canonical_ticket36_localization_contract_json,
    main,
    validate_ticket36_candidate_artifact_seal,
    validate_ticket36_localization_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "docs/demo/ticket36-localization-contract.json"


class Ticket36LocalizationContractTest(unittest.TestCase):
    def test_builder_freezes_declarative_localization_contract(self) -> None:
        contract = build_ticket36_localization_contract()

        self.assertEqual(contract["schema"], SCHEMA)
        self.assertEqual(
            contract["source_geometry"],
            {
                "coordinate_system": "source_pixels",
                "width": 1536,
                "height": 1536,
            },
        )
        self.assertEqual(contract["class_order"], list(CLASS_ORDER))
        self.assertEqual(contract["feature_stride"], FEATURE_STRIDE)
        self.assertEqual(contract["threshold"], 0.5)
        self.assertEqual(contract["calibration"], "forbidden")
        self.assertEqual(
            contract["sparse_truth"],
            {
                "coordinate_system": "source_pixels",
                "claim": "weak_sparse_tolerance_not_mask_or_segmentation_truth",
                "generator": {
                    "hash_algorithm": "sha256",
                    "normalization": "utf8_lf_bytes",
                    "path": "docs/demo/ticket35_sparse_localization.py",
                    "sha256": "cf5c200b11012ed47822590adc51ac5c41fd680cdc3e1e93125cdb4433f9a3fc",
                },
                "materialization": "regenerate_from_frozen_deterministic_generator",
                "provenance": "synthetic_defect_oracle",
                "authorized_consumer": "ticket36.03_only",
                "ticket35_manifest": {
                    "bundle_sha256": SPARSE_BUNDLE_SHA256,
                    "declared_consumer": "ticket35.04_only",
                    "file_sha256": "b6f12aac9c09b06732574c99aa7ae11ec8f03480da0ccb594f76d227c54dc646",
                    "hash_algorithm": "sha256",
                    "normalization": "utf8_lf_bytes",
                    "path": "docs/demo/ticket35-sparse-localization-manifest.json",
                    "role": "provenance_only_not_input",
                    "schema": "ticket35-sparse-localization-manifest.v1",
                },
                "truth_form": "sparse_point_scribble",
            },
        )
        geometry = contract["geometry"]
        self.assertEqual(
            geometry["particle"],
            {
                "core": "annotated_point",
                "tolerance": {"kind": "radius_source_pixels", "value": 8},
            },
        )
        self.assertEqual(
            geometry["scratch"],
            {
                "core": "complete_rasterized_scribble_segments",
                "tolerance": {"kind": "half_width_source_pixels", "value": 8},
            },
        )
        self.assertEqual(
            geometry["far_negative"],
            {
                "scope": "same_class",
                "definition": "same_class_tolerance_union_complement",
                "valid_range": "current_source_or_patch_valid_extent",
                "normal_grid": "all_valid_extent",
                "same_class_core": "never_negative",
            },
        )
        self.assertEqual(contract["candidate_artifacts"]["root"], CANDIDATE_ARTIFACT_ROOT)
        self.assertEqual(
            contract["candidate_artifacts"]["hash_policy"],
            "required_after_generation_sha256",
        )
        roles = contract["candidate_artifacts"]["roles"]
        self.assertEqual(tuple(role["role"] for role in roles), ARTIFACT_ROLES)
        self.assertTrue(all(role["bytes"] is None for role in roles))
        self.assertTrue(all(role["sha256"] is None for role in roles))
        validate_ticket36_localization_contract(contract)

        self.assertEqual(
            canonical_ticket36_localization_contract_json(contract) + "\n",
            ARTIFACT_PATH.read_text(encoding="utf-8"),
        )

    def test_rejects_geometry_path_and_hash_policy_tampering(self) -> None:
        for mutate, message in (
            (
                lambda value: value["geometry"]["particle"]["tolerance"].update(value=7),
                "particle tolerance",
            ),
            (
                lambda value: value["candidate_artifacts"]["roles"][0].update(
                    path="artifacts/ticket36-repair-candidate/../escape.pt"
                ),
                "path",
            ),
            (
                lambda value: value["candidate_artifacts"].update(
                    hash_policy="optional"
                ),
                "hash policy",
            ),
        ):
            tampered = copy.deepcopy(build_ticket36_localization_contract())
            mutate(tampered)
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_ticket36_localization_contract(tampered)

    def test_rejects_tampered_sparse_sources_without_editing_repository_files(self) -> None:
        contract = build_ticket36_localization_contract()
        generator = Path(REPO_ROOT / "docs/demo/ticket35_sparse_localization.py").read_text(
            encoding="utf-8"
        )
        manifest = Path(
            REPO_ROOT / "docs/demo/ticket35-sparse-localization-manifest.json"
        ).read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "generator SHA-256 mismatch"):
            validate_ticket36_localization_contract(
                contract,
                generator_text=generator + "\n# tampered",
                manifest_text=manifest,
            )
        with self.assertRaisesRegex(ValueError, "manifest file SHA-256 mismatch"):
            validate_ticket36_localization_contract(
                contract,
                generator_text=generator,
                manifest_text=manifest.replace(
                    '"declared_consumer":"ticket35.04_only"',
                    '"declared_consumer":"ticket36.03_only"',
                ),
            )

    def test_rejects_forbidden_absolute_duplicate_and_unsealed_hash_drift(self) -> None:
        base = build_ticket36_localization_contract()
        for mutate, message in (
            (
                lambda value: value["candidate_artifacts"]["roles"][0].update(
                    path="artifacts/ticket30-evidence-17/checkpoint.pt"
                ),
                "forbidden",
            ),
            (
                lambda value: value["candidate_artifacts"]["roles"][0].update(
                    path="C:/outside/checkpoint.pt"
                ),
                "absolute",
            ),
            (
                lambda value: value["candidate_artifacts"]["roles"][1].update(
                    role=value["candidate_artifacts"]["roles"][0]["role"]
                ),
                "unique",
            ),
            (
                lambda value: value["candidate_artifacts"]["roles"][0].update(
                    sha256="ABC"
                ),
                "sha256",
            ),
        ):
            tampered = copy.deepcopy(base)
            mutate(tampered)
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_ticket36_localization_contract(tampered)

    def test_cli_writes_canonical_contract_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "contract.json"
            self.assertEqual(main(["--output", str(output)]), 0)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                ARTIFACT_PATH.read_text(encoding="utf-8"),
            )
            self.assertEqual(main(["--output", str(output)]), 1)

    def test_artifact_seal_requires_positive_bytes_and_lowercase_sha256(self) -> None:
        seal = copy.deepcopy(build_ticket36_localization_contract()["candidate_artifacts"])
        for role in seal["roles"]:
            role["bytes"] = 1
            role["sha256"] = "0" * 64
        validate_ticket36_candidate_artifact_seal(seal)

        seal["roles"][0]["sha256"] = "A" * 64
        with self.assertRaisesRegex(ValueError, "sha256"):
            validate_ticket36_candidate_artifact_seal(seal)


if __name__ == "__main__":
    unittest.main()
