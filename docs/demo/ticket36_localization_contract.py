"""Declarative localization and future-candidate artifact contract for Ticket 36."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


SCHEMA = "ticket36-localization-contract.v1"
SOURCE_WIDTH = 1536
SOURCE_HEIGHT = 1536
CLASS_ORDER = ("scratch", "particle")
FEATURE_STRIDE = 2
THRESHOLD = 0.5
CALIBRATION = "forbidden"
SPARSE_TRUTH_PROVENANCE = "synthetic_defect_oracle"
SPARSE_TRUTH_FORM = "sparse_point_scribble"
SPARSE_GENERATOR_PATH = "docs/demo/ticket35_sparse_localization.py"
SPARSE_GENERATOR_SHA256 = (
    "cf5c200b11012ed47822590adc51ac5c41fd680cdc3e1e93125cdb4433f9a3fc"
)
SPARSE_MANIFEST_PATH = "docs/demo/ticket35-sparse-localization-manifest.json"
SPARSE_MANIFEST_FILE_SHA256 = (
    "b6f12aac9c09b06732574c99aa7ae11ec8f03480da0ccb594f76d227c54dc646"
)
SPARSE_BUNDLE_SHA256 = (
    "3ab3dfb1f4c6d4d38e8cf18d80ce0a3fbf1ef5788c057944ad0328c5a361076f"
)
PARTICLE_TOLERANCE_RADIUS = 8
SCRATCH_TOLERANCE_HALF_WIDTH = 8
CANDIDATE_ARTIFACT_ROOT = "artifacts/ticket36-repair-candidate"
ARTIFACT_ROLES = (
    "checkpoint",
    "confidence_maps",
    "per_instance_matching",
    "configuration",
)
ARTIFACT_FILENAMES = {
    "checkpoint": "checkpoint.pt",
    "confidence_maps": "confidence-maps.npz",
    "per_instance_matching": "per-instance-matching.json",
    "configuration": "configuration.json",
}
HASH_POLICY = "required_after_generation_sha256"
FORBIDDEN_PATH_TOKENS = (
    "ticket34-final",
    "ticket30-evidence",
    "ticket35-final-member",
    "ticket35_reserved",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def build_ticket36_localization_contract() -> dict[str, object]:
    """Build the frozen declarative contract without masks or model execution."""

    return {
        "schema": SCHEMA,
        "source_geometry": {
            "coordinate_system": "source_pixels",
            "width": SOURCE_WIDTH,
            "height": SOURCE_HEIGHT,
        },
        "class_order": list(CLASS_ORDER),
        "feature_stride": FEATURE_STRIDE,
        "threshold": THRESHOLD,
        "calibration": CALIBRATION,
        "sparse_truth": {
            "coordinate_system": "source_pixels",
            "provenance": SPARSE_TRUTH_PROVENANCE,
            "truth_form": SPARSE_TRUTH_FORM,
            "claim": "weak_sparse_tolerance_not_mask_or_segmentation_truth",
            "materialization": "regenerate_from_frozen_deterministic_generator",
            "authorized_consumer": "ticket36.03_only",
            "generator": {
                "path": SPARSE_GENERATOR_PATH,
                "sha256": SPARSE_GENERATOR_SHA256,
                "hash_algorithm": "sha256",
                "normalization": "utf8_lf_bytes",
            },
            "ticket35_manifest": {
                "path": SPARSE_MANIFEST_PATH,
                "file_sha256": SPARSE_MANIFEST_FILE_SHA256,
                "bundle_sha256": SPARSE_BUNDLE_SHA256,
                "hash_algorithm": "sha256",
                "normalization": "utf8_lf_bytes",
                "schema": "ticket35-sparse-localization-manifest.v1",
                "declared_consumer": "ticket35.04_only",
                "role": "provenance_only_not_input",
            },
        },
        "geometry": {
            "particle": {
                "core": "annotated_point",
                "tolerance": {
                    "kind": "radius_source_pixels",
                    "value": PARTICLE_TOLERANCE_RADIUS,
                },
            },
            "scratch": {
                "core": "complete_rasterized_scribble_segments",
                "tolerance": {
                    "kind": "half_width_source_pixels",
                    "value": SCRATCH_TOLERANCE_HALF_WIDTH,
                },
            },
            "far_negative": {
                "scope": "same_class",
                "definition": "same_class_tolerance_union_complement",
                "valid_range": "current_source_or_patch_valid_extent",
                "normal_grid": "all_valid_extent",
                "same_class_core": "never_negative",
            },
        },
        "candidate_artifacts": {
            "root": CANDIDATE_ARTIFACT_ROOT,
            "hash_policy": HASH_POLICY,
            "seal_status": "future_artifact_seal_required",
            "roles": [
                {
                    "role": role,
                    "path": f"{CANDIDATE_ARTIFACT_ROOT}/{ARTIFACT_FILENAMES[role]}",
                    "bytes": None,
                    "sha256": None,
                }
                for role in ARTIFACT_ROLES
            ],
        },
        "execution": {
            "pixel_masks": "forbidden",
            "training": "forbidden",
            "inference": "forbidden",
        },
        "boundary": {
            "cam_default": "cam_v2",
            "spatial_mil_v7_status": "experimental",
            "ticket35_06": "blocked",
            "ticket34_final_use": "forbidden",
            "ticket30_evidence_use": "forbidden",
            "ticket35_reserved_use": "forbidden",
        },
        "claims": [
            "weak sparse tolerance, not a mask",
            "not segmentation truth",
            "not Neurocle equivalence",
            "not production accuracy",
        ],
    }


def validate_ticket36_localization_contract(
    contract: Mapping[str, object],
    *,
    repo_root: Path | None = None,
    generator_text: str | None = None,
    manifest_text: str | None = None,
) -> None:
    """Fail closed on contract drift, path traversal, or premature artifact hashes."""

    if not isinstance(contract, Mapping):
        raise ValueError("Ticket 36 localization contract must be an object")
    expected = build_ticket36_localization_contract()
    if contract.get("schema") != SCHEMA:
        raise ValueError("Ticket 36 localization contract schema drift")
    if set(contract) != set(expected):
        raise ValueError("Ticket 36 localization contract fields drift")
    _require_equal(contract, "source_geometry", expected["source_geometry"], "source geometry")
    _require_equal(contract, "class_order", list(CLASS_ORDER), "class order")
    if contract.get("feature_stride") != FEATURE_STRIDE:
        raise ValueError("feature stride drift")
    if contract.get("threshold") != THRESHOLD:
        raise ValueError("threshold drift")
    if contract.get("calibration") != CALIBRATION:
        raise ValueError("calibration policy drift")
    _require_equal(contract, "sparse_truth", expected["sparse_truth"], "sparse truth")
    _validate_sparse_truth_sources(
        repo_root=repo_root,
        generator_text=generator_text,
        manifest_text=manifest_text,
    )
    geometry = contract.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("geometry must be an object")
    expected_geometry = expected["geometry"]
    if geometry != expected_geometry:
        if (
            isinstance(geometry.get("particle"), Mapping)
            and geometry["particle"].get("tolerance")
            != expected_geometry["particle"]["tolerance"]
        ):
            raise ValueError("particle tolerance drift")
        raise ValueError("geometry drift")
    _validate_candidate_artifacts(contract.get("candidate_artifacts"), expected["candidate_artifacts"])
    _require_equal(contract, "execution", expected["execution"], "execution boundary")
    _require_equal(contract, "boundary", expected["boundary"], "decision boundary")
    _require_equal(contract, "claims", expected["claims"], "claim boundary")


def validate_ticket36_candidate_artifact_seal(
    seal: Mapping[str, object],
) -> None:
    """Validate generated artifact bytes/hashes against this future contract.

    This seam only validates metadata supplied after generation.  It does not
    read, create, or infer any candidate artifact.
    """

    if not isinstance(seal, Mapping):
        raise ValueError("Ticket 36 artifact seal must be an object")
    expected = build_ticket36_localization_contract()["candidate_artifacts"]
    if set(seal) not in (
        {"root", "hash_policy", "roles"},
        {"root", "hash_policy", "seal_status", "roles"},
    ):
        raise ValueError("artifact seal fields drift")
    if seal.get("root") != expected["root"]:
        raise ValueError("artifact seal root drift")
    if seal.get("hash_policy") != HASH_POLICY:
        raise ValueError("artifact seal hash policy drift")
    if "seal_status" in seal and seal.get("seal_status") != "future_artifact_seal_required":
        raise ValueError("artifact seal status drift")
    _validate_roles(seal.get("roles"), require_sealed=True)


def canonical_ticket36_localization_contract_json(
    contract: Mapping[str, object],
) -> str:
    """Return stable canonical JSON after validating the declarative contract."""

    validate_ticket36_localization_contract(contract)
    return json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_contract_json(contract: Mapping[str, object]) -> str:
    """Compatibility alias matching earlier Ticket contract modules."""

    return canonical_ticket36_localization_contract_json(contract)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        if output.exists():
            raise ValueError(f"Ticket 36 localization contract output already exists: {output}")
        contract = build_ticket36_localization_contract()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            canonical_ticket36_localization_contract_json(contract) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"Ticket 36 localization contract failed: {error}", file=sys.stderr)
        return 1
    print(f"Ticket 36 localization contract: {output}", flush=True)
    return 0


def _validate_sparse_truth_sources(
    *,
    repo_root: Path | None,
    generator_text: str | None,
    manifest_text: str | None,
) -> None:
    root = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root)
    generator_path = root / SPARSE_GENERATOR_PATH
    manifest_path = root / SPARSE_MANIFEST_PATH
    if generator_text is None:
        generator_bytes = _read_normalized_file_bytes(
            generator_path, "sparse localization generator"
        )
    else:
        generator_bytes = _normalized_utf8_lf_bytes(generator_text, "sparse localization generator")
    if _sha256_bytes(generator_bytes) != SPARSE_GENERATOR_SHA256:
        raise ValueError("sparse localization generator SHA-256 mismatch")

    if manifest_text is None:
        manifest_bytes = _read_normalized_file_bytes(
            manifest_path, "Ticket 35 sparse manifest"
        )
    else:
        manifest_bytes = _normalized_utf8_lf_bytes(manifest_text, "Ticket 35 sparse manifest")
    if _sha256_bytes(manifest_bytes) != SPARSE_MANIFEST_FILE_SHA256:
        raise ValueError("Ticket 35 sparse manifest file SHA-256 mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Ticket 35 sparse manifest: {error}") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("Ticket 35 sparse manifest must be an object")
    if manifest.get("schema") != "ticket35-sparse-localization-manifest.v1":
        raise ValueError("Ticket 35 sparse manifest schema drift")
    if manifest.get("bundle_sha256") != SPARSE_BUNDLE_SHA256:
        raise ValueError("Ticket 35 sparse manifest bundle SHA-256 mismatch")
    if manifest.get("declared_consumer") != "ticket35.04_only":
        raise ValueError("Ticket 35 sparse manifest declared consumer drift")


def _normalized_utf8_lf_bytes(value: str, label: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be UTF-8 text")
    return value.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _read_normalized_file_bytes(path: Path, label: str) -> bytes:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}: {error}") from error
    return _normalized_utf8_lf_bytes(text, label)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_candidate_artifacts(
    value: object,
    expected: Mapping[str, object],
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("candidate artifacts must be an object")
    if set(value) != set(expected):
        raise ValueError("candidate artifact contract fields drift")
    if value.get("root") != CANDIDATE_ARTIFACT_ROOT:
        raise ValueError("candidate artifact root drift")
    if value.get("hash_policy") != HASH_POLICY:
        raise ValueError("hash policy drift")
    if value.get("seal_status") != "future_artifact_seal_required":
        raise ValueError("candidate artifact seal status drift")
    _validate_roles(value.get("roles"), require_sealed=False)


def _validate_roles(value: object, *, require_sealed: bool) -> None:
    if not isinstance(value, list):
        raise ValueError("candidate artifact roles must be a list")
    if len(value) != len(ARTIFACT_ROLES):
        raise ValueError("candidate artifact roles must contain every unique role")
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("candidate artifact role must be an object")
        if set(item) != {"role", "path", "bytes", "sha256"}:
            raise ValueError("candidate artifact role fields drift")
        role = item.get("role")
        path = item.get("path")
        if not isinstance(role, str) or role not in ARTIFACT_ROLES:
            raise ValueError("candidate artifact role is not allowed")
        if role in seen_roles:
            raise ValueError("candidate artifact roles must be unique")
        seen_roles.add(role)
        _validate_artifact_path(path)
        if path in seen_paths:
            raise ValueError("candidate artifact paths must be unique")
        seen_paths.add(path)
        expected_path = f"{CANDIDATE_ARTIFACT_ROOT}/{ARTIFACT_FILENAMES[role]}"
        if path != expected_path:
            raise ValueError("candidate artifact path does not match role")
        _validate_artifact_integrity(item, require_sealed=require_sealed)
    if tuple(item["role"] for item in value) != ARTIFACT_ROLES:
        raise ValueError("candidate artifact role order drift")


def _validate_artifact_path(path: object) -> None:
    if not isinstance(path, str) or not path:
        raise ValueError("candidate artifact path must be relative")
    lowered = path.lower()
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError("forbidden final token in candidate artifact path")
    if (
        path.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", path)
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError("candidate artifact path must reject absolute or traversal paths")
    if not path.startswith(CANDIDATE_ARTIFACT_ROOT + "/"):
        raise ValueError("candidate artifact path is outside allowed root")


def _validate_artifact_integrity(item: Mapping[str, object], *, require_sealed: bool) -> None:
    byte_count = item.get("bytes")
    digest = item.get("sha256")
    if require_sealed:
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count <= 0:
            raise ValueError("artifact bytes must be >0 at seal")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("artifact sha256 must be lowercase64 at seal")
        return
    if byte_count is None and digest is None:
        return
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count <= 0:
        raise ValueError("artifact bytes/sha256 must be >0/lowercase64 or pending")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError("artifact sha256 must be lowercase64 or pending")


def _require_equal(
    contract: Mapping[str, object], key: str, expected: object, label: str
) -> None:
    if contract.get(key) != expected:
        raise ValueError(f"{label} drift")


__all__ = [
    "ARTIFACT_FILENAMES",
    "ARTIFACT_ROLES",
    "CALIBRATION",
    "CANDIDATE_ARTIFACT_ROOT",
    "CLASS_ORDER",
    "FEATURE_STRIDE",
    "HASH_POLICY",
    "PARTICLE_TOLERANCE_RADIUS",
    "SCHEMA",
    "SCRATCH_TOLERANCE_HALF_WIDTH",
    "SPARSE_BUNDLE_SHA256",
    "SPARSE_GENERATOR_PATH",
    "SPARSE_GENERATOR_SHA256",
    "SPARSE_MANIFEST_FILE_SHA256",
    "SPARSE_MANIFEST_PATH",
    "SPARSE_TRUTH_FORM",
    "SPARSE_TRUTH_PROVENANCE",
    "build_ticket36_localization_contract",
    "canonical_contract_json",
    "canonical_ticket36_localization_contract_json",
    "main",
    "validate_ticket36_candidate_artifact_seal",
    "validate_ticket36_localization_contract",
]


if __name__ == "__main__":
    raise SystemExit(main())
