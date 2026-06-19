"""Declarative sparse-objective contract for Ticket 37."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


SCHEMA = "ticket37-sparse-objective-contract.v1"
SOURCE_WIDTH = 1536
SOURCE_HEIGHT = 1536
CLASS_ORDER = ("scratch", "particle")
CASE_SPECS = (
    ("scratch", 1, 0),
    ("scratch", 150, 0),
    ("particle", 0, 1),
    ("particle", 0, 150),
)
CASE_IDS = (
    "ticket35-density-101-train-scratch-001-000-01",
    "ticket35-density-101-train-scratch-150-000-05",
    "ticket35-density-101-train-particle-000-001-06",
    "ticket35-density-101-train-particle-000-150-10",
)
ARCHITECTURE = "resnet18_spatial_logits_v5"
WEIGHTS_POLICY = "imagenet"
WEIGHTS_ID = "ResNet18_Weights.IMAGENET1K_V1"
WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
BATCH_NORM_POLICY = "frozen_running_stats"
EPOCHS = 30
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP_NORM = 5.0
PATCH_SIZE = 128
PATCH_STRIDE = 64
FEATURE_STRIDE = 2
THRESHOLD = 0.5
TOLERANCE_PIXELS = 8
CALIBRATION = "forbidden"
HARDEST_FRACTION = 0.01
DETERMINISTIC_POLICY = {
    "cublas_workspace_config": ":4096:8",
    "torch_deterministic_algorithms": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
}
CANDIDATE_ARTIFACT_ROOT = "artifacts/ticket37-sparse-objective-candidate"
ARTIFACT_ROLES = (
    "checkpoint",
    "confidence_maps",
    "proposal_matching",
    "configuration",
)
ARTIFACT_FILENAMES = {
    "checkpoint": "checkpoint.pt",
    "confidence_maps": "confidence-maps.npz",
    "proposal_matching": "proposal-matching.json",
    "configuration": "configuration.json",
}
HASH_POLICY = "required_after_generation_sha256"
SOURCE_NORMALIZATION = "utf8_lf_bytes"
SOURCE_DEPENDENCIES = (
    (
        "docs/demo/ticket36_single_image_cnn_micro_overfit.py",
        "677a525119af5c394558a4ab9f93a57f72fed68378320193d53d4a4a8dbc3487",
    ),
    (
        "docs/demo/ticket37_proposal_evaluation.py",
        "383ca7d0c2bfffa52cefb8a5c0ce9f5e41e125e264ce5079a4b2e18402cf9121",
    ),
    (
        "src/wafer_defect_studio/spatial_mil.py",
        "78c1fa0bcb98e997ffbb63aff68ec2c2de2a29ee73c65e88fe3193be97715f08",
    ),
    (
        "src/wafer_defect_studio/model_registry.py",
        "dec93cb37c15f1bb183075916e785102341b23b43bef39032d4f4ca5f89f473b",
    ),
    (
        "docs/demo/ticket37_sparse_objective_candidate.py",
        "cf05072e0262955b9881636729b3cbffcc3c7f6e597737ad4da026c37f01ab88",
    ),
    (
        "docs/demo/ticket35_development_corpus.py",
        "fc89a9bc8c83c69aaf4bb4fbeefcb6162e6ee71d55275f87eace62046d65958c",
    ),
    (
        "docs/demo/ticket31_contract.py",
        "a6afc4ceb4a5863c0aebe768e3a3699422b0920da9a2b62f9dd0c3719eb94d99",
    ),
    (
        "docs/demo/defect_oracle.py",
        "7123f1cae9fc126172c97623f28d15c8f4f62a8f36721b0132e2920dacb1f2a6",
    ),
    (
        "docs/demo/ticket30_evidence_corpus.py",
        "1b9c5c41b9ddfa6a876134bb2fd9a0dcc9b19790cf4ba3c404adc7cea019b5de",
    ),
    (
        "docs/demo/ticket36_instance_matching.py",
        "0e39f99343e8b70ac0af1f540c96fe5d7c35e5bcd0b1c899eb69300c7644f0a9",
    ),
    (
        "docs/demo/wafer_quality_evidence.py",
        "bdce6078f5919e407ab198c35a6dd373aaba95a08a9c12f94251b768ec296229",
    ),
    (
        "src/wafer_defect_studio/cam_detection.py",
        "e5147804462ca28023c46f47865fffbf81055a09147361b89e54bbabecfc7964",
    ),
    (
        "src/wafer_defect_studio/confidence_stitching.py",
        "b2745a3d40931dbe8bb51b985ac74fe5e9f4ae1befb75395134b814f50b6befe",
    ),
    (
        "src/wafer_defect_studio/detection_windows.py",
        "4a1d3b66b4c67625ea133329ff16147af9002ae11e9beefeaf7093c05b5b20df",
    ),
    (
        "src/wafer_defect_studio/grid_geometry.py",
        "93532ef2e590fb1a24b940745893bb055456c4d1670bb200b4f2f862589ea8bf",
    ),
    (
        "src/wafer_defect_studio/normalization.py",
        "41664b71bf9ef9b15c195d05e73ffa0ad80e9d98fd84f5f3d364e5cc9700f721",
    ),
    (
        "src/wafer_defect_studio/training_dataset.py",
        "fbf1410d99dde94a5a640e66c119d88de3828f6ef73ee7808e6a03daf9ee743a",
    ),
    (
        "src/wafer_defect_studio/training_protocol.py",
        "762cb345aca3fcaec3da13e66eef8fb900b51e05a17144b3e8cabd756f8d95e7",
    ),
)
_FORBIDDEN_PATH_TOKENS = (
    "ticket30",
    "ticket34",
    "ticket35",
    "ticket36-repair-candidate",
)


def build_ticket37_sparse_objective_contract() -> dict[str, object]:
    """Build the declarative contract without pixels, artifacts, or execution."""

    return {
        "schema": SCHEMA,
        "source_geometry": {
            "coordinate_system": "source_pixels",
            "width": SOURCE_WIDTH,
            "height": SOURCE_HEIGHT,
        },
        "class_order": list(CLASS_ORDER),
        "membership": {
            "source": "ticket36_single_image_cnn_micro_overfit",
            "seed": 101,
            "split": "train",
            "case_specs": [list(spec) for spec in CASE_SPECS],
            "case_ids": list(CASE_IDS),
        },
        "recipe": {
            "architecture": ARCHITECTURE,
            "weights_policy": WEIGHTS_POLICY,
            "weights_id": WEIGHTS_ID,
            "weights_url": WEIGHTS_URL,
            "weights_sha256": WEIGHTS_SHA256,
            "batch_norm_policy": BATCH_NORM_POLICY,
            "epochs": EPOCHS,
            "optimizer": "adamw",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "patch_size": PATCH_SIZE,
            "patch_stride": PATCH_STRIDE,
            "feature_stride": FEATURE_STRIDE,
            "threshold": THRESHOLD,
            "tolerance_pixels": TOLERANCE_PIXELS,
            "calibration": CALIBRATION,
        },
        "objective": {
            "loss": "sparse_instance_localization_loss",
            "terms": [
                "per_instance_coverage",
                "all_row_hardest_tail_far",
                "local_full_far",
            ],
            "hardest_fraction": HARDEST_FRACTION,
            "context_weights": {
                "dense_absent_class_loss": 0.5,
                "same_image_grid_ranking_loss": 0.25,
                "overlap_consistency_loss": 0.10,
            },
            "forbidden_terms": [
                "positive_spatial_topk_loss",
                "present_sparse_budget_loss",
            ],
        },
        "evaluation": {
            "method": "rendered_truth_components",
            "gates": {
                "proposal_precision_min": 1.0,
                "proposal_recall_min": 1.0,
                "unique_truth_touch_recall_min": 1.0,
                "unmatched_truth_max": 0,
                "unmatched_proposal_max": 0,
                "cross_component_merge_max": 0,
                "normal_grid_leaks_max": 0,
                "asserted_grid_occupancy_p95_max": 0.25,
            },
            "remote_response": {
                "gating": "non_gating_diagnostic",
                "claim": "sparse_tolerance_not_segmentation_mask",
            },
        },
        "deterministic_policy": dict(DETERMINISTIC_POLICY),
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
        "source_dependencies": [
            {
                "path": path,
                "sha256": sha256,
                "hash_algorithm": "sha256",
                "normalization": SOURCE_NORMALIZATION,
            }
            for path, sha256 in SOURCE_DEPENDENCIES
        ],
        "execution": {
            "formal_run_count": 0,
            "training": "forbidden_until_contract_commit",
            "inference": "forbidden_until_contract_commit",
            "cuda": "forbidden_until_contract_commit",
            "pixel_materialization": "forbidden_until_contract_commit",
        },
        "boundary": {
            "cam_default": "cam_v2",
            "spatial_mil_status": "experimental",
            "segmentation": "not_claimed",
            "neurocle_equivalence": "not_claimed",
            "production_accuracy": "not_claimed",
        },
        "claims": [
            "not segmentation",
            "not Neurocle equivalence",
            "not production accuracy",
        ],
    }


def validate_ticket37_sparse_objective_contract(
    contract: Mapping[str, object],
    *,
    repo_root: Path | None = None,
    source_texts: Mapping[str, str] | None = None,
    verify_live_sources: bool = True,
) -> None:
    """Fail closed on contract drift, source drift, or unsafe artifact paths."""

    if not isinstance(contract, Mapping):
        raise ValueError("Ticket 37 sparse objective contract must be an object")
    expected = build_ticket37_sparse_objective_contract()
    if set(contract) != set(expected):
        raise ValueError("Ticket 37 sparse objective contract fields drift")
    for key, value in expected.items():
        if key not in {"source_dependencies", "candidate_artifacts"} and contract.get(key) != value:
            raise ValueError(f"Ticket 37 {key} contract drift")
    _validate_source_dependencies(
        contract.get("source_dependencies"),
        repo_root=repo_root,
        source_texts=source_texts,
        verify_live_sources=verify_live_sources,
    )
    _validate_candidate_artifacts(contract.get("candidate_artifacts"))


def canonical_ticket37_sparse_objective_contract_json(
    contract: Mapping[str, object],
) -> str:
    """Return canonical JSON after validating the declarative contract."""

    validate_ticket37_sparse_objective_contract(contract, verify_live_sources=False)
    return json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_contract_json(contract: Mapping[str, object]) -> str:
    return canonical_ticket37_sparse_objective_contract_json(contract)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        if output.exists():
            raise ValueError(f"Ticket 37 sparse objective contract output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            canonical_ticket37_sparse_objective_contract_json(
                build_ticket37_sparse_objective_contract()
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"Ticket 37 sparse objective contract failed: {error}", file=sys.stderr)
        return 1
    print(f"Ticket 37 sparse objective contract: {output}", flush=True)
    return 0


def _validate_source_dependencies(
    value: object,
    *,
    repo_root: Path | None,
    source_texts: Mapping[str, str] | None,
    verify_live_sources: bool,
) -> None:
    expected = build_ticket37_sparse_objective_contract()["source_dependencies"]
    if value != expected:
        raise ValueError("Ticket 37 source dependency contract drift")
    if source_texts is not None:
        if not isinstance(source_texts, Mapping):
            raise ValueError("Ticket 37 source texts must be an object")
        paths = {path for path, _sha256 in SOURCE_DEPENDENCIES}
        if any(path not in paths for path in source_texts):
            raise ValueError("Ticket 37 source text path is not declared")
    if not isinstance(verify_live_sources, bool):
        raise ValueError("Ticket 37 live-source policy must be boolean")
    root = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root)
    dependencies = SOURCE_DEPENDENCIES if verify_live_sources else tuple(
        item for item in SOURCE_DEPENDENCIES if source_texts is not None and item[0] in source_texts
    )
    for path, expected_sha256 in dependencies:
        if source_texts is not None and path in source_texts:
            raw = _normalized_utf8_lf_bytes(source_texts[path], path)
        else:
            raw = _read_normalized_file_bytes(root / path, path)
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_sha256:
            raise ValueError(f"Ticket 37 source dependency SHA-256 mismatch: {path}")


def _validate_candidate_artifacts(value: object) -> None:
    expected = build_ticket37_sparse_objective_contract()["candidate_artifacts"]
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError("Ticket 37 candidate artifact contract fields drift")
    if value.get("root") != CANDIDATE_ARTIFACT_ROOT:
        raise ValueError("Ticket 37 candidate artifact root drift")
    if value.get("hash_policy") != HASH_POLICY:
        raise ValueError("Ticket 37 candidate artifact hash policy drift")
    if value.get("seal_status") != "future_artifact_seal_required":
        raise ValueError("Ticket 37 candidate artifact seal status drift")
    roles = value.get("roles")
    if not isinstance(roles, list) or len(roles) != len(ARTIFACT_ROLES):
        raise ValueError("Ticket 37 candidate artifact roles drift")
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    for item in roles:
        if not isinstance(item, Mapping) or set(item) != {"role", "path", "bytes", "sha256"}:
            raise ValueError("Ticket 37 candidate artifact role fields drift")
        role = item["role"]
        path = item["path"]
        if not isinstance(role, str) or role not in ARTIFACT_ROLES or role in seen_roles:
            raise ValueError("Ticket 37 candidate artifact roles must be unique and allowed")
        seen_roles.add(role)
        _validate_artifact_path(path)
        if path in seen_paths:
            raise ValueError("Ticket 37 candidate artifact paths must be unique")
        seen_paths.add(path)
        if path != f"{CANDIDATE_ARTIFACT_ROOT}/{ARTIFACT_FILENAMES[role]}":
            raise ValueError("Ticket 37 candidate artifact path does not match role")
        if item["bytes"] is not None or item["sha256"] is not None:
            raise ValueError("Ticket 37 candidate artifact hashes must remain pending")
    if tuple(item["role"] for item in roles) != ARTIFACT_ROLES:
        raise ValueError("Ticket 37 candidate artifact role order drift")


def _validate_artifact_path(path: object) -> None:
    if not isinstance(path, str) or not path:
        raise ValueError("Ticket 37 candidate artifact path must be relative")
    lowered = path.lower()
    if any(token in lowered for token in _FORBIDDEN_PATH_TOKENS):
        raise ValueError("Ticket 37 candidate artifact path contains forbidden evidence token")
    if (
        path.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", path)
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError("Ticket 37 candidate artifact path must reject absolute or traversal paths")
    if not path.startswith(CANDIDATE_ARTIFACT_ROOT + "/"):
        raise ValueError("Ticket 37 candidate artifact path is outside allowed root")


def _normalized_utf8_lf_bytes(value: str, label: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be UTF-8 text")
    return value.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _read_normalized_file_bytes(path: Path, label: str) -> bytes:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read Ticket 37 source dependency {label}: {error}") from error
    return _normalized_utf8_lf_bytes(text, label)


__all__ = [
    "ARCHITECTURE",
    "ARTIFACT_FILENAMES",
    "ARTIFACT_ROLES",
    "BATCH_NORM_POLICY",
    "CASE_IDS",
    "CASE_SPECS",
    "CANDIDATE_ARTIFACT_ROOT",
    "CLASS_ORDER",
    "EPOCHS",
    "FEATURE_STRIDE",
    "HASH_POLICY",
    "SCHEMA",
    "SOURCE_DEPENDENCIES",
    "TOLERANCE_PIXELS",
    "WEIGHTS_ID",
    "WEIGHTS_SHA256",
    "WEIGHTS_URL",
    "DETERMINISTIC_POLICY",
    "build_ticket37_sparse_objective_contract",
    "canonical_contract_json",
    "canonical_ticket37_sparse_objective_contract_json",
    "main",
    "validate_ticket37_sparse_objective_contract",
]


if __name__ == "__main__":
    raise SystemExit(main())
