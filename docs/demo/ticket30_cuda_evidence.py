"""Frozen matched-CUDA evidence manifest contract for Ticket 30."""

from __future__ import annotations

import hashlib
import json

from docs.demo.ticket30_evidence_corpus import CLASS_CODES, FROZEN_CORPUS_SHA256, SEEDS

EXPECTED_GPU = "NVIDIA GeForce RTX 3090"
MODELS = ("cam_v2", "patch_v3", "spatial_mil_v4")
STAGE_ORDER = ("train", "threshold", "map", "metrics")
SCORE_DOMAINS = {
    "cam_v2": "normalized_window_cam",
    "patch_v3": "patch_sigmoid",
    "spatial_mil_v4": "absolute_spatial_probability",
}
BASE_ARTIFACTS = ("checkpoint", "threshold", "map", "metrics")


def build_ticket30_cuda_evidence_plan(git_commit: str) -> dict:
    if not _is_hex(git_commit, 40):
        raise ValueError(f"invalid git commit: {git_commit!r}")
    runs = []
    for seed in SEEDS:
        for model in MODELS:
            run = {
                "seed": seed,
                "model": model,
                "epochs": 20,
                "weights": "imagenet",
                "score_domain": SCORE_DOMAINS[model],
            }
            if model in ("patch_v3", "spatial_mil_v4"):
                run["geometry"] = {"patch_size": 128, "patch_stride": 64}
            if model == "spatial_mil_v4":
                run["refinement_epochs"] = 5
            runs.append(run)
    return {
        "schema_version": 1,
        "git_commit": git_commit,
        "corpus_sha256": FROZEN_CORPUS_SHA256,
        "device": "cuda",
        "gpu": EXPECTED_GPU,
        "class_codes": list(CLASS_CODES),
        "stage_order": list(STAGE_ORDER),
        "runs": runs,
    }


def serialize_ticket30_cuda_evidence_manifest(manifest: dict) -> str:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"))


def hash_ticket30_cuda_evidence_manifest(manifest: dict) -> str:
    return hashlib.sha256(serialize_ticket30_cuda_evidence_manifest(manifest).encode("utf-8")).hexdigest()


def validate_completed_ticket30_cuda_evidence_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict):
        raise ValueError("invalid ticket30 evidence manifest")
    expected = build_ticket30_cuda_evidence_plan(manifest.get("git_commit"))
    if set(manifest) != set(expected):
        raise ValueError("ticket30 evidence manifest fields drift")
    for field in ("schema_version", "corpus_sha256", "device", "gpu", "class_codes", "stage_order"):
        if manifest.get(field) != expected[field]:
            label = {"device": "CUDA device", "gpu": "GPU"}.get(field, field.replace("_", " "))
            raise ValueError(f"ticket30 evidence {label} drift")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != len(expected["runs"]):
        raise ValueError("ticket30 evidence run membership drift")
    actual_keys = [(run.get("seed"), run.get("model")) for run in runs if isinstance(run, dict)]
    expected_keys = [(run["seed"], run["model"]) for run in expected["runs"]]
    if actual_keys != expected_keys or len(set(actual_keys)) != len(actual_keys):
        raise ValueError("ticket30 evidence run seed/model membership or order drift")
    artifact_paths = set()
    for run, planned in zip(runs, expected["runs"], strict=True):
        if set(run) != set(planned) | {"artifacts"}:
            raise ValueError(f"ticket30 evidence run fields drift: seed={planned['seed']} model={planned['model']}")
        for field, value in planned.items():
            if run.get(field) != value:
                label = "score domain" if field == "score_domain" else field
                raise ValueError(f"ticket30 evidence {label} drift: seed={planned['seed']} model={planned['model']}")
        required = set(BASE_ARTIFACTS)
        if planned["model"] == "spatial_mil_v4":
            required.add("hard_negative_selection")
        artifacts = run.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != required:
            raise ValueError(f"ticket30 evidence artifact membership drift: model={planned['model']}")
        for name, artifact in artifacts.items():
            if (
                not isinstance(artifact, dict)
                or set(artifact) != {"path", "sha256"}
                or not isinstance(artifact["path"], str)
                or not artifact["path"]
                or not _is_hex(artifact["sha256"], 64)
            ):
                raise ValueError(f"ticket30 evidence artifact invalid: model={planned['model']} artifact={name}")
            if artifact["path"] in artifact_paths:
                raise ValueError(f"ticket30 evidence duplicate artifact path: {artifact['path']}")
            artifact_paths.add(artifact["path"])
    return manifest


def _is_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(character in "0123456789abcdef" for character in value)
