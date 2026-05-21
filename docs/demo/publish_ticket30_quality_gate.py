"""Validate frozen Ticket 30 evidence and publish its quality gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from docs.demo.ticket30_cuda_evidence import (
    hash_ticket30_cuda_evidence_manifest,
    validate_completed_ticket30_cuda_evidence_manifest,
)
from docs.demo.ticket30_quality_gate import (
    CLASS_CODES,
    METRIC_KEYS,
    MODEL,
    evaluate_ticket30_spatial_quality,
)


DEFAULT_EVIDENCE_ROOT = REPO / "docs" / "demo" / "evidence" / "ticket30-final"
DEFAULT_OUTPUT = REPO / "docs" / "demo" / "ticket30-quality-gate.json"
EXPECTED_ARTIFACT_COUNT = 39


def publish_ticket30_quality_gate(
    evidence_root: Path,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Publish one quality-gate artifact from a completed evidence directory."""

    root = Path(evidence_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    sidecar_path = root / "manifest.sha256"
    manifest = _read_json(manifest_path, "evidence manifest")
    try:
        validate_completed_ticket30_cuda_evidence_manifest(manifest)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError(f"Ticket 30 completed manifest is invalid: {manifest_path}: {error}") from error

    manifest_sha256 = hash_ticket30_cuda_evidence_manifest(manifest)
    sidecar = _read_text(sidecar_path, "manifest SHA-256 sidecar").strip()
    if sidecar != manifest_sha256:
        raise ValueError(
            "Ticket 30 manifest SHA-256 mismatch: "
            f"path={sidecar_path} expected={manifest_sha256} actual={sidecar}"
        )

    _validate_artifacts(root, manifest)
    metrics_by_seed = _project_spatial_metrics(root, manifest)
    result = evaluate_ticket30_spatial_quality(manifest_sha256, metrics_by_seed)

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    return result


def _validate_artifacts(root: Path, manifest: Mapping[str, object]) -> None:
    count = 0
    for run in manifest["runs"]:
        for name, artifact in run["artifacts"].items():
            relative = artifact["path"]
            path = (root / relative).resolve()
            context = f"seed={run['seed']} model={run['model']} artifact={name}"
            if not path.is_relative_to(root):
                raise ValueError(f"Ticket 30 artifact escapes evidence root: {context} path={relative}")
            if not path.is_file():
                raise FileNotFoundError(f"Ticket 30 artifact is missing: {context} path={path}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != artifact["sha256"]:
                raise ValueError(
                    f"Ticket 30 artifact SHA-256 mismatch: {context} "
                    f"path={path} expected={artifact['sha256']} actual={actual}"
                )
            count += 1
    if count != EXPECTED_ARTIFACT_COUNT:
        raise ValueError(
            f"Ticket 30 artifact count mismatch: expected={EXPECTED_ARTIFACT_COUNT} actual={count}"
        )


def _project_spatial_metrics(root: Path, manifest: Mapping[str, object]) -> dict[int, dict[str, dict[str, object]]]:
    projected: dict[int, dict[str, dict[str, object]]] = {}
    for run in manifest["runs"]:
        if run["model"] != MODEL:
            continue
        seed = run["seed"]
        metrics_artifact = run["artifacts"]["metrics"]
        path = (root / metrics_artifact["path"]).resolve()
        payload = _read_json(path, f"seed={seed} spatial metrics")
        per_class = payload.get("per_class") if isinstance(payload, Mapping) else None
        if not isinstance(per_class, Mapping) or set(per_class) != set(CLASS_CODES):
            raise ValueError(f"Ticket 30 spatial metrics class membership drift: seed={seed} path={path}")
        projected[seed] = {}
        for class_code in CLASS_CODES:
            raw = per_class[class_code]
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"Ticket 30 spatial metrics class payload is invalid: seed={seed} class={class_code} path={path}"
                )
            missing = [name for name in METRIC_KEYS if name not in raw]
            if missing:
                raise ValueError(
                    f"Ticket 30 spatial metrics fields missing: seed={seed} class={class_code} "
                    f"missing={missing} path={path}"
                )
            projected[seed][class_code] = {name: raw[name] for name in METRIC_KEYS}
    return projected


def _read_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"Ticket 30 cannot read {label}: {path}: {error}") from error


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(_read_text(path, label))
    except json.JSONDecodeError as error:
        raise ValueError(f"Ticket 30 {label} is not valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Ticket 30 {label} must be a JSON object: {path}")
    return value


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    result = publish_ticket30_quality_gate(args.evidence_root, args.output)
    print(f"Ticket 30 quality gate: {result['overall']} recommendation={result['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
