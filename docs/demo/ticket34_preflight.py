"""Preflight and integrity manifest for Ticket 34 final execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from docs.demo.ticket34_contract import (
    FINAL_MEMBERS,
    FROZEN_CORPUS_SHA256,
    build_ticket34_contract,
    canonical_contract_json,
    validate_ticket33_development_report,
)


SCHEMA = "ticket34-preflight.v1"
_ENVIRONMENT_FIELDS = ("python", "torch", "cuda", "gpu")
_BASE_ROLES = (
    "ticket34_contract",
    "ticket33_development_report",
    "ticket30_corpus_source",
)
_REQUIRED_ROLES = _BASE_ROLES + ("v6_training", "v6_evaluation")
_CHUNK_SIZE = 1024 * 1024


def build_preflight_manifest(
    contract_path: Path,
    report_path: Path,
    corpus_source_path: Path,
    v6_source_paths: Mapping[str, Path],
    *,
    git_commit: str | None = None,
    environment: Mapping[str, object] | None = None,
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Validate non-final inputs and return a canonical preflight manifest.

    This seam only reads the declared source/report/contract files. It never
    builds or renders the Ticket 30 corpus and rejects paths named as final
    evidence members before opening them.
    """

    root = Path.cwd() if repo_root is None else Path(repo_root)
    commit = _resolve_git_commit(git_commit, root)
    normalized_environment = _normalize_environment(environment)
    normalized_sources = _normalize_v6_sources(v6_source_paths)

    contract_file = Path(contract_path)
    report_file = Path(report_path)
    corpus_file = Path(corpus_source_path)
    contract = _read_and_validate_contract(contract_file)
    _read_and_validate_report(report_file)
    _read_and_validate_corpus_source(corpus_file, contract)

    files = [
        _file_entry("ticket34_contract", contract_file, root),
        _file_entry("ticket33_development_report", report_file, root),
        _file_entry("ticket30_corpus_source", corpus_file, root),
    ]
    files.extend(
        _file_entry(role, path, root)
        for role, path in normalized_sources.items()
    )
    files.sort(key=lambda entry: (entry["role"], entry["path"]))

    development_members = contract["development_members"]
    return {
        "schema": SCHEMA,
        "git_commit": commit,
        "environment": normalized_environment,
        "required_roles": list(_REQUIRED_ROLES),
        "development_members": development_members,
        "final_members": list(FINAL_MEMBERS),
        "files": files,
    }


def validate_preflight_manifest(
    manifest: Mapping[str, object], *, root: Path | None = None
) -> dict[str, object]:
    """Revalidate a previously produced manifest and all declared inputs."""

    if not isinstance(manifest, Mapping) or manifest.get("schema") != SCHEMA:
        raise ValueError("invalid Ticket 34 preflight manifest schema")
    expected_keys = {
        "schema",
        "git_commit",
        "environment",
        "required_roles",
        "development_members",
        "final_members",
        "files",
    }
    if set(manifest) != expected_keys:
        raise ValueError("Ticket 34 preflight manifest fields drift")
    _validate_git_commit(manifest["git_commit"])
    _normalize_environment(manifest["environment"])
    if manifest["required_roles"] != list(_REQUIRED_ROLES):
        raise ValueError("Ticket 34 preflight required roles drift")

    development = manifest["development_members"]
    if not isinstance(development, Mapping):
        raise ValueError("Ticket 34 preflight development membership is invalid")
    train = development.get("train")
    validation = development.get("validation")
    try:
        expected_contract = build_ticket34_contract(train, validation)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Ticket 34 preflight development membership is invalid: {error}") from error
    if manifest["final_members"] != expected_contract["final_members"]:
        raise ValueError("Ticket 34 preflight final membership drift")
    if manifest["development_members"] != expected_contract["development_members"]:
        raise ValueError("Ticket 34 preflight development membership drift")

    entries = manifest["files"]
    if not isinstance(entries, list):
        raise ValueError("Ticket 34 preflight files must be a list")
    roles = []
    paths = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"role", "path", "bytes", "sha256"}:
            raise ValueError("Ticket 34 preflight file entry is invalid")
        role = entry["role"]
        path = entry["path"]
        if not isinstance(role, str) or not role:
            raise ValueError("Ticket 34 preflight file role is invalid")
        if not isinstance(path, str) or not path:
            raise ValueError("Ticket 34 preflight file path is invalid")
        if not isinstance(entry["bytes"], int) or entry["bytes"] < 0:
            raise ValueError(f"Ticket 34 preflight file byte count is invalid: {role}")
        if not _is_hex(entry["sha256"], 64):
            raise ValueError(f"Ticket 34 preflight file SHA-256 is invalid: {role}")
        roles.append(role)
        paths.append(path)
        _verify_file_entry(entry, root)
    if entries != sorted(entries, key=lambda entry: (entry["role"], entry["path"])):
        raise ValueError("Ticket 34 preflight file order drift")
    if len(roles) != len(set(roles)) or len(paths) != len(set(paths)):
        raise ValueError("Ticket 34 preflight file roles or paths are duplicated")
    missing = set(_REQUIRED_ROLES) - set(roles)
    if missing:
        raise ValueError("Ticket 34 preflight required file role missing: " + ", ".join(sorted(missing)))

    by_role = {entry["role"]: entry for entry in entries}
    contract_text = _read_manifest_text(by_role["ticket34_contract"], root)
    _validate_contract_text(contract_text, expected_contract)
    report_text = _read_manifest_text(by_role["ticket33_development_report"], root)
    try:
        validate_ticket33_development_report(report_text)
    except ValueError as error:
        raise ValueError(f"Ticket 33 development report validation failed: {error}") from error
    corpus_text = _read_manifest_text(by_role["ticket30_corpus_source"], root)
    _validate_corpus_source_text(corpus_text, expected_contract["final_corpus_sha256"])
    return dict(manifest)


def canonical_preflight_manifest_json(manifest: Mapping[str, object]) -> str:
    """Serialize a manifest deterministically for hashing or persistence."""

    return json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hash_preflight_manifest(manifest: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_preflight_manifest_json(manifest).encode("utf-8")).hexdigest()


def collect_environment() -> dict[str, str]:
    """Collect identities without requiring or probing CUDA hardware."""

    torch_version = "unavailable"
    cuda_version = "unavailable"
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda or "unavailable")
    except (ImportError, AttributeError):
        pass
    return {
        "python": platform.python_version(),
        "torch": torch_version,
        "cuda": cuda_version,
        "gpu": os.environ.get("NVIDIA_GPU_NAME", "unprobed"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--corpus-source", type=Path, required=True)
    parser.add_argument(
        "--v6-source",
        action="append",
        required=True,
        metavar="ROLE=PATH",
        help="repeat for v6_training=PATH and v6_evaluation=PATH",
    )
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        sources = dict(_parse_role_path(value) for value in args.v6_source)
        manifest = build_preflight_manifest(
            args.contract,
            args.report,
            args.corpus_source,
            sources,
            git_commit=args.git_commit,
            repo_root=args.repo_root,
        )
        output = args.output.resolve()
        if output.exists():
            raise ValueError(f"preflight output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_preflight_manifest_json(manifest) + "\n", encoding="utf-8")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Ticket 34 preflight failed: {error}", file=sys.stderr)
        return 1
    print(f"Ticket 34 preflight: {output}", flush=True)
    return 0


def _read_and_validate_contract(path: Path) -> dict[str, object]:
    text = _read_text(_source_path(path, "Ticket 34 contract"), "Ticket 34 contract")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Ticket 34 contract: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Ticket 34 contract must be a JSON object")
    try:
        development = payload["development_members"]
        expected = build_ticket34_contract(development["train"], development["validation"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid Ticket 34 contract: {error}") from error
    _validate_contract_text(text, expected)
    return dict(payload)


def _validate_contract_text(text: str, expected: Mapping[str, object]) -> None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Ticket 34 contract: {error}") from error
    if canonical_contract_json(payload) != text:
        raise ValueError("Ticket 34 contract serialization is not canonical")
    if payload != expected:
        raise ValueError("Ticket 34 contract fields drift")


def _read_and_validate_report(path: Path) -> None:
    text = _read_text(
        _source_path(path, "Ticket 33 development report"),
        "Ticket 33 development report",
    )
    try:
        validate_ticket33_development_report(text)
    except ValueError as error:
        raise ValueError(f"Ticket 33 development report validation failed: {error}") from error


def _read_and_validate_corpus_source(path: Path, contract: Mapping[str, object]) -> None:
    text = _read_text(_source_path(path, "Ticket 30 corpus source"), "Ticket 30 corpus source")
    try:
        expected = contract["final_corpus_sha256"]
    except KeyError as error:
        raise ValueError("Ticket 34 contract is missing final corpus SHA-256") from error
    _validate_corpus_source_text(text, expected)


def _validate_corpus_source_text(text: str, expected: object) -> None:
    match = re.search(
        r"(?m)^FROZEN_CORPUS_SHA256\s*=\s*[\"']([0-9a-f]{64})[\"']\s*$", text
    )
    if match is None or match.group(1) != expected or expected != FROZEN_CORPUS_SHA256:
        raise ValueError("Ticket 30 corpus source frozen SHA-256 drift")


def _file_entry(role: str, path: Path, root: Path) -> dict[str, object]:
    resolved = _source_path(path, role)
    bytes_count, digest = _hash_file(resolved)
    return {
        "role": role,
        "path": _manifest_path(resolved, root),
        "bytes": bytes_count,
        "sha256": digest,
    }


def _verify_file_entry(entry: Mapping[str, object], root: Path | None) -> None:
    path = _resolve_manifest_path(str(entry["path"]), root)
    try:
        bytes_count, digest = _hash_file(_source_path(path, str(entry["role"])))
    except (OSError, ValueError) as error:
        raise ValueError(f"artifact integrity mismatch: role={entry['role']} path={entry['path']}: {error}") from error
    if bytes_count != entry["bytes"] or digest != entry["sha256"]:
        raise ValueError(
            "artifact integrity mismatch: "
            f"role={entry['role']} path={entry['path']} "
            f"expected=({entry['bytes']},{entry['sha256']}) actual=({bytes_count},{digest})"
        )


def _read_manifest_text(entry: Mapping[str, object], root: Path | None) -> str:
    path = _resolve_manifest_path(str(entry["path"]), root)
    try:
        return _read_text(_source_path(path, str(entry["role"])), str(entry["role"]))
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid {entry['role']} input: {error}") from error


def _source_path(path: Path, role: str) -> Path:
    path = Path(path)
    if path.name in FINAL_MEMBERS:
        raise ValueError(f"final evidence member is forbidden for {role}: {path.name}")
    if not path.is_file():
        raise ValueError(f"declared {role} input is not a file: {path}")
    return path


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    bytes_count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
            bytes_count += len(chunk)
    return bytes_count, digest.hexdigest()


def _read_text(path: Path, label: str) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}: {error}") from error


def _manifest_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_manifest_path(path: str, root: Path | None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or root is None:
        return candidate
    return Path(root) / candidate


def _normalize_v6_sources(paths: Mapping[str, Path]) -> dict[str, Path]:
    if not isinstance(paths, Mapping):
        raise ValueError("v6 source paths must be a mapping")
    aliases = {
        "training": "v6_training",
        "v6_training_source": "v6_training",
        "evaluation": "v6_evaluation",
        "v6_evaluation_source": "v6_evaluation",
    }
    normalized: dict[str, Path] = {}
    for raw_role, raw_path in paths.items():
        if not isinstance(raw_role, str) or not raw_role:
            raise ValueError("v6 source role must be a non-empty string")
        role = aliases.get(raw_role, raw_role)
        if role in _BASE_ROLES or role in normalized:
            raise ValueError(f"duplicate preflight file role: {role}")
        normalized[role] = Path(raw_path)
    missing = {"v6_training", "v6_evaluation"} - set(normalized)
    if missing:
        raise ValueError("v6 source role missing: " + ", ".join(sorted(missing)))
    return normalized


def _normalize_environment(environment: Mapping[str, object] | None) -> dict[str, str]:
    values = collect_environment() if environment is None else environment
    if not isinstance(values, Mapping):
        raise ValueError("preflight environment must be a mapping")
    normalized = {}
    for field in _ENVIRONMENT_FIELDS:
        value = values.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"preflight environment identity missing: {field}")
        normalized[field] = value
    return normalized


def _resolve_git_commit(git_commit: str | None, root: Path) -> str:
    if git_commit is not None:
        _validate_git_commit(git_commit)
        return git_commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"cannot resolve git commit at {root}: {error}") from error
    commit = result.stdout.strip()
    _validate_git_commit(commit)
    return commit


def _validate_git_commit(value: object) -> None:
    if not _is_hex(value, 40):
        raise ValueError(f"invalid git commit: {value!r}")


def _parse_role_path(value: str) -> tuple[str, Path]:
    role, separator, path = value.partition("=")
    if not separator or not role or not path:
        raise ValueError(f"invalid --v6-source (expected ROLE=PATH): {value!r}")
    return role, Path(path)


def _is_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "SCHEMA",
    "build_preflight_manifest",
    "canonical_preflight_manifest_json",
    "collect_environment",
    "hash_preflight_manifest",
    "main",
    "validate_preflight_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
