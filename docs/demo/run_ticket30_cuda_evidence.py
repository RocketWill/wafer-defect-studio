"""Run and publish the frozen Ticket 30 matched CUDA evidence plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from docs.demo.ticket30_cuda_evidence import (
    EXPECTED_GPU,
    build_ticket30_cuda_evidence_plan,
    hash_ticket30_cuda_evidence_manifest,
    serialize_ticket30_cuda_evidence_manifest,
    validate_completed_ticket30_cuda_evidence_manifest,
)
from docs.demo.ticket30_cuda_evidence import REPLAY_CONFIG

StageExecutor = Callable[[str, Mapping[str, object], Path, Path], Mapping[str, Path]]


def run_ticket30_cuda_evidence(
    output: Path,
    source_image: Path,
    git_commit: str,
    *,
    executor: StageExecutor,
) -> dict:
    """Execute every frozen run/stage and publish a content-addressed manifest."""

    output = output.expanduser().resolve()
    source_image = source_image.expanduser().resolve()
    if not source_image.is_file():
        raise FileNotFoundError(f"Ticket 30 source image does not exist: {source_image}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_ticket30_cuda_evidence_plan(git_commit)
    for run in manifest["runs"]:
        run_dir = output / str(run["seed"]) / str(run["model"])
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for stage in manifest["stage_order"]:
            try:
                produced = executor(stage, run, run_dir, source_image)
            except Exception as error:
                raise RuntimeError(
                    f"Ticket 30 CUDA stage failed: seed={run['seed']} model={run['model']} stage={stage}: {error}"
                ) from error
            for name, artifact_path in produced.items():
                resolved = Path(artifact_path).resolve()
                if not resolved.is_file() or not resolved.is_relative_to(output):
                    raise RuntimeError(
                        f"Ticket 30 artifact is not a file below output: seed={run['seed']} "
                        f"model={run['model']} stage={stage} artifact={name} path={resolved}"
                    )
                relative = resolved.relative_to(output).as_posix()
                artifacts[name] = {
                    "path": relative,
                    "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
                }
        run["artifacts"] = artifacts
    validate_completed_ticket30_cuda_evidence_manifest(manifest)
    (output / "manifest.json").write_text(
        serialize_ticket30_cuda_evidence_manifest(manifest), encoding="utf-8"
    )
    (output / "manifest.sha256").write_text(
        hash_ticket30_cuda_evidence_manifest(manifest) + "\n", encoding="ascii"
    )
    return manifest


def assert_ticket30_cuda_device() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Ticket 30 evidence requires CUDA; torch.cuda.is_available() is false")
    actual = torch.cuda.get_device_name(0)
    if actual != EXPECTED_GPU:
        raise RuntimeError(f"Ticket 30 evidence requires exact GPU {EXPECTED_GPU!r}; actual={actual!r}")


def assert_ticket30_source_asset(source_image: Path) -> None:
    expected = REPLAY_CONFIG["source_asset"]["sha256"]
    actual = hashlib.sha256(source_image.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"Ticket 30 source asset SHA-256 mismatch: expected={expected} actual={actual}"
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-image", type=Path,
        default=REPO / REPLAY_CONFIG["source_asset"]["path"],
    )
    parser.add_argument("--git-commit")
    return parser.parse_args()


def main() -> int:
    from docs.demo.ticket30_real_executor import create_ticket30_real_executor

    args = _arguments()
    assert_ticket30_cuda_device()
    source = args.source_image.expanduser().resolve()
    assert_ticket30_source_asset(source)
    git_commit = args.git_commit or subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    run_ticket30_cuda_evidence(
        args.output, source, git_commit, executor=create_ticket30_real_executor()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
