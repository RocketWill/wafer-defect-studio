"""Publish Ticket 37 results from the completed one-run artifact bundle."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from docs.demo.ticket37_sparse_objective_candidate import (
    CANDIDATE_ARTIFACT_ROOT,
    CASE_IDS,
    REPORT_PATH,
    _build_report,
    _contract_sha256,
    _git_state,
    _read_float32_maps_artifact,
    _read_json_artifact,
    _resolve_fixed_report_path,
    _validate_checkpoint_artifact,
    _validate_configuration_artifact,
    _verify_ticket37_artifact_bytes,
    _write_json,
    evaluate_ticket37_stored_maps,
    seal_ticket37_artifacts,
)
from docs.demo.ticket37_sparse_objective_contract import (
    canonical_ticket37_sparse_objective_contract_json,
    validate_ticket37_sparse_objective_contract,
)


EVIDENCE_HEAD = "043db6a9a5d3c26b99430a2d08589da7bb4b8a37"
FROZEN_ARTIFACT_SEAL = {
    "root": CANDIDATE_ARTIFACT_ROOT,
    "hash_policy": "required_after_generation_sha256",
    "seal_status": "sealed_sha256",
    "roles": [
        {
            "role": "checkpoint",
            "path": f"{CANDIDATE_ARTIFACT_ROOT}/checkpoint.pt",
            "bytes": 179161721,
            "sha256": "4967524fea27de9fa54ffcbe5dd1f7a9a9701ede4fea7fd6df397828e9c27e18",
        },
        {
            "role": "confidence_maps",
            "path": f"{CANDIDATE_ARTIFACT_ROOT}/confidence-maps.npz",
            "bytes": 11709408,
            "sha256": "8f23bd2ec865f5c238376feb9c2836803e2d149e9b4ac434c5b56ecf9e934029",
        },
        {
            "role": "proposal_matching",
            "path": f"{CANDIDATE_ARTIFACT_ROOT}/proposal-matching.json",
            "bytes": 4712,
            "sha256": "1a94166dc2ddfb03cde6ebb8a7097ff886bab4a56e12397edc90c33c9b4124ac",
        },
        {
            "role": "configuration",
            "path": f"{CANDIDATE_ARTIFACT_ROOT}/configuration.json",
            "bytes": 8434,
            "sha256": "cbfb667b8b12b28838fba66ff461f21e77fce967a99375f7501f4467ce0d9764",
        },
    ],
}


def validate_ticket37_matching_against_recomputed(
    stored: Mapping[str, object],
    recomputed: Mapping[str, object],
    *,
    case_order: Sequence[str] = CASE_IDS,
) -> None:
    """Require exact matching content without depending on JSON object order."""

    expected_order = tuple(case_order)
    if (
        not isinstance(stored, Mapping)
        or stored.get("schema") != recomputed.get("schema")
        or stored.get("case_order") != list(expected_order)
        or not isinstance(stored.get("per_case"), Mapping)
        or set(stored["per_case"]) != set(expected_order)
        or stored != recomputed
    ):
        raise ValueError("Ticket 37 stored matching differs from sealed-map recomputation")


def validate_ticket37_frozen_artifact_seal(observed: Mapping[str, object]) -> None:
    """Anchor recovery to the four immutable files produced by the formal run."""

    if observed != FROZEN_ARTIFACT_SEAL:
        raise ValueError("Ticket 37 completed-run artifact receipt drift")


def write_ticket37_recovered_report(path: Path, report: Mapping[str, object]) -> None:
    """Write the recovered report once using the candidate canonical writer."""

    _write_json(Path(path), report)


def normalize_ticket37_configuration_key_order(
    value: Mapping[str, object],
    *,
    case_order: Sequence[str] = CASE_IDS,
) -> dict[str, object]:
    """Restore the declared loss-history order without changing its values."""

    normalized = copy.deepcopy(dict(value))
    history = normalized.get("loss_history")
    expected_order = tuple(case_order)
    if not isinstance(history, Mapping) or set(history) != set(expected_order):
        raise ValueError("Ticket 37 configuration loss-history cases drift")
    normalized["loss_history"] = {case_id: history[case_id] for case_id in expected_order}
    return normalized


def _validate_configuration_without_object_order(path: Path) -> dict[str, object]:
    value = _read_json_artifact(path)
    normalized = normalize_ticket37_configuration_key_order(value)
    with tempfile.TemporaryDirectory() as temporary:
        normalized_path = Path(temporary) / "configuration.json"
        normalized_path.write_text(
            json.dumps(normalized, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _validate_configuration_artifact(normalized_path)
    return value


def publish_ticket37_existing_artifact_report(
    *,
    repo_root: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, object]:
    """Publish a report without training, inference, or artifact mutation."""

    root = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root).expanduser().resolve()
    report_file = _resolve_fixed_report_path(root, report_path)
    candidate_root = root / CANDIDATE_ARTIFACT_ROOT
    if not candidate_root.is_dir():
        raise FileNotFoundError("Ticket 37 completed candidate artifact root is missing")
    if report_file.exists():
        raise FileExistsError("Ticket 37 candidate report already exists")
    head, status = _git_state(root)
    if status:
        raise RuntimeError("Ticket 37 report recovery requires a clean committed git worktree")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EVIDENCE_HEAD, head],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if ancestry.returncode != 0:
        raise RuntimeError("Ticket 37 recovery HEAD does not descend from the evidence run")

    contract_path = root / "docs/demo/ticket37-sparse-objective-contract.json"
    tracked_text = contract_path.read_text(encoding="utf-8")
    contract = json.loads(tracked_text)
    if tracked_text != canonical_ticket37_sparse_objective_contract_json(contract) + "\n":
        raise ValueError("Ticket 37 tracked contract is not canonical")
    validate_ticket37_sparse_objective_contract(contract, repo_root=root)

    configuration_path = candidate_root / "configuration.json"
    configuration = _validate_configuration_without_object_order(configuration_path)
    if (
        configuration.get("git_head") != EVIDENCE_HEAD
        or configuration.get("contract_sha256") != _contract_sha256(contract)
        or configuration.get("formal_run_count") != 1
        or configuration.get("rerun") is not False
    ):
        raise ValueError("Ticket 37 completed-run configuration provenance drift")

    observed_seal = seal_ticket37_artifacts(candidate_root, repo_root=root)
    validate_ticket37_frozen_artifact_seal(observed_seal)
    seal = copy.deepcopy(FROZEN_ARTIFACT_SEAL)
    _verify_ticket37_artifact_bytes(seal, root)
    _validate_checkpoint_artifact(candidate_root / "checkpoint.pt")
    maps = _read_float32_maps_artifact(candidate_root / "confidence-maps.npz")
    rows, recomputed_matching = evaluate_ticket37_stored_maps(maps)
    stored_matching = _read_json_artifact(candidate_root / "proposal-matching.json")
    validate_ticket37_matching_against_recomputed(stored_matching, recomputed_matching)

    report = _build_report(
        contract,
        rows,
        recomputed_matching,
        seal,
        deterministic_policy=configuration["deterministic_policy"],
    )
    write_ticket37_recovered_report(report_file, report)
    if _read_json_artifact(report_file) != report:
        raise ValueError("Ticket 37 recovered report re-read drift")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)
    try:
        report = publish_ticket37_existing_artifact_report(report_path=args.report)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Ticket 37 existing artifact report failed: {error}", file=sys.stderr)
        return 1
    print(f"Ticket 37 existing artifact report: {report['overall']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
