"""Resumable five-seed development gate for Spatial MIL v6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from docs.demo.ticket31_contract import DEVELOPMENT_SEEDS
from docs.demo.ticket31_development_gate import _canonical
from docs.demo.ticket33_development_smoke import run_development_seed


SCHEMA = "ticket33-development-gate.v1"


def run_development_gate(output_root: Path) -> dict[str, object]:
    per_seed = {}
    for seed in DEVELOPMENT_SEEDS:
        seed_root = output_root / str(seed)
        seed_root.mkdir(parents=True, exist_ok=True)
        complete_path = seed_root / "complete.json"
        if complete_path.is_file():
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            _verify_completed_seed(seed, seed_root, complete)
            result = complete["result"]
            print(f"Ticket 33 seed {seed}: resumed completed evidence", flush=True)
        else:
            result = run_development_seed(seed, seed_root)
            complete = {
                "schema": "ticket33-development-seed-complete.v1",
                "result": result,
                "artifacts": {
                    name: _artifact(path, seed_root)
                    for name, path in (
                        ("model", seed_root / "model-state.pt"),
                        ("maps", seed_root / "validation-maps.npz"),
                    )
                },
            }
            complete_path.write_text(_canonical(complete), encoding="utf-8")
        per_seed[str(seed)] = {
            **result,
            "artifacts": {
                "model": _artifact(seed_root / "model-state.pt", output_root),
                "maps": _artifact(seed_root / "validation-maps.npz", output_root),
                "complete": _artifact(complete_path, output_root),
            },
        }
    rows = [row for result in per_seed.values() for row in result["per_class"].values()]
    overall = "PASS" if all(row["overall"] == "PASS" for row in rows) else "FAIL"
    return {
        "schema": SCHEMA,
        "overall": overall,
        "recommendation": "run_final_held_out_gate" if overall == "PASS" else "cam_v2",
        "spatial_mil_v6_status": "development_pass" if overall == "PASS" else "experimental",
        "per_seed": per_seed,
        "claims": ["development evidence", "approximate localization", "not segmentation", "not Neurocle equivalence"],
    }


def _artifact(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _verify_completed_seed(seed: int, seed_root: Path, complete) -> None:
    if complete.get("schema") != "ticket33-development-seed-complete.v1" or complete.get("result", {}).get("seed") != seed:
        raise ValueError(f"Ticket 33 seed {seed} completion marker is invalid")
    for name, artifact in complete["artifacts"].items():
        path = seed_root / artifact["path"]
        observed = _artifact(path, seed_root)
        if observed != artifact:
            raise ValueError(f"Ticket 33 seed {seed} {name} artifact integrity mismatch")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report = run_development_gate(output_root)
    args.report.resolve().write_text(_canonical(report), encoding="utf-8")
    print(f"Ticket 33 development gate: {report['overall']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "run_development_gate"]
