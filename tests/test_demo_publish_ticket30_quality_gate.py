import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from docs.demo.publish_ticket30_quality_gate import publish_ticket30_quality_gate
from docs.demo.ticket30_cuda_evidence import (
    build_ticket30_cuda_evidence_plan,
    hash_ticket30_cuda_evidence_manifest,
    serialize_ticket30_cuda_evidence_manifest,
)


SEEDS = (17, 42, 91)
CLASSES = ("scratch", "particle")
METRIC_KEYS = (
    "defect_instances",
    "defect_coverage_recall",
    "grid_precision",
    "grid_recall",
    "normal_grid_leak_rate",
    "asserted_grid_occupancy_p95",
)


def _metric_values(seed, class_code):
    values = {
        "defect_instances": 150,
        "defect_coverage_recall": 1.0,
        "grid_precision": 0.95,
        "grid_recall": 0.95,
        "normal_grid_leak_rate": 0.05,
        "asserted_grid_occupancy_p95": 0.25,
    }
    if seed == 42 and class_code == "scratch":
        values["grid_precision"] = 0.5
    return values


def _write_evidence(root: Path):
    manifest = build_ticket30_cuda_evidence_plan("a" * 40)
    for run in manifest["runs"]:
        names = ["checkpoint", "threshold", "map", "metrics"]
        if run["model"] == "spatial_mil_v4":
            names.append("hard_negative_selection")
        run["artifacts"] = {
            name: {"path": f"{run['seed']}/{run['model']}/{name}.json"}
            for name in names
        }
        for name, artifact in _artifacts(run).items():
            path = root / artifact["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "metrics":
                payload = {
                    "case_count": 1,
                    "per_class": {
                        code: {
                            **_metric_values(run["seed"], code),
                            "covered_defect_instances": 150,
                            "grid_tp": 1,
                        }
                        for code in CLASSES
                    },
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
            else:
                path.write_bytes(f"{run['seed']}-{run['model']}-{name}".encode("ascii"))
            artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = root / "manifest.json"
    manifest_path.write_text(serialize_ticket30_cuda_evidence_manifest(manifest), encoding="utf-8")
    (root / "manifest.sha256").write_text(
        hash_ticket30_cuda_evidence_manifest(manifest) + "\n", encoding="ascii"
    )
    return hash_ticket30_cuda_evidence_manifest(manifest)


def _artifacts(run):
    return run["artifacts"]


class Ticket30QualityGatePublisherTest(unittest.TestCase):
    def test_validates_manifest_and_artifacts_projects_metrics_and_writes_canonical_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            root.mkdir()
            manifest_sha256 = _write_evidence(root)
            output = Path(temporary) / "published" / "ticket30-quality-gate.json"

            result = publish_ticket30_quality_gate(root, output)

            self.assertEqual(result["overall"], "FAIL")
            self.assertEqual(result["recommendation"], "cam_v2")
            entry = result["per_seed"][42]["scratch"]
            self.assertEqual(set(entry["metrics"]), set(METRIC_KEYS))
            self.assertNotIn("covered_defect_instances", entry["metrics"])
            self.assertEqual(entry["metrics"]["grid_precision"], 0.5)
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(persisted["schema"], "ticket30-quality-gate.v1")
            self.assertEqual(persisted["manifest_sha256"], manifest_sha256)
            self.assertEqual(len(persisted["per_seed"]), 3)
            self.assertEqual(len(persisted["per_seed"]["17"]), 2)

    def test_rejects_manifest_sidecar_or_artifact_tamper_with_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            root.mkdir()
            _write_evidence(root)
            output = Path(temporary) / "out.json"

            (root / "manifest.sha256").write_text("0" * 64, encoding="ascii")
            with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
                publish_ticket30_quality_gate(root, output)

            _write_evidence(root)
            artifact = root / "17" / "spatial_mil_v4" / "metrics.json"
            artifact.write_text(artifact.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed=17 model=spatial_mil_v4 artifact=metrics"):
                publish_ticket30_quality_gate(root, output)


if __name__ == "__main__":
    unittest.main()
