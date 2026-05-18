import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from docs.demo.run_ticket30_cuda_evidence import run_ticket30_cuda_evidence
from docs.demo.ticket30_cuda_evidence import validate_completed_ticket30_cuda_evidence_manifest


class Ticket30CudaRunnerTest(unittest.TestCase):
    def test_script_entrypoint_imports_from_repo_root(self):
        result = subprocess.run(
            [sys.executable, "docs/demo/run_ticket30_cuda_evidence.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runs_each_frozen_stage_and_publishes_content_addressed_manifest(self):
        calls = []

        def executor(stage, run, run_dir, source_image):
            calls.append((run["seed"], run["model"], stage, source_image.name))
            names = ["checkpoint" if stage == "train" else stage]
            if stage == "train" and run["model"] == "spatial_mil_v4":
                names.append("hard_negative_selection")
            result = {}
            for name in names:
                path = run_dir / f"{name}.json"
                path.write_text(json.dumps({"seed": run["seed"], "model": run["model"], "stage": name}))
                result[name] = path
            return result

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "wafer.png"
            source.write_bytes(b"source")
            manifest = run_ticket30_cuda_evidence(root / "evidence", source, "a" * 40, executor=executor)

            self.assertEqual(
                calls,
                [
                    (seed, model, stage, "wafer.png")
                    for seed in (17, 42, 91)
                    for model in ("cam_v2", "patch_v3", "spatial_mil_v4")
                    for stage in ("train", "threshold", "map", "metrics")
                ],
            )
            self.assertEqual(validate_completed_ticket30_cuda_evidence_manifest(manifest), manifest)
            persisted = json.loads((root / "evidence" / "manifest.json").read_text())
            self.assertEqual(persisted, manifest)
            self.assertEqual(len((root / "evidence" / "manifest.sha256").read_text().strip()), 64)


if __name__ == "__main__":
    unittest.main()
