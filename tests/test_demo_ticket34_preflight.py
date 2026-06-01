import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from docs.demo.ticket34_contract import (
    FROZEN_CORPUS_SHA256,
    build_ticket34_contract,
    canonical_contract_json,
)
from docs.demo.ticket34_preflight import (
    build_preflight_manifest,
    canonical_preflight_manifest_json,
    validate_preflight_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "docs/demo/ticket33-development-gate.json"
ENVIRONMENT = {
    "python": "3.11.9",
    "torch": "2.3.1+cu121",
    "cuda": "12.1",
    "gpu": "NVIDIA GeForce RTX 3090",
}


class Ticket34PreflightTest(unittest.TestCase):
    def _write_inputs(self, root: Path, *, train=("train-a",), validation=("validation-a",)):
        contract_path = root / "ticket34-contract.json"
        contract = build_ticket34_contract(train, validation)
        contract_path.write_text(canonical_contract_json(contract), encoding="utf-8")
        report_path = root / "ticket33-development-gate.json"
        shutil.copyfile(REPORT_PATH, report_path)
        corpus_path = root / "ticket30_evidence_corpus.py"
        corpus_path.write_text(
            f"FROZEN_CORPUS_SHA256 = '{FROZEN_CORPUS_SHA256}'\n", encoding="utf-8"
        )
        training_path = root / "ticket33_development_smoke.py"
        training_path.write_text("def train():\n    return None\n", encoding="utf-8")
        evaluation_path = root / "ticket33_development_gate.py"
        evaluation_path.write_text("def evaluate():\n    return None\n", encoding="utf-8")
        return contract_path, report_path, corpus_path, {
            "v6_training": training_path,
            "v6_evaluation": evaluation_path,
        }

    def _build(self, root: Path):
        contract, report, corpus, sources = self._write_inputs(root)
        manifest = build_preflight_manifest(
            contract_path=contract,
            report_path=report,
            corpus_source_path=corpus,
            v6_source_paths=sources,
            git_commit="a" * 40,
            environment=ENVIRONMENT,
            repo_root=root,
        )
        return manifest, (contract, report, corpus, sources)

    def test_builds_canonical_manifest_and_validates_it_without_cuda(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self._build(root)

            self.assertEqual(manifest["schema"], "ticket34-preflight.v1")
            self.assertEqual(manifest["git_commit"], "a" * 40)
            self.assertEqual(manifest["environment"], ENVIRONMENT)
            self.assertEqual(
                tuple(entry["role"] for entry in manifest["files"]),
                (
                    "ticket30_corpus_source",
                    "ticket33_development_report",
                    "ticket34_contract",
                    "v6_evaluation",
                    "v6_training",
                ),
            )
            for entry in manifest["files"]:
                self.assertEqual(set(entry), {"role", "path", "bytes", "sha256"})
            self.assertEqual(validate_preflight_manifest(manifest, root=root), manifest)
            self.assertEqual(
                canonical_preflight_manifest_json(manifest),
                canonical_preflight_manifest_json(json.loads(canonical_preflight_manifest_json(manifest))),
            )
            self.assertNotIn(
                "ticket30-evidence-17.png",
                tuple(entry["path"] for entry in manifest["files"]),
            )

    def test_tampered_declared_input_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, (_, report, corpus, sources) = self._build(root)
            for path in (report, corpus, *sources.values()):
                with self.subTest(path=path.name):
                    original = path.read_bytes()
                    path.write_bytes(original + b"tampered")
                    with self.assertRaisesRegex(ValueError, "artifact integrity mismatch"):
                        validate_preflight_manifest(manifest, root=root)
                    path.write_bytes(original)

    def test_rejects_development_final_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, report, corpus, sources = self._write_inputs(root)
            payload = json.loads(contract.read_text(encoding="utf-8"))
            payload["development_members"]["train"] = ["ticket30-evidence-17.png"]
            contract.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "final evidence member"):
                build_preflight_manifest(
                    contract_path=contract,
                    report_path=report,
                    corpus_source_path=corpus,
                    v6_source_paths=sources,
                    git_commit="a" * 40,
                    environment=ENVIRONMENT,
                    repo_root=root,
                )

    def test_rejects_final_image_path_without_reading_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, report, corpus, sources = self._write_inputs(root)
            final_path = root / "ticket30-evidence-17.png"
            final_path.write_bytes(b"pixel payload must not be read")
            sources["v6_training"] = final_path
            with self.assertRaisesRegex(ValueError, "final evidence member"):
                build_preflight_manifest(
                    contract_path=contract,
                    report_path=report,
                    corpus_source_path=corpus,
                    v6_source_paths=sources,
                    git_commit="a" * 40,
                    environment=ENVIRONMENT,
                    repo_root=root,
                )

    def test_cli_writes_canonical_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, report, corpus, sources = self._write_inputs(root)
            output = root / "preflight.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "docs/demo/ticket34_preflight.py",
                    "--contract",
                    str(contract),
                    "--report",
                    str(report),
                    "--corpus-source",
                    str(corpus),
                    "--v6-source",
                    f"v6_training={sources['v6_training']}",
                    "--v6-source",
                    f"v6_evaluation={sources['v6_evaluation']}",
                    "--git-commit",
                    "a" * 40,
                    "--output",
                    str(output),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                canonical_preflight_manifest_json(json.loads(output.read_text(encoding="utf-8"))) + "\n",
            )

    def test_cli_does_not_overwrite_sealed_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, report, corpus, sources = self._write_inputs(root)
            output = root / "preflight.json"
            output.write_text("sealed\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "docs/demo/ticket34_preflight.py",
                    "--contract",
                    str(contract),
                    "--report",
                    str(report),
                    "--corpus-source",
                    str(corpus),
                    "--v6-source",
                    f"v6_training={sources['v6_training']}",
                    "--v6-source",
                    f"v6_evaluation={sources['v6_evaluation']}",
                    "--git-commit",
                    "a" * 40,
                    "--output",
                    str(output),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "sealed\n")


if __name__ == "__main__":
    unittest.main()
