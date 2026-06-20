import unittest
from pathlib import Path


class PublicDemoDocsTest(unittest.TestCase):
    def test_demo_docs_are_portable_and_index_current_evidence(self):
        root = Path(__file__).resolve().parents[1]
        index = (root / "docs/demo/README.md").read_text(encoding="utf-8")
        tutorial = (
            root / "docs/demo/phase2-end-to-end-tutorial.md"
        ).read_text(encoding="utf-8")

        for statement in (
            "CAM v2 remains the default",
            "Spatial MIL remains experimental",
            "ticket30-quality-gate.json",
            "ticket34-final-gate.json",
            "ticket38-core-union-gate.json",
            "continue_development",
        ):
            self.assertIn(statement, index)

        self.assertIn("python docs/demo/run_phase2_demo.py", tutorial)
        for document in (index, tutorial):
            self.assertNotRegex(document, "[\\u4e00-\\u9fff]")
            for local_reference in (".scratch", "E:\\", "docs/agents"):
                self.assertNotIn(local_reference, document)


if __name__ == "__main__":
    unittest.main()
