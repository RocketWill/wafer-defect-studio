import json
import unittest
from pathlib import Path


class PublicReadmeTest(unittest.TestCase):
    def test_readme_presents_current_public_product_state(self):
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")
        report = json.loads(
            (root / "docs/demo/ticket38-core-union-gate.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(report["overall"], "PASS")
        self.assertEqual(report["recommendation"], "continue_development")
        for statement in (
            "CAM v2 is the default",
            "Spatial MIL remains experimental",
            "docs/demo/ticket38-core-union-gate.json",
            "continue_development",
        ):
            self.assertIn(statement, readme)

        for local_reference in (".scratch", "docs/agents", "E:\\", "AGENTS.md"):
            self.assertNotIn(local_reference, readme)

        architecture_svg = "docs/diagrams/wafer-defect-studio-architecture.svg"
        architecture_source = "docs/diagrams/wafer-defect-studio-architecture.html"
        self.assertIn(f"]({architecture_svg})", readme)
        self.assertIn(f"]({architecture_source})", readme)
        self.assertTrue((root / architecture_svg).is_file())
        self.assertTrue((root / architecture_source).is_file())


if __name__ == "__main__":
    unittest.main()
