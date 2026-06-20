import tomllib
import unittest
from pathlib import Path


class ProjectMetadataTest(unittest.TestCase):
    def test_runtime_dependencies_cover_application_imports(self):
        root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = metadata["project"]["dependencies"]
        names = {requirement.split(">=", 1)[0].lower() for requirement in dependencies}

        self.assertEqual(names, {"pyside6", "numpy", "torch", "torchvision"})


if __name__ == "__main__":
    unittest.main()
