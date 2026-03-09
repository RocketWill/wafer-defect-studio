import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wafer_defect_studio.environment import collect_training_environment


class TrainingEnvironmentTest(unittest.TestCase):
    def test_collect_training_environment_is_serializable_and_project_free(self):
        required = {
            "python",
            "pytorch",
            "torchvision",
            "cuda",
            "cuda_driver",
            "os",
            "gpu",
            "packages",
        }
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            before = tuple(workspace.iterdir())
            environment = collect_training_environment()
            after = tuple(workspace.iterdir())

        self.assertEqual(before, after)
        self.assertTrue(required.issubset(environment))
        json.dumps(environment)
        for key in required - {"packages"}:
            self.assertIsInstance(environment[key], str)
            self.assertTrue(environment[key])
        self.assertIsInstance(environment["packages"], dict)


if __name__ == "__main__":
    unittest.main()
