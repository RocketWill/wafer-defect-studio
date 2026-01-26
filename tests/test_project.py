from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wafer_defect_studio import project


class ProjectTest(unittest.TestCase):
    def test_create_reopen_and_refuse_non_empty_foreign_folder(self):
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "new-project"

            created = project.create_project(project_path)

            self.assertEqual(created.path, project_path.resolve())
            self.assertEqual(created.schema_version, 4)
            self.assertTrue(created.project_id)
            self.assertEqual(
                {entry.name for entry in project_path.iterdir()},
                {"project.sqlite", "models", "runs", "exports", "backups", "cache"},
            )

            reopened = project.open_project(str(project_path))
            self.assertEqual(reopened, created)

            foreign_path = workspace / "foreign"
            foreign_path.mkdir()
            sentinel = foreign_path / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaises(project.ProjectError):
                project.create_project(foreign_path)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertEqual([entry.name for entry in foreign_path.iterdir()], ["sentinel.txt"])


if __name__ == "__main__":
    unittest.main()
