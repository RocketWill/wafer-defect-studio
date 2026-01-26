import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wafer_defect_studio import project
from wafer_defect_studio.grid_profile import (
    GridProfile,
    GridProfileConflictError,
    load_grid_profiles,
    save_grid_profile,
)


class GridProfileTest(unittest.TestCase):
    def test_append_only_revision_and_v3_first_save_migration(self):
        with TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project"
            project.create_project(project_path)
            database_path = project_path / "project.sqlite"

            connection = sqlite3.connect(database_path)
            try:
                connection.execute("DROP TABLE grid_profiles")
                connection.execute("UPDATE project_metadata SET schema_version = 3")
                connection.execute("PRAGMA user_version = 3")
                connection.commit()
            finally:
                connection.close()

            before_read = database_path.read_bytes()
            self.assertEqual(project.open_project(project_path).schema_version, 3)
            self.assertEqual(load_grid_profiles(project_path), ())
            self.assertEqual(database_path.read_bytes(), before_read)

            first = save_grid_profile(project_path, 10, 6)
            self.assertIsInstance(first, GridProfile)
            self.assertEqual((first.version, first.cell_width, first.cell_height), (1, 10, 6))
            self.assertEqual(project.open_project(project_path).schema_version, 4)
            self.assertEqual(load_grid_profiles(project_path), (first,))

            second = save_grid_profile(project_path, 20, 12, previous=first)
            self.assertEqual(second.grid_profile_id, first.grid_profile_id)
            self.assertEqual((second.version, second.cell_width, second.cell_height), (2, 20, 12))
            self.assertEqual(load_grid_profiles(project_path), (first, second))

            with self.assertRaises(ValueError):
                save_grid_profile(project_path, 20, 12, previous=second)
            with self.assertRaises(GridProfileConflictError):
                save_grid_profile(project_path, 30, 18, previous=first)

            connection = sqlite3.connect(database_path)
            try:
                columns = tuple(row[1] for row in connection.execute("PRAGMA table_info(grid_profiles)"))
                rows = connection.execute(
                    "SELECT grid_profile_id, version, cell_width, cell_height "
                    "FROM grid_profiles ORDER BY grid_profile_id, version"
                ).fetchall()
            finally:
                connection.close()
            self.assertEqual(columns, ("grid_profile_id", "version", "cell_width", "cell_height"))
            self.assertEqual(rows, [(first.grid_profile_id, 1, 10, 6), (second.grid_profile_id, 2, 20, 12)])


if __name__ == "__main__":
    unittest.main()
