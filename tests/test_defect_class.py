import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wafer_defect_studio import project
from wafer_defect_studio.defect_class import (
    DefectClass,
    load_defect_classes,
    save_defect_classes,
)


class DefectClassTest(unittest.TestCase):
    def test_ordered_classes_round_trip_and_first_write_migration(self):
        with TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project"
            project.create_project(project_path)
            database_path = project_path / "project.sqlite"

            before_read = database_path.read_bytes()
            self.assertEqual(load_defect_classes(project_path), ())
            self.assertEqual(database_path.read_bytes(), before_read)
            self.assertEqual(project.open_project(project_path).schema_version, 6)

            archived = DefectClass(
                code="particle",
                name="Particle",
                color="#4488cc",
                icon="circle",
                description="Small contamination particle",
                order=0,
                enabled=False,
            )
            enabled = DefectClass(
                code="scratch",
                name="Scratch",
                color="#cc4444",
                icon="line",
                description="Linear surface mark",
                order=1,
                enabled=True,
            )

            save_defect_classes(project_path, (enabled, archived))

            self.assertEqual(project.open_project(project_path).schema_version, 7)
            self.assertEqual(load_defect_classes(project_path), (archived, enabled))

            with self.assertRaises(ValueError):
                save_defect_classes(project_path, (enabled, enabled))
            self.assertEqual(load_defect_classes(project_path), (archived, enabled))

            connection = sqlite3.connect(database_path)
            try:
                columns = tuple(row[1] for row in connection.execute("PRAGMA table_info(defect_classes)"))
            finally:
                connection.close()
            self.assertEqual(
                columns,
                (
                    "code",
                    "name",
                    "color",
                    "icon",
                    "description",
                    "display_order",
                    "enabled",
                ),
            )


if __name__ == "__main__":
    unittest.main()
