import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wafer_defect_studio import project
from wafer_defect_studio.annotation import (
    GridAnnotation,
    load_grid_annotation,
    save_grid_annotation,
)
from wafer_defect_studio.defect_class import (
    DefectClass,
    load_defect_classes,
    save_defect_classes,
)


class GridAnnotationTest(unittest.TestCase):
    def test_round_trip_replace_empty_and_reject_invalid_codes_atomically(self):
        with TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project"
            project.create_project(project_path)
            save_defect_classes(
                project_path,
                (
                    DefectClass("scratch", "Scratch", "#cc4444", order=0),
                    DefectClass("stain", "Stain", "#4488cc", order=1),
                    DefectClass("particle", "Particle", "#44aa66", order=2, enabled=False),
                ),
            )
            database_path = project_path / "project.sqlite"
            connection = sqlite3.connect(database_path)
            try:
                connection.execute(
                    "INSERT INTO image_assets "
                    "(image_asset_id, path, width, height, dtype, format, fingerprint) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("image-1", "source.tif", 100, 80, "uint8", "TIFF", "fingerprint"),
                )
                connection.commit()
            finally:
                connection.close()

            self.assertIsNone(load_grid_annotation(project_path, "image-1", -1, 2))
            self.assertEqual(project.open_project(project_path).schema_version, 7)

            original_classes = load_defect_classes(project_path)
            labels = GridAnnotation("image-1", -1, 2, ("scratch", "stain"))
            save_grid_annotation(project_path, labels)

            self.assertEqual(project.open_project(project_path).schema_version, 8)
            self.assertEqual(load_grid_annotation(project_path, "image-1", -1, 2), labels)
            self.assertEqual(load_defect_classes(project_path), original_classes)

            connection = sqlite3.connect(database_path)
            try:
                row = connection.execute(
                    "SELECT class_codes_json FROM grid_annotations "
                    "WHERE image_asset_id = ? AND row = ? AND column = ?",
                    ("image-1", -1, 2),
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(row, ((json.dumps(labels.class_codes, separators=(",", ":"))),))

            empty = GridAnnotation("image-1", -1, 2, ())
            save_grid_annotation(project_path, empty)
            self.assertEqual(load_grid_annotation(project_path, "image-1", -1, 2), empty)

            for invalid_codes in (("scratch", "scratch"), ("unknown",), ("particle",)):
                with self.assertRaises(ValueError):
                    save_grid_annotation(
                        project_path,
                        GridAnnotation("image-1", -1, 2, invalid_codes),
                    )
                self.assertEqual(load_grid_annotation(project_path, "image-1", -1, 2), empty)
                self.assertEqual(load_defect_classes(project_path), original_classes)


if __name__ == "__main__":
    unittest.main()
