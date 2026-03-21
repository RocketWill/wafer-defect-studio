import hashlib
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wafer_defect_studio import project
from wafer_defect_studio.defect_class import DefectClass, save_defect_classes
from wafer_defect_studio.training_scope import (
    DataGroup,
    DataGroupAssignment,
    TrainingScope,
    assign_image_to_data_group,
    eligible_image_ids,
    load_data_groups,
    load_image_data_group_assignments,
    load_training_scope,
    save_data_groups,
    save_training_scope,
)


class TrainingScopeTest(unittest.TestCase):
    def test_data_groups_and_assignments_read_deterministically_without_legacy_writes(self):
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "project"
            project.create_project(project_path)
            database = project_path / "project.sqlite"
            legacy_bytes = database.read_bytes()

            self.assertEqual(load_data_groups(project_path), ())
            self.assertEqual(load_image_data_group_assignments(project_path), ())
            self.assertEqual(database.read_bytes(), legacy_bytes)
            self.assertEqual(project.open_project(project_path).schema_version, 6)

            save_defect_classes(
                project_path,
                (DefectClass("scratch", "Scratch", "#cc4444"),),
            )
            _seed_reviewed_images(project_path, workspace)
            save_data_groups(
                project_path,
                (DataGroup("line-b", "Line B", 1), DataGroup("line-a", "Line A", 0)),
            )
            assign_image_to_data_group(project_path, "available", "line-b")
            assign_image_to_data_group(project_path, "changed", "line-a")

            self.assertEqual(
                load_data_groups(project_path),
                (DataGroup("line-a", "Line A", 0), DataGroup("line-b", "Line B", 1)),
            )
            self.assertEqual(
                load_image_data_group_assignments(project_path),
                (
                    DataGroupAssignment("changed", "line-a"),
                    DataGroupAssignment("available", "line-b"),
                ),
            )

    def test_scope_round_trips_and_only_reviewed_unchanged_sources_are_eligible(self):
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "project"
            project.create_project(project_path)
            save_defect_classes(
                project_path,
                (
                    DefectClass("scratch", "Scratch", "#cc4444", order=0),
                    DefectClass("chip", "Chip", "#4488cc", order=1),
                    DefectClass("archived", "Archived", "#777777", order=2, enabled=False),
                ),
            )
            _seed_reviewed_images(project_path, workspace)

            save_data_groups(
                project_path,
                (DataGroup("line-b", "Line B", 1), DataGroup("line-a", "Line A", 0)),
            )
            for image_id in ("available", "changed", "unreviewed"):
                assign_image_to_data_group(project_path, image_id, "line-a")
            assign_image_to_data_group(project_path, "other-group", "line-b")

            save_training_scope(
                project_path,
                TrainingScope(("line-a",), ("chip", "scratch")),
            )

            self.assertEqual(
                load_training_scope(project_path),
                TrainingScope(("line-a",), ("scratch", "chip")),
            )
            self.assertEqual(eligible_image_ids(project_path), ("available",))
            with self.assertRaises(ValueError):
                save_training_scope(
                    project_path,
                    TrainingScope(("line-a",), ("archived",)),
                )


def _seed_reviewed_images(project_path: Path, workspace: Path) -> None:
    rows = []
    for image_id in ("available", "changed", "unreviewed", "other-group"):
        source = workspace / f"{image_id}.tif"
        source.write_bytes(image_id.encode("ascii"))
        rows.append(
            (
                image_id,
                str(source.resolve()),
                1,
                1,
                "uint8",
                "TIFF",
                hashlib.sha256(source.read_bytes()).hexdigest(),
                0,
            )
        )

    database_path = project_path / "project.sqlite"
    connection = sqlite3.connect(database_path)
    try:
        connection.executemany(
            "INSERT INTO image_assets "
            "(image_asset_id, path, width, height, dtype, format, fingerprint, lossy_source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.execute(
            "CREATE TABLE grid_annotations ("
            "image_asset_id TEXT NOT NULL, row INTEGER NOT NULL, column INTEGER NOT NULL, "
            "class_codes_json TEXT NOT NULL, PRIMARY KEY (image_asset_id, row, column))"
        )
        connection.execute(
            "CREATE TABLE image_reviews ("
            "image_asset_id TEXT NOT NULL PRIMARY KEY, reviewed INTEGER NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO image_reviews VALUES (?, ?)",
            (("available", 1), ("changed", 1), ("unreviewed", 0), ("other-group", 1)),
        )
        connection.execute("UPDATE project_metadata SET schema_version = 9")
        connection.execute("PRAGMA user_version = 9")
        connection.commit()
    finally:
        connection.close()
    (workspace / "changed.tif").write_bytes(b"changed after registration")


if __name__ == "__main__":
    unittest.main()
