import hashlib
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wafer_defect_studio import project
from wafer_defect_studio.annotation import GridAnnotation, save_grid_annotation
from wafer_defect_studio.dataset_snapshot import (
    DatasetSnapshotError,
    SamplingPolicy,
    SnapshotSample,
    create_dataset_snapshot,
    load_dataset_snapshot,
)
from wafer_defect_studio.defect_class import DefectClass, save_defect_classes
from wafer_defect_studio.normalization import NormalizationBounds
from wafer_defect_studio.training_scope import (
    DataGroup,
    TrainingScope,
    assign_image_to_data_group,
    save_data_groups,
    save_training_scope,
)


class DatasetSnapshotTest(unittest.TestCase):
    def test_snapshot_freezes_eligible_inputs_and_rejects_unreviewed_scope(self):
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "project"
            project.create_project(project_path)
            source = workspace / "wafer.tif"
            source.write_bytes(b"native source pixels")
            _seed_reviewed_image(project_path, source)
            save_defect_classes(
                project_path,
                (DefectClass("scratch", "Scratch", "#cc4444"),),
            )
            save_data_groups(project_path, (DataGroup("line-a", "Line A"),))
            assign_image_to_data_group(project_path, "wafer-1", "line-a")
            save_training_scope(
                project_path, TrainingScope(("line-a",), ("scratch",))
            )
            bounds = (
                NormalizationBounds("uint8", 0, 255, 10.0, 240.0, 1.0, 99.0),
            )

            created = create_dataset_snapshot(
                project_path, bounds, SamplingPolicy(normal_to_positive_ratio=1.0)
            )
            frozen = load_dataset_snapshot(project_path, created.snapshot_id)

            database = project_path / "project.sqlite"
            save_defect_classes(
                project_path,
                (DefectClass("scratch", "Renamed", "#cc4444"),),
            )
            save_grid_annotation(project_path, GridAnnotation("wafer-1", 0, 0, ()))
            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "UPDATE image_assets SET fingerprint = 'changed' "
                    "WHERE image_asset_id = 'wafer-1'"
                )
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        "UPDATE dataset_snapshots SET payload_json = '{}' WHERE snapshot_id = ?",
                        (created.snapshot_id,),
                    )
                connection.commit()
            finally:
                connection.close()

            self.assertEqual(load_dataset_snapshot(project_path, created.snapshot_id), frozen)
            self.assertEqual(frozen.scope, TrainingScope(("line-a",), ("scratch",)))
            self.assertEqual(frozen.classes[0].name, "Scratch")
            self.assertEqual(frozen.sources[0].fingerprint, hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(frozen.grid_versions[0].grid_profile_version, 1)
            self.assertTrue(frozen.annotation_versions[0].content_hash)
            self.assertEqual(
                frozen.samples,
                (
                    SnapshotSample("wafer-1", 0, 0, 0, 0, 2, 2, ("scratch",)),
                    SnapshotSample("wafer-1", 0, 1, 2, 0, 2, 2, ()),
                ),
            )
            self.assertEqual(frozen.normalization_bounds, bounds)
            self.assertEqual(frozen.sampling_policy, SamplingPolicy(1.0))

            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "UPDATE image_assets SET fingerprint = ? WHERE image_asset_id = 'wafer-1'",
                    (hashlib.sha256(source.read_bytes()).hexdigest(),),
                )
                connection.commit()
            finally:
                connection.close()
            second = create_dataset_snapshot(project_path, bounds, SamplingPolicy(1.0))
            self.assertNotEqual(second.snapshot_id, created.snapshot_id)

            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "UPDATE image_reviews SET reviewed = 0 WHERE image_asset_id = 'wafer-1'"
                )
                connection.commit()
            finally:
                connection.close()
            with self.assertRaises(DatasetSnapshotError):
                create_dataset_snapshot(project_path, bounds, SamplingPolicy(1.0))


def _seed_reviewed_image(project_path: Path, source: Path) -> None:
    connection = sqlite3.connect(project_path / "project.sqlite")
    try:
        connection.execute(
            "INSERT INTO image_assets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "wafer-1",
                str(source.resolve()),
                6,
                2,
                "uint8",
                "TIFF",
                hashlib.sha256(source.read_bytes()).hexdigest(),
                0,
            ),
        )
        connection.execute(
            "INSERT INTO grid_profiles VALUES ('grid', 1, 2, 2)"
        )
        connection.execute(
            "INSERT INTO image_grid_placements VALUES ('wafer-1', 'grid', 1, 0, 0)"
        )
        connection.execute(
            "CREATE TABLE defect_classes (code TEXT PRIMARY KEY, name TEXT NOT NULL, "
            "color TEXT NOT NULL, icon TEXT NOT NULL DEFAULT '', description TEXT NOT NULL "
            "DEFAULT '', display_order INTEGER NOT NULL, enabled INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE grid_annotations (image_asset_id TEXT NOT NULL, row INTEGER NOT NULL, "
            "column INTEGER NOT NULL, class_codes_json TEXT NOT NULL, "
            "PRIMARY KEY (image_asset_id, row, column))"
        )
        connection.execute(
            "INSERT INTO grid_annotations VALUES ('wafer-1', 0, 0, '[\"scratch\"]')"
        )
        connection.execute(
            "INSERT INTO effective_wafer_areas VALUES "
            "('wafer-1', 'polygon', '{\"vertices\":[[0,0],[4,0],[4,2],[0,2]]}', 1)"
        )
        connection.execute(
            "CREATE TABLE image_reviews (image_asset_id TEXT PRIMARY KEY, reviewed INTEGER NOT NULL)"
        )
        connection.execute("INSERT INTO image_reviews VALUES ('wafer-1', 1)")
        connection.execute("UPDATE project_metadata SET schema_version = 9")
        connection.execute("PRAGMA user_version = 9")
        connection.commit()
    finally:
        connection.close()


if __name__ == "__main__":
    unittest.main()
