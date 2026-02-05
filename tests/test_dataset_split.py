import hashlib
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from dataclasses import replace

from wafer_defect_studio import project
from wafer_defect_studio.dataset_snapshot import DatasetSnapshot, SamplingPolicy, SnapshotSource
from wafer_defect_studio.dataset_split import create_dataset_split, load_dataset_split
from wafer_defect_studio.training_scope import TrainingScope


class DatasetSplitTest(unittest.TestCase):
    def test_seeded_image_split_is_deterministic_unique_and_immutable(self):
        with TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project"
            info = project.create_project(project_path)
            image_ids = tuple(f"wafer-{index:02d}" for index in range(12))
            snapshot = DatasetSnapshot(
                "snapshot-1",
                "2026-08-22T00:00:00+00:00",
                TrainingScope(("line-a",), ("scratch",)),
                (),
                tuple(SnapshotSource(image_id, "line-a", "fingerprint") for image_id in image_ids),
                (),
                (),
                (),
                SamplingPolicy(),
            )
            database = project_path / "project.sqlite"
            connection = sqlite3.connect(database)
            try:
                _upgrade_to_snapshot_schema(connection, info.project_id)
                connection.execute(
                    "INSERT INTO dataset_snapshots VALUES (?, ?, ?)",
                    (snapshot.snapshot_id, snapshot.created_at, _snapshot_json(snapshot)),
                )
                connection.commit()
            finally:
                connection.close()

            first = create_dataset_split(project_path, snapshot.snapshot_id, 42)
            repeated = create_dataset_split(project_path, snapshot.snapshot_id, 42)
            loaded = load_dataset_split(project_path, first.split_id)

            memberships = first.train_image_ids + first.validation_image_ids + first.test_image_ids
            self.assertEqual(first, repeated)
            self.assertEqual(loaded, first)
            self.assertEqual(set(memberships), set(image_ids))
            self.assertEqual(len(memberships), len(set(memberships)))
            self.assertEqual(tuple(map(len, (first.train_image_ids, first.validation_image_ids, first.test_image_ids))), (10, 1, 1))

            tiny_snapshot = replace(
                snapshot,
                snapshot_id="snapshot-tiny",
                sources=snapshot.sources[:2],
            )
            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "INSERT INTO dataset_snapshots VALUES (?, ?, ?)",
                    (tiny_snapshot.snapshot_id, tiny_snapshot.created_at, _snapshot_json(tiny_snapshot)),
                )
                connection.commit()
            finally:
                connection.close()
            tiny = create_dataset_split(project_path, tiny_snapshot.snapshot_id, 42)
            self.assertEqual(
                tuple(map(len, (tiny.train_image_ids, tiny.validation_image_ids, tiny.test_image_ids))),
                (2, 0, 0),
            )

            connection = sqlite3.connect(database)
            try:
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        "UPDATE dataset_splits SET payload_json = '{}' WHERE split_id = ?",
                        (first.split_id,),
                    )
            finally:
                connection.close()


def _upgrade_to_snapshot_schema(connection: sqlite3.Connection, project_id: str) -> None:
    connection.executescript(
        "CREATE TABLE defect_classes (code TEXT PRIMARY KEY, name TEXT NOT NULL, color TEXT NOT NULL, icon TEXT NOT NULL, description TEXT NOT NULL, display_order INTEGER NOT NULL, enabled INTEGER NOT NULL);"
        "CREATE TABLE grid_annotations (image_asset_id TEXT NOT NULL, row INTEGER NOT NULL, column INTEGER NOT NULL, class_codes_json TEXT NOT NULL, PRIMARY KEY (image_asset_id, row, column));"
        "CREATE TABLE image_reviews (image_asset_id TEXT PRIMARY KEY, reviewed INTEGER NOT NULL);"
        "CREATE TABLE data_groups (data_group_id TEXT PRIMARY KEY, name TEXT NOT NULL, display_order INTEGER NOT NULL);"
        "CREATE TABLE image_data_groups (image_asset_id TEXT PRIMARY KEY, data_group_id TEXT NOT NULL);"
        "CREATE TABLE training_scope (singleton INTEGER PRIMARY KEY, data_group_ids_json TEXT NOT NULL, class_codes_json TEXT NOT NULL);"
        "CREATE TABLE dataset_snapshots (snapshot_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload_json TEXT NOT NULL);"
    )
    connection.execute("UPDATE project_metadata SET schema_version = 11 WHERE project_id = ?", (project_id,))
    connection.execute("PRAGMA user_version = 11")


def _snapshot_json(snapshot: DatasetSnapshot) -> str:
    import json
    from dataclasses import asdict

    return json.dumps(asdict(snapshot), sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    unittest.main()
