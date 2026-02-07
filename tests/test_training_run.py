import hashlib
import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from wafer_defect_studio import project
from wafer_defect_studio.defect_class import load_defect_classes
from wafer_defect_studio.project import (
    _DATA_GROUPS_TABLE_SQL,
    _DATASET_SNAPSHOT_COLUMNS,
    _DATASET_SNAPSHOTS_TABLE_SQL,
    _DATASET_SPLIT_COLUMNS,
    _DATASET_SPLITS_TABLE_SQL,
    _DEFECT_CLASSES_TABLE_SQL,
    _EFFECTIVE_WAFER_AREAS_TABLE_SQL,
    _GRID_ANNOTATIONS_TABLE_SQL,
    _GRID_PROFILES_TABLE_SQL,
    _IMAGE_DATA_GROUPS_TABLE_SQL,
    _IMAGE_GRID_PLACEMENTS_TABLE_SQL,
    _IMAGE_REVIEWS_TABLE_SQL,
    _TRAINING_SCOPE_TABLE_SQL,
)
from wafer_defect_studio.training_run import (
    RunConfig,
    TrainingRunError,
    create_training_run,
    load_training_run,
    update_training_run_terminal,
)


class TrainingRunTest(unittest.TestCase):
    def test_persists_immutable_run_and_publishes_only_validated_stage(self):
        with TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project"
            project.create_project(project_path)
            database_path = project_path / "project.sqlite"
            connection = sqlite3.connect(database_path)
            try:
                for statement in (
                    _GRID_PROFILES_TABLE_SQL,
                    _IMAGE_GRID_PLACEMENTS_TABLE_SQL,
                    _EFFECTIVE_WAFER_AREAS_TABLE_SQL,
                    _DEFECT_CLASSES_TABLE_SQL,
                    _GRID_ANNOTATIONS_TABLE_SQL,
                    _IMAGE_REVIEWS_TABLE_SQL,
                    _DATA_GROUPS_TABLE_SQL,
                    _IMAGE_DATA_GROUPS_TABLE_SQL,
                    _TRAINING_SCOPE_TABLE_SQL,
                    _DATASET_SNAPSHOTS_TABLE_SQL,
                    _DATASET_SPLITS_TABLE_SQL,
                ):
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO dataset_snapshots VALUES (?, ?, ?)",
                    ("snapshot-1", "2026-01-01T00:00:00+00:00", "{}"),
                )
                connection.execute(
                    "INSERT INTO dataset_splits VALUES (?, ?, ?, ?)",
                    ("split-1", "snapshot-1", 7, "{}"),
                )
                connection.execute(
                    "INSERT INTO defect_classes VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("particle", "Particle", "#ffffff", "", "", 0, 1),
                )
                connection.execute(
                    "UPDATE project_metadata SET schema_version = 12"
                )
                connection.execute("PRAGMA user_version = 12")
                connection.commit()
            finally:
                connection.close()

            config = RunConfig(
                snapshot_id="snapshot-1",
                split_id="split-1",
                class_count=2,
                batch_size=8,
                device="cuda",
                augmentation={"rotation_degrees": 5},
            )
            run = create_training_run(
                project_path,
                config,
                parent_run_id="parent-0",
                environment={"torch": "2.8", "cuda": "12.8"},
                run_id="run-1",
            )
            self.assertEqual(project.open_project(project_path).schema_version, 13)
            self.assertEqual(load_defect_classes(project_path)[0].code, "particle")
            self.assertEqual(run.config, config)
            self.assertEqual(run.parent_run_id, "parent-0")
            self.assertEqual(run.status, "created")
            self.assertTrue(run.staging_path.is_dir())

            with self.assertRaises(sqlite3.DatabaseError):
                connection = sqlite3.connect(database_path)
                try:
                    connection.execute(
                        "UPDATE training_runs SET config_json = '{}' WHERE run_id = 'run-1'"
                    )
                finally:
                    connection.close()
            with self.assertRaises(sqlite3.DatabaseError):
                connection = sqlite3.connect(database_path)
                try:
                    connection.execute("DELETE FROM training_runs WHERE run_id = 'run-1'")
                finally:
                    connection.close()

            model = run.staging_path / "model.pt"
            model.write_bytes(b"weights")
            (run.staging_path / "manifest.json").write_text(
                json.dumps(
                    {
                        "required_files": ["model.pt"],
                        "files": [
                            {
                                "path": "model.pt",
                                "sha256": hashlib.sha256(b"weights").hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            completed = update_training_run_terminal(
                project_path,
                "run-1",
                "completed",
                metrics={"loss": 0.25},
                log="finished",
            )
            self.assertEqual(completed.status, "completed")
            self.assertEqual(completed.metrics, {"loss": 0.25})
            self.assertEqual(completed.log, "finished")
            self.assertIsNotNone(completed.artifact_path)
            self.assertTrue(completed.artifact_path.is_dir())
            self.assertFalse(run.staging_path.exists())
            self.assertEqual(load_training_run(project_path, "run-1"), completed)

            invalid = create_training_run(
                project_path,
                config,
                run_id="run-2",
            )
            (invalid.staging_path / "manifest.json").write_text(
                json.dumps({"required_files": ["model.pt"], "files": []}),
                encoding="utf-8",
            )
            with self.assertRaises(TrainingRunError):
                update_training_run_terminal(project_path, "run-2", "completed")
            self.assertFalse((project_path / "runs" / "run-2").exists())
            self.assertEqual(load_training_run(project_path, "run-2").status, "created")


if __name__ == "__main__":
    unittest.main()
