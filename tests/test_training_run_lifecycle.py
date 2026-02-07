import hashlib
import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from wafer_defect_studio import project
from wafer_defect_studio.project import (
    _DATA_GROUPS_TABLE_SQL,
    _DATASET_SNAPSHOTS_TABLE_SQL,
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
    clone_after_oom,
    create_training_run,
    load_training_run,
    resume_training_run,
    update_training_run_terminal,
)


class TrainingRunLifecycleTest(unittest.TestCase):
    def test_resume_and_oom_clone_create_auditable_children(self):
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
                epochs=1,
                device="cpu",
            )
            parent = create_training_run(project_path, config, run_id="parent-run")
            checkpoint = parent.staging_path / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            manifest = {
                "required_files": ["model.pt"],
                "files": [
                    {
                        "path": "model.pt",
                        "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
                    }
                ],
            }
            (parent.staging_path / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            completed = update_training_run_terminal(
                project_path, parent.run_id, "completed", message="done"
            )

            resumed = resume_training_run(
                project_path,
                completed.run_id,
                new_run_id="resumed-run",
                config_overrides={"epochs": 2},
            )
            self.assertEqual(load_training_run(project_path, parent.run_id), completed)
            self.assertEqual(resumed.parent_run_id, parent.run_id)
            self.assertEqual(resumed.config.snapshot_id, config.snapshot_id)
            self.assertEqual(resumed.config.split_id, config.split_id)
            self.assertEqual(resumed.config.class_count, config.class_count)
            self.assertEqual(resumed.config.epochs, 2)
            self.assertNotEqual(resumed.run_id, parent.run_id)
            self.assertTrue(resumed.staging_path.is_dir())

            oom_parent = create_training_run(
                project_path, config, run_id="oom-parent"
            )
            failed = update_training_run_terminal(
                project_path,
                oom_parent.run_id,
                "failed",
                message="Out of memory at requested batch_size=8",
                error_code="out_of_memory",
            )
            clone = clone_after_oom(
                project_path,
                failed.run_id,
                batch_size=2,
                new_run_id="oom-clone",
            )
            self.assertEqual(load_training_run(project_path, failed.run_id), failed)
            self.assertEqual(clone.parent_run_id, failed.run_id)
            self.assertEqual(clone.config.batch_size, 2)
            self.assertEqual(clone.config.snapshot_id, config.snapshot_id)
            self.assertEqual(clone.config.split_id, config.split_id)
            self.assertEqual(clone.config.class_count, config.class_count)
            self.assertTrue(clone.staging_path.is_dir())


if __name__ == "__main__":
    unittest.main()
