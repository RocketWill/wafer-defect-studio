import hashlib
import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

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
    validate_project_checkpoint,
)


class TrainingRunTest(unittest.TestCase):
    def test_run_config_requires_complete_patch_classification_geometry(self):
        legacy = RunConfig("snapshot-1", "split-1")
        self.assertEqual(
            (legacy.patch_size, legacy.patch_stride, legacy.bag_pooling),
            (None, None, None),
        )
        patch_config = RunConfig(
            "snapshot-1",
            "split-1",
            patch_size=128,
            patch_stride=64,
            bag_pooling="max",
        )
        self.assertEqual(
            (patch_config.patch_size, patch_config.patch_stride, patch_config.bag_pooling),
            (128, 64, "max"),
        )
        for values, message in (
            ({"patch_size": 128}, "provided together"),
            ({"patch_size": 128, "patch_stride": 64}, "bag_pooling must be max"),
            (
                {"patch_size": 128, "patch_stride": 129, "bag_pooling": "max"},
                "patch_stride cannot exceed patch_size",
            ),
            (
                {"patch_size": 128, "patch_stride": 64, "bag_pooling": "mean"},
                "bag_pooling must be max",
            ),
        ):
            with self.assertRaisesRegex(TrainingRunError, message):
                RunConfig("snapshot-1", "split-1", **values)

    def test_project_checkpoint_contract_rejects_incomplete_preprocessing(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.pt"
            _write_checkpoint(path, 1)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            del checkpoint["normalization_bounds"]
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(TrainingRunError, "incomplete"):
                validate_project_checkpoint(path)

    def test_completed_run_requires_complete_environment_before_publish(self):
        with TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project"
            _prepare_training_project(project_path)
            config = RunConfig(snapshot_id="snapshot-1", split_id="split-1")
            corrupt = create_training_run(
                project_path,
                config,
                run_id="run-corrupt",
            )
            _stage_valid_model(corrupt)
            connection = sqlite3.connect(project_path / "project.sqlite")
            try:
                connection.execute("DROP TRIGGER training_runs_no_immutable_update")
                connection.execute(
                    "UPDATE training_runs SET environment_json = '{}' "
                    "WHERE run_id = ?",
                    (corrupt.run_id,),
                )
                connection.execute(
                    "CREATE TRIGGER training_runs_no_immutable_update "
                    "BEFORE UPDATE OF run_id, created_at, snapshot_id, split_id, "
                    "parent_run_id, config_json, environment_json ON training_runs "
                    "BEGIN SELECT RAISE(ABORT, 'Training Run configuration is immutable'); END"
                )
                connection.commit()
            finally:
                connection.close()

            with self.assertRaisesRegex(TrainingRunError, "complete environment"):
                update_training_run_terminal(project_path, corrupt.run_id, "completed")
            self.assertTrue(corrupt.staging_path.is_dir())
            self.assertFalse((project_path / "runs" / corrupt.run_id).exists())

            valid = create_training_run(
                project_path,
                config,
                run_id="run-valid",
            )
            _stage_valid_model(valid)
            completed = update_training_run_terminal(
                project_path,
                valid.run_id,
                "completed",
            )
            self.assertEqual(completed.status, "completed")
            self.assertIsNotNone(completed.artifact_path)
            self.assertTrue(completed.artifact_path.is_dir())
            self.assertFalse(valid.staging_path.exists())

    def test_create_run_collects_and_merges_environment_provenance(self):
        with TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project"
            _prepare_training_project(project_path)
            config = RunConfig(snapshot_id="snapshot-1", split_id="split-1")
            collected = {
                "python": "3.11.15",
                "pytorch": "2.13.0",
                "torchvision": "0.28.0",
                "cuda": "unavailable",
                "cuda_driver": "unavailable",
                "os": "test-os",
                "gpu": "unavailable",
                "packages": {"numpy": "2.4.6", "torch": "2.13.0"},
            }
            overrides = {
                "python": "caller-python",
                "cuda": "caller-cuda",
                "packages": {"torch": "caller-torch", "custom": "1.0"},
                "custom": "caller-value",
            }

            with patch(
                "wafer_defect_studio.training_run.collect_training_environment",
                return_value=collected,
            ) as collector:
                run = create_training_run(
                    project_path,
                    config,
                    run_id="run-collected",
                )
                overridden = create_training_run(
                    project_path,
                    config,
                    environment=overrides,
                    run_id="run-overridden",
                )

            self.assertEqual(collector.call_count, 2)
            self.assertEqual(run.environment, collected)
            expected = dict(collected)
            expected.update(overrides)
            expected["packages"] = {
                **collected["packages"],
                **overrides["packages"],
            }
            self.assertEqual(overridden.environment, expected)
            self.assertEqual(
                load_training_run(project_path, "run-overridden").environment,
                expected,
            )

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
            _write_checkpoint(model, run.config.class_count)
            (run.staging_path / "manifest.json").write_text(
                json.dumps(
                    {
                        "required_files": ["model.pt"],
                        "files": [
                            {
                                "path": "model.pt",
                                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
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

def _prepare_training_project(project_path: Path) -> None:
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
            "INSERT INTO defect_classes VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("particle", "Particle", "#ffffff", "", "", 0, 1),
        )
        connection.execute(
            "INSERT INTO dataset_snapshots VALUES (?, ?, ?)",
            ("snapshot-1", "2026-01-01T00:00:00+00:00", "{}"),
        )
        connection.execute(
            "INSERT INTO dataset_splits VALUES (?, ?, ?, ?)",
            ("split-1", "snapshot-1", 7, "{}"),
        )
        connection.execute("UPDATE project_metadata SET schema_version = 12")
        connection.execute("PRAGMA user_version = 12")
        connection.commit()
    finally:
        connection.close()


def _stage_valid_model(run) -> None:
    model = run.staging_path / "model.pt"
    _write_checkpoint(model, run.config.class_count)
    (run.staging_path / "manifest.json").write_text(
        json.dumps(
            {
                "required_files": ["model.pt"],
                "files": [
                    {
                        "path": "model.pt",
                        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_checkpoint(path: Path, class_count: int) -> None:
    torch.save(
        {
            "checkpoint_format": "wafer_defect_studio.resnet18.v1",
            "architecture": "resnet18",
            "class_count": class_count,
            "class_codes": [f"class-{index}" for index in range(class_count)],
            "normalization_bounds": [
                {
                    "dtype": "uint8",
                    "source_min": 0,
                    "source_max": 255,
                    "low": 0.0,
                    "high": 255.0,
                    "low_percentile": 1.0,
                    "high_percentile": 99.0,
                }
            ],
            "input_size": {"width": 32, "height": 32},
            "state_dict": {"dummy": torch.zeros(1)},
        },
        path,
    )


if __name__ == "__main__":
    unittest.main()
