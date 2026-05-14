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
    def test_run_config_preserves_hard_negative_refinement_values(self):
        config = RunConfig(
            "snapshot-1", "split-1", training_policy="spatial_mil_v4",
            patch_size=32, patch_stride=16,
            priority_normal_bag_ids=["bag-2", "bag-1"],
            hard_negative_selection_sha256="AB" * 32,
        )
        self.assertEqual(config.priority_normal_bag_ids, ("bag-2", "bag-1"))
        self.assertEqual(config.hard_negative_selection_sha256, "ab" * 32)
        with self.assertRaisesRegex(TrainingRunError, "provided together"):
            RunConfig(
                "snapshot-1", "split-1", training_policy="spatial_mil_v4",
                patch_size=32, patch_stride=16,
                priority_normal_bag_ids=("bag-2",),
            )

    def test_run_config_requires_complete_patch_classification_geometry(self):
        legacy = RunConfig("snapshot-1", "split-1")
        self.assertEqual(legacy.training_policy, "legacy")
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
            ({"bag_pooling": "max"}, "provided with bag_pooling"),
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

    def test_spatial_mil_v4_config_and_checkpoint_metadata_are_frozen(self):
        config = RunConfig(
            "snapshot-1",
            "split-1",
            training_policy="spatial_mil_v4",
            patch_size=128,
            patch_stride=64,
        )
        self.assertEqual(config.model_config["training_policy"], "spatial_mil_v4")
        self.assertIsNone(config.model_config["bag_pooling"])
        with self.assertRaisesRegex(TrainingRunError, "requires patch geometry"):
            RunConfig("snapshot-1", "split-1", training_policy="spatial_mil_v4")
        with self.assertRaisesRegex(TrainingRunError, "bag_pooling must be null"):
            RunConfig(
                "snapshot-1",
                "split-1",
                training_policy="spatial_mil_v4",
                patch_size=128,
                patch_stride=64,
                bag_pooling="max",
            )

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.pt"
            checkpoint = {
                "checkpoint_format": "wafer_defect_studio.resnet18.v4",
                "architecture": "resnet18_spatial_logits",
                "feature_stride": 4,
                "training_policy": "spatial_mil_v4",
                "patch_size": 128,
                "patch_stride": 64,
                "loss_weights": {
                    "positive_spatial_mil": 1.0,
                    "absent_class_hard_negative": 1.0,
                    "overlap_consistency": 1.0,
                },
                "positive_class_weighting": {
                    "formula": "negative_bag_count / positive_bag_count",
                    "minimum": 1.0,
                    "maximum": 10.0,
                },
                "augmentation_policy": {
                    "name": "spatial_mil_v4_defect_preserving_affine",
                    "contrast": [0.9, 1.1],
                    "brightness": [-0.03, 0.03],
                    "seed": 7,
                    "seed_formula": "run_seed + epoch * 1_000_003 + bag_index",
                },
                "class_count": 1,
                "class_codes": ["scratch"],
                "normalization_bounds": [{
                    "dtype": "uint8", "source_min": 0, "source_max": 255,
                    "low": 0.0, "high": 255.0,
                    "low_percentile": 1.0, "high_percentile": 99.0,
                }],
                "input_size": {"width": 128, "height": 128},
                "state_dict": {"dummy": torch.zeros(1)},
            }
            torch.save(checkpoint, path)
            self.assertEqual(validate_project_checkpoint(path), checkpoint)
            refined = {
                **checkpoint,
                "hard_negative_refinement": {
                    "selection_sha256": "a" * 64,
                    "priority_normal_bag_ids": ["bag-2", "bag-1"],
                    "base_epochs": 3,
                    "refinement_epochs": 5,
                },
            }
            torch.save(refined, path)
            self.assertEqual(validate_project_checkpoint(path), refined)
            invalid_refinement = {
                **refined,
                "hard_negative_refinement": {
                    **refined["hard_negative_refinement"],
                    "refinement_epochs": 4,
                },
            }
            torch.save(invalid_refinement, path)
            with self.assertRaisesRegex(TrainingRunError, "hard_negative_refinement"):
                validate_project_checkpoint(path)
            refinement_tampering = (
                {**refined["hard_negative_refinement"], "extra": True},
                {**refined["hard_negative_refinement"], "selection_sha256": "z" * 64},
                {**refined["hard_negative_refinement"], "priority_normal_bag_ids": []},
                {**refined["hard_negative_refinement"], "priority_normal_bag_ids": ["bag", "bag"]},
                {**refined["hard_negative_refinement"], "base_epochs": True},
                {**refined["hard_negative_refinement"], "base_epochs": 0},
                {**refined["hard_negative_refinement"], "refinement_epochs": 4},
            )
            for refinement_value in refinement_tampering:
                torch.save(
                    {**refined, "hard_negative_refinement": refinement_value},
                    path,
                )
                with self.assertRaisesRegex(TrainingRunError, "hard_negative_refinement"):
                    validate_project_checkpoint(path)
            tampered = (
                ({"architecture": "resnet18"}, "architecture"),
                ({"feature_stride": 16}, "feature_stride"),
                ({"training_policy": "legacy"}, "training_policy"),
                ({"patch_size": 0}, "patch_size"),
                ({"patch_stride": 0}, "patch_stride"),
                ({"patch_size": 32, "patch_stride": 64}, "cannot exceed"),
            )
            for changes, message in tampered:
                invalid = {**checkpoint, **changes}
                torch.save(invalid, path)
                with self.assertRaisesRegex(TrainingRunError, message):
                    validate_project_checkpoint(path)
            for name in checkpoint["loss_weights"]:
                invalid = {**checkpoint, "loss_weights": {**checkpoint["loss_weights"], name: 0.5}}
                torch.save(invalid, path)
                with self.assertRaisesRegex(TrainingRunError, "loss_weights"):
                    validate_project_checkpoint(path)
            for name, value in (("formula", "P / N"), ("minimum", 0.0), ("maximum", 5.0)):
                invalid = {
                    **checkpoint,
                    "positive_class_weighting": {
                        **checkpoint["positive_class_weighting"],
                        name: value,
                    },
                }
                torch.save(invalid, path)
                with self.assertRaisesRegex(TrainingRunError, "positive_class_weighting"):
                    validate_project_checkpoint(path)
            for name, value in (("contrast", [0.8, 1.2]), ("brightness", [-0.1, 0.1]), ("seed", True)):
                invalid = {
                    **checkpoint,
                    "augmentation_policy": {**checkpoint["augmentation_policy"], name: value},
                }
                torch.save(invalid, path)
                with self.assertRaisesRegex(TrainingRunError, "augmentation_policy"):
                    validate_project_checkpoint(path)

    def test_load_old_persisted_config_defaults_training_policy_to_legacy(self):
        with TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project"
            _prepare_training_project(project_path)
            run = create_training_run(
                project_path,
                RunConfig("snapshot-1", "split-1"),
                run_id="old-config",
            )
            database_path = project_path / "project.sqlite"
            connection = sqlite3.connect(database_path)
            try:
                connection.execute("DROP TRIGGER training_runs_no_immutable_update")
                payload = json.loads(connection.execute(
                    "SELECT config_json FROM training_runs WHERE run_id = ?",
                    (run.run_id,),
                ).fetchone()[0])
                payload.pop("training_policy")
                connection.execute(
                    "UPDATE training_runs SET config_json = ? WHERE run_id = ?",
                    (json.dumps(payload), run.run_id),
                )
                connection.commit()
            finally:
                connection.close()
            self.assertEqual(
                load_training_run(project_path, run.run_id).config.training_policy,
                "legacy",
            )

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
