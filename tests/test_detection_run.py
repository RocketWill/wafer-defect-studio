import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from wafer_defect_studio import project
from wafer_defect_studio.detection_run import (
    DetectionRunError,
    create_detection_profile,
    create_detection_run,
    load_detection_profile,
    load_detection_run,
)
from wafer_defect_studio.evaluation_run import (
    create_evaluation,
    load_evaluation,
    record_evaluation_decision,
)
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
from wafer_defect_studio.training_run import RunConfig, create_training_run


class DetectionRunTest(unittest.TestCase):
    def _project_with_approved_evaluation(self, root: Path) -> Path:
        project_path = root / "project"
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
                "INSERT INTO image_assets "
                "(image_asset_id, path, width, height, dtype, format, fingerprint) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("image-1", "wafer.tif", 128, 96, "uint8", "TIFF", "fp-1"),
            )
            connection.execute("UPDATE project_metadata SET schema_version = 12")
            connection.execute("PRAGMA user_version = 12")
            connection.commit()
        finally:
            connection.close()

        training = create_training_run(
            project_path,
            RunConfig("snapshot-1", "split-1", class_count=1),
            run_id="run-1",
        )
        evaluation = create_evaluation(
            project_path,
            training.run_id,
            {"macro_f1": 0.9},
            {"target_satisfied": True},
            {"minimum_recall_target": 0.95},
            evaluation_id="evaluation-1",
        )
        record_evaluation_decision(
            project_path,
            evaluation.evaluation_id,
            "validated",
            actor="engineer",
            notes="validated",
        )
        record_evaluation_decision(
            project_path,
            evaluation.evaluation_id,
            "approved",
            actor="engineer",
            notes="approved for detection",
        )
        return project_path

    def test_approved_profile_run_is_immutable_and_source_guarded(self):
        with TemporaryDirectory() as temporary_directory:
            project_path = self._project_with_approved_evaluation(Path(temporary_directory))
            profile = create_detection_profile(
                project_path,
                evaluation_id="evaluation-1",
                window_size=(32, 24),
                stride=(16, 12),
                reflect_padding=True,
                thresholds={"scratch": 0.6},
                center_weighting="hann",
                map_generation={"smoothing": 3, "minimum_area": 4},
                profile_id="profile-1",
            )
            self.assertEqual(load_detection_profile(project_path, "profile-1"), profile)
            self.assertEqual(project.open_project(project_path).schema_version, 15)

            run = create_detection_run(
                project_path,
                profile_id=profile.profile_id,
                evaluation_id="evaluation-1",
                source_fingerprints={"image-1": "fp-1"},
                provenance={"coordinate_system": "source-image-pixels", "window_count": 8},
                run_id="detection-1",
            )
            self.assertEqual(load_detection_run(project_path, run.run_id), run)
            self.assertEqual(run.training_run_id, "run-1")
            self.assertEqual(run.status, "created")
            self.assertEqual(load_evaluation(project_path, "evaluation-1").training_run_id, "run-1")

            connection = sqlite3.connect(project_path / "project.sqlite")
            try:
                with self.assertRaises(sqlite3.DatabaseError):
                    connection.execute(
                        "UPDATE detection_profiles SET settings_json = '{}' WHERE profile_id = ?",
                        (profile.profile_id,),
                    )
                with self.assertRaises(sqlite3.DatabaseError):
                    connection.execute(
                        "DELETE FROM detection_runs WHERE detection_run_id = ?",
                        (run.run_id,),
                    )
            finally:
                connection.close()

            connection = sqlite3.connect(project_path / "project.sqlite")
            try:
                connection.execute(
                    "UPDATE image_assets SET fingerprint = 'fp-changed' WHERE image_asset_id = 'image-1'"
                )
                connection.commit()
            finally:
                connection.close()
            with self.assertRaises(DetectionRunError):
                create_detection_run(
                    project_path,
                    profile_id=profile.profile_id,
                    evaluation_id="evaluation-1",
                    source_fingerprints={"image-1": "fp-1"},
                    run_id="detection-2",
                )

    def test_unapproved_evaluation_cannot_create_profile(self):
        with TemporaryDirectory() as temporary_directory:
            project_path = self._project_with_approved_evaluation(Path(temporary_directory))
            evaluation = create_evaluation(
                project_path,
                "run-1",
                {"macro_f1": 0.2},
                {"target_satisfied": False},
                {},
                evaluation_id="evaluation-2",
            )
            with self.assertRaises(DetectionRunError):
                create_detection_profile(
                    project_path,
                    evaluation_id=evaluation.evaluation_id,
                    window_size=(32, 32),
                    stride=(16, 16),
                    reflect_padding=True,
                    thresholds={"scratch": 0.5},
                    profile_id="profile-unapproved",
                )


if __name__ == "__main__":
    unittest.main()
