import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from wafer_defect_studio import project
from wafer_defect_studio.detection_run import (
    create_detection_profile,
    create_detection_run,
)
from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.evaluation_run import (
    create_evaluation,
    record_evaluation_decision,
)
from wafer_defect_studio.proposal_generation import DefectProposal
from wafer_defect_studio.proposal_store import (
    ProposalStoreError,
    load_defect_proposal,
    load_defect_proposals,
    save_defect_proposal,
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


class ProposalStoreTest(unittest.TestCase):
    def _project_with_detection_run(self, root: Path) -> tuple[Path, str, str]:
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
        profile = create_detection_profile(
            project_path,
            evaluation_id=evaluation.evaluation_id,
            window_size=(32, 24),
            stride=(16, 12),
            reflect_padding=True,
            thresholds={"scratch": 0.6},
            center_weighting="hann",
            map_generation={"smoothing": 3, "minimum_area": 4},
            profile_id="profile-1",
        )
        run = create_detection_run(
            project_path,
            profile_id=profile.profile_id,
            evaluation_id=evaluation.evaluation_id,
            source_fingerprints={"image-1": "fp-1"},
            provenance={
                "coordinate_system": "source-image-pixels",
                "window_count": 8,
            },
            run_id="detection-1",
        )
        return project_path, run.run_id, profile.profile_id

    def test_schema_migrates_and_proposals_round_trip_without_replacement(self):
        with TemporaryDirectory() as temporary_directory:
            project_path, detection_run_id, profile_id = self._project_with_detection_run(
                Path(temporary_directory)
            )
            self.assertEqual(project.open_project(project_path).schema_version, 15)
            proposal = DefectProposal(
                proposal_id="proposal-1",
                class_name="scratch",
                source_rect=Rect(11, 13, 5, 4),
                area=12,
                peak_confidence=0.91,
                mean_confidence=0.74,
                provenance={
                    "detection_run_id": detection_run_id,
                    "profile_id": profile_id,
                    "source_coordinate_system": "source-image-pixels",
                    "image_asset_id": "image-1",
                    "source_transform": {"padding": "reflect"},
                },
            )

            saved = save_defect_proposal(
                project_path,
                proposal,
                detection_run_id=detection_run_id,
                profile_id=profile_id,
            )

            self.assertEqual(saved, proposal)
            self.assertEqual(project.open_project(project_path).schema_version, 16)
            self.assertEqual(load_defect_proposal(project_path, proposal.proposal_id), proposal)
            self.assertEqual(
                load_defect_proposals(project_path, detection_run_id=detection_run_id),
                (proposal,),
            )

            with self.assertRaises(ProposalStoreError):
                save_defect_proposal(
                    project_path,
                    proposal,
                    detection_run_id=detection_run_id,
                    profile_id=profile_id,
                )

            connection = sqlite3.connect(project_path / "project.sqlite")
            try:
                with self.assertRaises(sqlite3.DatabaseError):
                    connection.execute(
                        "UPDATE defect_proposals SET class_name = 'particle' "
                        "WHERE proposal_id = ?",
                        (proposal.proposal_id,),
                    )
                with self.assertRaises(sqlite3.DatabaseError):
                    connection.execute(
                        "DELETE FROM defect_proposals WHERE proposal_id = ?",
                        (proposal.proposal_id,),
                    )
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
