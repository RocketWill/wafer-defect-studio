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
from wafer_defect_studio.proposal_review import (
    ProposalReviewError,
    latest_proposal_review,
    load_proposal_revisions,
    record_review,
)
from wafer_defect_studio.proposal_store import (
    load_defect_proposal,
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


class ProposalReviewTest(unittest.TestCase):
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
            connection.execute(
                "INSERT INTO grid_annotations VALUES (?, ?, ?, ?)",
                ("image-1", 0, 0, '["scratch"]'),
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

    def test_append_only_review_revisions_preserve_proposal_and_annotations(self):
        with TemporaryDirectory() as temporary_directory:
            project_path, detection_run_id, profile_id = self._project_with_detection_run(
                Path(temporary_directory)
            )
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
                    "image_asset_id": "image-1",
                },
            )
            save_defect_proposal(
                project_path,
                proposal,
                detection_run_id=detection_run_id,
                profile_id=profile_id,
            )

            self.assertEqual(project.open_project(project_path).schema_version, 16)
            initial = latest_proposal_review(project_path, proposal.proposal_id)
            self.assertEqual((initial.revision_number, initial.status, initial.source_rect), (0, "unreviewed", proposal.source_rect))

            accepted = record_review(project_path, proposal.proposal_id, "accepted", actor="engineer")
            rejected = record_review(project_path, proposal.proposal_id, "rejected", actor="engineer")
            corrected = record_review(
                project_path,
                proposal.proposal_id,
                "corrected",
                source_rect=Rect(10, 12, 8, 7),
                provenance={"manual_correction": True},
                actor="engineer",
            )

            self.assertEqual([item.revision_number for item in (accepted, rejected, corrected)], [1, 2, 3])
            self.assertEqual(corrected.status, "corrected")
            self.assertEqual(corrected.source_rect, Rect(10, 12, 8, 7))
            self.assertEqual(corrected.provenance["source_proposal_id"], proposal.proposal_id)
            self.assertTrue(corrected.provenance["manual_correction"])
            self.assertEqual(latest_proposal_review(project_path, proposal.proposal_id), corrected)
            self.assertEqual(load_proposal_revisions(project_path, proposal.proposal_id), (accepted, rejected, corrected))

            self.assertEqual(load_defect_proposal(project_path, proposal.proposal_id), proposal)
            connection = sqlite3.connect(project_path / "project.sqlite")
            try:
                annotations_before = connection.execute(
                    "SELECT image_asset_id, row, column, class_codes_json FROM grid_annotations"
                ).fetchall()
                self.assertEqual(project.open_project(project_path).schema_version, 17)
                with self.assertRaises(sqlite3.DatabaseError):
                    connection.execute(
                        "UPDATE proposal_review_revisions SET status = 'accepted' "
                        "WHERE proposal_id = ? AND revision_number = 1",
                        (proposal.proposal_id,),
                    )
                with self.assertRaises(sqlite3.DatabaseError):
                    connection.execute(
                        "DELETE FROM proposal_review_revisions WHERE proposal_id = ?",
                        (proposal.proposal_id,),
                    )
                annotations_after = connection.execute(
                    "SELECT image_asset_id, row, column, class_codes_json FROM grid_annotations"
                ).fetchall()
            finally:
                connection.close()
            self.assertEqual(annotations_after, annotations_before)

            with self.assertRaises(ProposalReviewError):
                record_review(project_path, proposal.proposal_id, "unknown")
            with self.assertRaises(ProposalReviewError):
                record_review(project_path, proposal.proposal_id, "corrected")
            with self.assertRaises(ProposalReviewError):
                record_review(
                    project_path,
                    proposal.proposal_id,
                    "corrected",
                    source_rect=proposal.source_rect,
                    provenance={"manual_correction": True},
                )

            second = DefectProposal(
                proposal_id="proposal-2",
                class_name="scratch",
                source_rect=Rect(1, 2, 3, 4),
                area=5,
                peak_confidence=0.8,
                mean_confidence=0.7,
                provenance={
                    "detection_run_id": detection_run_id,
                    "profile_id": profile_id,
                    "image_asset_id": "image-1",
                },
            )
            self.assertEqual(
                save_defect_proposal(
                    project_path,
                    second,
                    detection_run_id=detection_run_id,
                    profile_id=profile_id,
                ),
                second,
            )


if __name__ == "__main__":
    unittest.main()
