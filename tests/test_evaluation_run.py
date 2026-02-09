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
from wafer_defect_studio.evaluation_run import (
    EvaluationRunError,
    create_evaluation,
    load_evaluation,
    load_evaluation_decisions,
    record_evaluation_decision,
)
from wafer_defect_studio.training_run import RunConfig, create_training_run, load_training_run


class EvaluationRunTest(unittest.TestCase):
    def test_evaluation_and_decisions_are_immutable_and_auditable(self):
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
                connection.execute("UPDATE project_metadata SET schema_version = 12")
                connection.execute("PRAGMA user_version = 12")
                connection.commit()
            finally:
                connection.close()

            run = create_training_run(
                project_path,
                RunConfig("snapshot-1", "split-1", class_count=1),
                run_id="run-1",
            )
            thresholds = {
                "per_class": [
                    {
                        "class_name": "scratch",
                        "policy": "min_recall",
                        "threshold": 0.9,
                        "target": 0.95,
                        "target_satisfied": False,
                    }
                ],
                "min_recall_target": 0.95,
            }
            evaluation = create_evaluation(
                project_path,
                run.run_id,
                {"macro_f1": 0.4},
                thresholds,
                {"minimum_recall_target": 0.95},
                environment={"torch": "2.8"},
                notes="test split",
                evaluation_id="evaluation-1",
            )
            self.assertFalse(evaluation.target_satisfied)
            self.assertEqual(load_evaluation(project_path, evaluation.evaluation_id), evaluation)
            self.assertEqual(load_evaluation(project_path, evaluation.evaluation_id).snapshot_id, "snapshot-1")
            decisions = load_evaluation_decisions(project_path, evaluation.evaluation_id)
            self.assertEqual([item.status for item in decisions], ["candidate"])
            self.assertFalse(decisions[0].target_satisfied)

            with self.assertRaises(EvaluationRunError):
                record_evaluation_decision(
                    project_path,
                    evaluation.evaluation_id,
                    "validated",
                    actor="engineer",
                    notes="not ready",
                )

            record_evaluation_decision(
                project_path,
                evaluation.evaluation_id,
                "validated",
                actor="engineer",
                criteria={"allow_unmet_targets": True},
                notes="accepted exception",
            )
            record_evaluation_decision(
                project_path,
                evaluation.evaluation_id,
                "approved",
                actor="engineer",
                criteria={"allow_unmet_targets": True},
                notes="explicit approval",
            )
            self.assertEqual(
                [item.status for item in load_evaluation_decisions(project_path, evaluation.evaluation_id)],
                ["candidate", "validated", "approved"],
            )
            self.assertEqual(load_training_run(project_path, run.run_id), run)

            with self.assertRaises(sqlite3.DatabaseError):
                connection = sqlite3.connect(database_path)
                try:
                    connection.execute(
                        "UPDATE evaluation_runs SET notes = 'changed' WHERE evaluation_id = 'evaluation-1'"
                    )
                finally:
                    connection.close()
            with self.assertRaises(sqlite3.DatabaseError):
                connection = sqlite3.connect(database_path)
                try:
                    connection.execute(
                        "DELETE FROM evaluation_decisions WHERE evaluation_id = 'evaluation-1'"
                    )
                finally:
                    connection.close()

            reevaluation = create_evaluation(
                project_path,
                run.run_id,
                {"macro_f1": 0.5},
                thresholds,
                {"minimum_recall_target": 0.95},
                evaluation_id="evaluation-2",
            )
            self.assertNotEqual(reevaluation.evaluation_id, evaluation.evaluation_id)
            self.assertEqual(load_evaluation(project_path, "evaluation-1").metrics["macro_f1"], 0.4)


if __name__ == "__main__":
    unittest.main()
