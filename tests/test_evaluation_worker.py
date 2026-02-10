import json
import multiprocessing as mp
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from wafer_defect_studio import project
from wafer_defect_studio.evaluation_run import load_evaluation
from wafer_defect_studio.evaluation_worker import (
    EvaluationProgress,
    EvaluationRequest,
    EvaluationTerminal,
    create_evaluation_from_staged,
    decode_message,
    start_evaluation_worker,
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
from wafer_defect_studio.training_run import RunConfig, create_training_run, load_training_run


class EvaluationWorkerTest(unittest.TestCase):
    def _project_with_run(self, root: Path) -> tuple[Path, object]:
        project_path = root / "project"
        project.create_project(project_path)
        connection = sqlite3.connect(project_path / "project.sqlite")
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
            RunConfig("snapshot-1", "split-1", class_count=2, device="cuda"),
            environment={"torch": "2.8", "cuda": "12.8"},
            run_id="run-1",
        )
        return project_path, run

    def _collect(self, handle):
        messages = []
        while True:
            message = decode_message(handle.queue.get(timeout=30))
            messages.append(message)
            if isinstance(message, EvaluationTerminal):
                handle.join(timeout=30)
                self.assertFalse(handle.is_alive())
                return messages

    def test_spawned_worker_stages_metrics_and_service_publishes_after_validation(self):
        self.assertEqual(mp.get_context("spawn").get_start_method(), "spawn")
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            project_path, run = self._project_with_run(root)
            before = load_training_run(project_path, run.run_id)
            request = EvaluationRequest(
                request_id="evaluation-request-1",
                run_id=run.run_id,
                snapshot_id="snapshot-1",
                split_id="split-1",
                y_true=((1, 0), (1, 0), (0, 1), (0, 1)),
                y_score=((0.9, 0.2), (0.4, 0.8), (0.3, 0.7), (0.2, 0.4)),
                class_names=("scratch", "crack"),
                policies={"scratch": "max_f1", "crack": "min_recall"},
                criteria={"minimum_recall_target": 0.95},
                environment={"torch": "2.8", "cuda": "12.8"},
                notes="immutable test split",
                staging_path=root / "evaluation-stage",
            )
            handle = start_evaluation_worker(request)
            messages = self._collect(handle)
            self.assertTrue(any(isinstance(item, EvaluationProgress) for item in messages))
            terminal = messages[-1]
            self.assertEqual(terminal.status, "completed")
            stage = root / "evaluation-stage"
            self.assertTrue((stage / "metrics.json").is_file())
            self.assertTrue((stage / "manifest.json").is_file())
            payload = json.loads((stage / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], run.run_id)
            self.assertAlmostEqual(payload["metrics"]["macro_f1"], 0.9)
            self.assertEqual(len(payload["thresholds"]["per_class"]), 2)

            evaluation = create_evaluation_from_staged(
                project_path,
                stage,
                actor="engineer",
                evaluation_id="evaluation-1",
            )
            self.assertEqual(evaluation.training_run_id, run.run_id)
            self.assertEqual(load_evaluation(project_path, evaluation.evaluation_id), evaluation)
            self.assertEqual(load_training_run(project_path, run.run_id), before)

            cancelled_request = EvaluationRequest(
                request_id="evaluation-request-2",
                run_id=run.run_id,
                snapshot_id="snapshot-1",
                split_id="split-1",
                y_true=((1,), (0,)),
                y_score=((0.9,), (0.1,)),
                staging_path=root / "cancelled-stage",
            )
            cancelled = start_evaluation_worker(cancelled_request, step_delay=0.2)
            cancelled.cancel()
            cancelled_messages = self._collect(cancelled)
            self.assertEqual(cancelled_messages[-1].status, "cancelled")
            self.assertFalse((root / "cancelled-stage" / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
