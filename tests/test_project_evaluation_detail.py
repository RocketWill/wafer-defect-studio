import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QTableWidget

from tests.test_evaluation_input_inventory import _seed_project
from wafer_defect_studio.dataset_split import create_dataset_split
from wafer_defect_studio.evaluation_run import create_evaluation, record_evaluation_decision
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.training_run import RunConfig, create_training_run


class ProjectEvaluationDetailTest(unittest.TestCase):
    def test_selected_project_evaluation_renders_immutable_detail_and_history(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path, snapshot_id = _seed_project(root)
            split = create_dataset_split(project_path, snapshot_id, 42)
            run = create_training_run(
                project_path,
                RunConfig(snapshot_id, split.split_id, class_count=1),
                run_id="run-1",
            )
            evaluation = create_evaluation(
                project_path,
                run.run_id,
                {
                    "macro_f1": 0.4,
                    "per_class": [
                        {
                            "class_name": "scratch",
                            "precision": 0.5,
                            "recall": 0.8,
                            "f1": 0.62,
                            "support": 5,
                            "fpr": 0.1,
                        }
                    ],
                },
                {
                    "min_recall_target": 0.95,
                    "per_class": [
                        {
                            "class_name": "scratch",
                            "policy": "min_recall",
                            "threshold": 0.9,
                            "target": 0.95,
                            "target_satisfied": False,
                        }
                    ],
                },
                {"minimum_recall_target": 0.95},
                evaluation_id="evaluation-1",
            )
            record_evaluation_decision(
                project_path,
                evaluation.evaluation_id,
                "validated",
                "engineer",
                criteria={"allow_unmet_targets": True},
                notes="Reviewed metrics",
            )
            database = project_path / "project.sqlite"
            database_bytes = database.read_bytes()
            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
            window = MainWindow(settings=settings)
            window.show()
            app.processEvents()
            try:
                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(project_path),
                ):
                    window.open_project_action.trigger()
                window.workspace_actions["Evaluate"].trigger()
                app.processEvents()

                metrics = window.findChild(QTableWidget, "evaluationMetricsTable")
                history = window.findChild(QTableWidget, "evaluationDecisionHistoryTable")
                target = window.findChild(QLabel, "evaluationTargetStatusLabel")
                evaluation_id = window.findChild(QLabel, "evaluationIdLabel")
                wafer = window.findChild(QLabel, "waferEvaluationPlaceholderLabel")
                self.assertEqual(evaluation_id.text(), "Evaluation: evaluation-1 | Training Run: run-1")
                self.assertEqual(metrics.rowCount(), 1)
                self.assertEqual(metrics.item(0, 0).text(), "scratch")
                self.assertEqual(metrics.item(0, 6).text(), "0.9000")
                self.assertEqual(history.rowCount(), 2)
                self.assertEqual(history.item(1, 2).text(), "Validated")
                self.assertIn("UNMET", target.text())
                self.assertIn("basic structure", wafer.text())
                self.assertEqual(database.read_bytes(), database_bytes)
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
