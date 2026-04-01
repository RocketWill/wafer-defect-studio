import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
)

from tests.test_evaluation_input_inventory import _seed_project
from wafer_defect_studio.dataset_split import create_dataset_split
from wafer_defect_studio.evaluation_run import (
    create_evaluation,
    load_evaluation_decisions,
)
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.training_run import RunConfig, create_training_run


class ProjectEvaluationDecisionTest(unittest.TestCase):
    def test_project_evaluation_validate_persists_append_only_and_keeps_unmet_guard(self):
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
            create_evaluation(
                project_path,
                run.run_id,
                {
                    "macro_f1": 0.4,
                    "per_class": [{"class_name": "scratch", "recall": 0.8}],
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
            database = project_path / "project.sqlite"
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

                actor = window.findChild(QLineEdit, "evaluationApprovalActorEdit")
                criteria = window.findChild(QLineEdit, "evaluationApprovalCriteriaEdit")
                notes = window.findChild(QPlainTextEdit, "evaluationApprovalNotesEdit")
                validate = window.findChild(QPushButton, "validateEvaluationButton")
                status = window.findChild(QLabel, "evaluationApprovalStatusLabel")
                history = window.findChild(QTableWidget, "evaluationDecisionHistoryTable")
                actor.setText("engineer")
                notes.setPlainText("Reviewed project evaluation")
                before = database.read_bytes()
                validate.click()
                app.processEvents()
                self.assertIn("allow_unmet_targets=true", status.text())
                self.assertEqual(database.read_bytes(), before)
                self.assertEqual(len(load_evaluation_decisions(project_path, "evaluation-1")), 1)

                criteria.setText("allow_unmet_targets=true")
                validate.click()
                deadline = time.monotonic() + 2
                while history.rowCount() < 2 and time.monotonic() < deadline:
                    app.processEvents()
                    time.sleep(0.01)
                self.assertEqual(history.rowCount(), 2)
                self.assertEqual(history.item(1, 2).text(), "Validated")
                self.assertIn("Recorded Validated decision", status.text())
                self.assertEqual(
                    [item.status for item in load_evaluation_decisions(project_path, "evaluation-1")],
                    ["candidate", "validated"],
                )
                self.assertNotEqual(database.read_bytes(), before)
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
