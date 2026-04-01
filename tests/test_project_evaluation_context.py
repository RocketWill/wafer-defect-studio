import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
)

from tests.test_evaluation_input_inventory import _seed_project
from wafer_defect_studio import project
from wafer_defect_studio.dataset_split import create_dataset_split
from wafer_defect_studio.evaluation_run import create_evaluation
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.training_run import RunConfig, create_training_run


class ProjectEvaluationContextTest(unittest.TestCase):
    def test_reopen_restores_evaluation_and_missing_context_is_not_actionable(self):
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
            for evaluation_id in ("evaluation-1", "evaluation-2"):
                create_evaluation(
                    project_path,
                    run.run_id,
                    {"macro_f1": 0.4},
                    {"per_class": []},
                    {},
                    evaluation_id=evaluation_id,
                )
            database = project_path / "project.sqlite"
            database_bytes = database.read_bytes()
            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)

            first = MainWindow(settings=settings)
            first.show()
            app.processEvents()
            try:
                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(project_path),
                ):
                    first.open_project_action.trigger()
                first.workspace_actions["Evaluate"].trigger()
                app.processEvents()
                evaluations = first.findChild(QComboBox, "evaluationInventoryComboBox")
                evaluations.setCurrentIndex(evaluations.findData("evaluation-2"))
                app.processEvents()
                self.assertEqual(evaluations.currentData(), "evaluation-2")
            finally:
                first.close()
                first.deleteLater()
                app.processEvents()

            second = MainWindow(settings=settings)
            second.show()
            app.processEvents()
            try:
                self.assertEqual(second.current_workspace, "Evaluate")
                self.assertEqual(
                    second.findChild(QComboBox, "evaluationInventoryComboBox").currentData(),
                    "evaluation-2",
                )
                self.assertIn(
                    "Evaluation: evaluation-2",
                    second.findChild(QLabel, "evaluationIdLabel").text(),
                )
            finally:
                second.close()
                second.deleteLater()
                app.processEvents()

            project_id = project.open_project(project_path).project_id
            settings.setValue(f"evaluationContext/{project_id}/evaluationId", "missing-evaluation")
            settings.sync()
            third = MainWindow(settings=settings)
            third.show()
            app.processEvents()
            try:
                status = third.findChild(QLabel, "evaluationInventoryStatusLabel")
                actor = third.findChild(QLineEdit, "evaluationApprovalActorEdit")
                notes = third.findChild(QPlainTextEdit, "evaluationApprovalNotesEdit")
                validate = third.findChild(QPushButton, "validateEvaluationButton")
                approve = third.findChild(QPushButton, "approveEvaluationButton")
                actor.setText("engineer")
                notes.setPlainText("Should stay disabled")
                self.assertIn("Evaluation unavailable: missing-evaluation", status.text())
                self.assertFalse(validate.isEnabled())
                self.assertFalse(approve.isEnabled())
                self.assertEqual(database.read_bytes(), database_bytes)
            finally:
                third.close()
                third.deleteLater()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
