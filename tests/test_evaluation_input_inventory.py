import os
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QComboBox, QFileDialog, QLabel

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.annotation import GridAnnotation, save_grid_annotation
from wafer_defect_studio.dataset_split import create_dataset_split
from wafer_defect_studio.dataset_workflow import create_project_dataset_snapshot
from wafer_defect_studio.defect_class import DefectClass, save_defect_classes
from wafer_defect_studio.effective_area import confirm_effective_wafer_area, set_effective_ellipse
from wafer_defect_studio.evaluation_run import create_evaluation
from wafer_defect_studio.grid_profile import save_grid_profile
from wafer_defect_studio.image_grid_placement import set_image_grid_origin
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.review import mark_image_reviewed
from wafer_defect_studio.training_run import RunConfig, create_training_run
from wafer_defect_studio.training_scope import DataGroup, assign_image_to_data_group, save_data_groups


class EvaluationInputInventoryTest(unittest.TestCase):
    def test_evaluate_workspace_lists_valid_and_unavailable_project_evaluations_read_only(self):
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
                {"macro_f1": 0.4},
                {"per_class": []},
                {},
                evaluation_id="evaluation-1",
            )
            connection = sqlite3.connect(project_path / "project.sqlite")
            try:
                connection.execute(
                    "INSERT INTO evaluation_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "evaluation-corrupt",
                        run.run_id,
                        snapshot_id,
                        split.split_id,
                        "2026-01-02T00:00:00+00:00",
                        "{bad-json",
                        "{}",
                        "{}",
                        "{}",
                        0,
                        "corrupt row",
                    ),
                )
                connection.commit()
            finally:
                connection.close()

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

                evaluations = window.findChild(QComboBox, "evaluationInventoryComboBox")
                status = window.findChild(QLabel, "evaluationInventoryStatusLabel")
                self.assertEqual(evaluations.count(), 2)
                self.assertEqual(evaluations.itemData(0), "evaluation-1")
                self.assertEqual(
                    evaluations.itemText(0),
                    "Evaluation evaluation-1 — Training Run run-1",
                )
                self.assertTrue(evaluations.model().item(0).isEnabled())
                corrupt_index = evaluations.findData("evaluation-corrupt")
                self.assertGreaterEqual(corrupt_index, 0)
                self.assertIn("Unavailable Evaluation: evaluation-corrupt", evaluations.itemText(corrupt_index))
                self.assertFalse(evaluations.model().item(corrupt_index).isEnabled())
                self.assertIn("unavailable", status.text().lower())
                self.assertEqual(database.read_bytes(), database_bytes)
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


def _seed_project(root: Path) -> tuple[Path, str]:
    project_path = root / "project"
    project.create_project(project_path)
    source = root / "wafer.png"
    image = QImage(2, 2, QImage.Format.Format_Grayscale8)
    image.bits()[:4] = bytes((0, 64, 128, 255))
    image.save(str(source), "PNG")
    asset = image_asset.register_wafer_image(project_path, source)
    profile = save_grid_profile(project_path, 2, 2)
    set_image_grid_origin(project_path, asset.image_asset_id, profile.grid_profile_id, 0, 0)
    set_effective_ellipse(project_path, asset.image_asset_id, 1, 1, 1, 1)
    confirm_effective_wafer_area(project_path, asset.image_asset_id)
    save_defect_classes(project_path, (DefectClass("scratch", "Scratch", "#cc4444"),))
    save_grid_annotation(project_path, GridAnnotation(asset.image_asset_id, 0, 0, ("scratch",)))
    mark_image_reviewed(project_path, asset.image_asset_id)
    save_data_groups(project_path, (DataGroup("line-a", "Line A"),))
    assign_image_to_data_group(project_path, asset.image_asset_id, "line-a")
    snapshot_id, _preview = create_project_dataset_snapshot(
        project_path, ("line-a",), ("scratch",)
    )
    return project_path, snapshot_id


if __name__ == "__main__":
    unittest.main()
