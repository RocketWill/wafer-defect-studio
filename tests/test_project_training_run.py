import os
import queue
import sqlite3
import threading
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QFileDialog, QLabel, QPushButton, QSpinBox

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.annotation import GridAnnotation, save_grid_annotation
from wafer_defect_studio.dataset_split import create_dataset_split
from wafer_defect_studio.dataset_workflow import create_project_dataset_snapshot
from wafer_defect_studio.defect_class import DefectClass, save_defect_classes
from wafer_defect_studio.effective_area import confirm_effective_wafer_area, set_effective_ellipse
from wafer_defect_studio.grid_profile import save_grid_profile
from wafer_defect_studio.image_grid_placement import set_image_grid_origin
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.review import mark_image_reviewed
from wafer_defect_studio.training_protocol import TerminalMessage
from wafer_defect_studio.training_input_bundle import TrainingInputBundle
from wafer_defect_studio.training_run import load_training_run, validate_project_checkpoint
from wafer_defect_studio.training_worker import run_worker
from wafer_defect_studio.training_scope import DataGroup, assign_image_to_data_group, save_data_groups


@dataclass
class _FakeHandle:
    queue: queue.Queue
    alive: bool = True

    def cancel(self, reason="user"):
        return True

    def is_alive(self):
        return self.alive


class ProjectTrainingRunTest(unittest.TestCase):
    def test_spatial_mil_v4_runs_and_publishes_from_active_project(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path, snapshot_id = _seed_project(
                root, image_size=32, grid_size=32
            )
            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
            window = MainWindow(settings=settings)
            started = []

            def launcher(request):
                output = queue.Queue()
                run_worker(request, output, threading.Event())
                started.append(request)
                return _FakeHandle(output, alive=False)

            window.show()
            app.processEvents()
            try:
                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(project_path),
                ):
                    window.open_project_action.trigger()
                window.workspace_actions["Train"].trigger()
                mode = window.findChild(QComboBox, "trainingModelModeComboBox")
                mode.setCurrentText("Spatial MIL v4")
                window.findChild(QSpinBox, "trainingPatchSizeSpinBox").setValue(32)
                window.findChild(QSpinBox, "trainingPatchStrideSpinBox").setValue(32)
                window.findChild(QSpinBox, "trainingEpochsSpinBox").setValue(1)
                window.findChild(QSpinBox, "trainingBatchSizeSpinBox").setValue(1)
                window.findChild(QComboBox, "trainingDeviceComboBox").setCurrentText("cpu")
                window.configure_project_training(launcher=launcher)
                window.findChild(QPushButton, "startTrainingButton").click()

                deadline = time.monotonic() + 30
                while (
                    not started
                    or _training_status(project_path, started[0].request_id)
                    != "completed"
                ) and time.monotonic() < deadline:
                    QTest.qWait(20)
                self.assertEqual(len(started), 1)
                request = started[0]
                self.assertEqual(request.config.training_policy, "spatial_mil_v4")
                persisted = load_training_run(project_path, request.request_id)
                self.assertEqual(persisted.config.training_policy, "spatial_mil_v4")
                self.assertIsNone(persisted.config.bag_pooling)
                self.assertEqual(persisted.status, "completed")
                self.assertIsNotNone(persisted.artifact_path)
                self.assertFalse(persisted.staging_path.exists())
                checkpoint = validate_project_checkpoint(
                    persisted.artifact_path / "model.pt"
                )
                self.assertEqual(
                    checkpoint["checkpoint_format"],
                    "wafer_defect_studio.resnet18.v4",
                )
                self.assertEqual(
                    checkpoint["architecture"], "resnet18_spatial_logits"
                )
                self.assertEqual(checkpoint["training_policy"], "spatial_mil_v4")
                self.assertTrue((persisted.artifact_path / "manifest.json").is_file())
                self.assertEqual(request.config.snapshot_id, snapshot_id)
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()

    def test_project_training_start_persists_truthful_terminal_run(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path, _snapshot_id = _seed_project(root)
            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
            window = MainWindow(settings=settings)
            handle = _FakeHandle(queue.Queue())
            started = []

            def launcher(request):
                started.append(request)
                return handle

            window.show()
            app.processEvents()
            try:
                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(project_path),
                ), patch("wafer_defect_studio.main_window.start_training_worker", side_effect=launcher):
                    window.open_project_action.trigger()
                window.workspace_actions["Train"].trigger()
                app.processEvents()

                legacy_request = window._training_controls._request_source()
                legacy_bundle = TrainingInputBundle.from_json(
                    Path(legacy_request.input_bundle_path).read_text(encoding="utf-8")
                )
                self.assertEqual(legacy_bundle.version, 1)
                self.assertFalse(legacy_bundle.patch_bags)

                window.findChild(QComboBox, "trainingModelModeComboBox").setCurrentText(
                    "Patch Classification v3"
                )
                window.findChild(QSpinBox, "trainingPatchSizeSpinBox").setValue(1)
                window.findChild(QSpinBox, "trainingPatchStrideSpinBox").setValue(1)
                start = window.findChild(QPushButton, "startTrainingButton")
                start.click()
                self.assertEqual(len(started), 1)
                request = started[0]
                self.assertEqual(request.config.snapshot_id, _snapshot_id)
                self.assertEqual((request.config.patch_size, request.config.patch_stride), (1, 1))
                persisted = load_training_run(project_path, request.request_id)
                self.assertEqual(
                    (
                        persisted.config.patch_size,
                        persisted.config.patch_stride,
                        persisted.config.bag_pooling,
                    ),
                    (1, 1, "max"),
                )
                bundle = TrainingInputBundle.from_json(
                    Path(request.input_bundle_path).read_text(encoding="utf-8")
                )
                self.assertEqual(bundle.version, 2)
                self.assertTrue(bundle.patch_bags)
                self.assertTrue(request.config.split_id)
                self.assertEqual(_training_status(project_path, request.request_id), "running")

                handle.queue.put(
                    TerminalMessage(
                        request_id=request.request_id,
                        status="cancelled",
                        message="Cancelled by test.",
                    )
                )
                deadline = time.monotonic() + 2
                while _training_status(project_path, request.request_id) != "cancelled" and time.monotonic() < deadline:
                    QTest.qWait(20)
                self.assertEqual(_training_status(project_path, request.request_id), "cancelled")
                self.assertIn(
                    "Cancelled by test",
                    window.findChild(QLabel, "trainingStatusLabel").text(),
                )
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


def _training_status(project_path: Path, run_id: str) -> str | None:
    connection = sqlite3.connect(project_path / "project.sqlite")
    try:
        row = connection.execute(
            "SELECT terminal_status FROM training_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    finally:
        connection.close()
    return row[0] if row else None


def _seed_project(
    root: Path, *, image_size: int = 2, grid_size: int = 2
) -> tuple[Path, str]:
    project_path = root / "project"
    project.create_project(project_path)
    source = root / "wafer.png"
    image = QImage(image_size, image_size, QImage.Format.Format_Grayscale8)
    for y in range(image_size):
        for x in range(image_size):
            value = (x * 17 + y * 31) % 256
            image.setPixelColor(x, y, QColor(value, value, value))
    assert image.pixelColor(0, 0).red() == 0
    assert image.pixelColor(1, 0).red() == 17
    image.save(str(source), "PNG")
    asset = image_asset.register_wafer_image(project_path, source)
    profile = save_grid_profile(project_path, grid_size, grid_size)
    set_image_grid_origin(
        project_path, asset.image_asset_id, profile.grid_profile_id, 0, 0
    )
    radius = image_size // 2
    set_effective_ellipse(project_path, asset.image_asset_id, radius, radius, radius, radius)
    confirm_effective_wafer_area(project_path, asset.image_asset_id)
    save_defect_classes(project_path, (DefectClass("scratch", "Scratch", "#cc4444"),))
    save_grid_annotation(project_path, GridAnnotation(asset.image_asset_id, 0, 0, ("scratch",)))
    mark_image_reviewed(project_path, asset.image_asset_id)
    save_data_groups(project_path, (DataGroup("line-a", "Line A"),))
    assign_image_to_data_group(project_path, asset.image_asset_id, "line-a")
    snapshot_id, _preview = create_project_dataset_snapshot(
        project_path, ("line-a",), ("scratch",)
    )
    create_dataset_split(project_path, snapshot_id, 42)
    return project_path, snapshot_id


if __name__ == "__main__":
    unittest.main()
