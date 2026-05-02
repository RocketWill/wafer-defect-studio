import os
import hashlib
import json
import queue
import sqlite3
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QFileDialog, QLabel, QPushButton

import tests.test_detection_run as detection_run_test
from wafer_defect_studio.detection_run import create_detection_profile
from wafer_defect_studio.detection_worker import DetectionTerminal
from wafer_defect_studio.cam_detection import CamDetectionArtifact
from wafer_defect_studio.evaluation_run import create_evaluation, record_evaluation_decision
from wafer_defect_studio.image_asset import ImageAsset
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.training_run import (
    RunConfig,
    create_training_run,
    load_training_run,
    update_training_run_terminal,
)


class _FakeHandle:
    def __init__(self):
        self.queue = queue.Queue()
        self.cancelled = False

    def cancel(self, _reason="user"):
        self.cancelled = True
        return True

    def is_alive(self):
        return True


class ProjectDetectionLaunchTest(unittest.TestCase):
    def test_v3_project_detection_request_renders_existing_class_map_controls(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = detection_run_test.DetectionRunTest()._project_with_approved_evaluation(root)
            run = create_training_run(
                project_path,
                RunConfig(
                    "snapshot-1",
                    "split-1",
                    patch_size=2,
                    patch_stride=1,
                    bag_pooling="max",
                ),
                run_id="run-v3",
            )
            _write_v3_checkpoint(run.staging_path / "model.pt")
            _write_manifest(run.staging_path, run.staging_path / "model.pt")
            update_training_run_terminal(project_path, run.run_id, "completed")
            evaluation = create_evaluation(
                project_path,
                run.run_id,
                {"macro_f1": 0.9},
                {"target_satisfied": True},
                {},
                evaluation_id="evaluation-v3",
            )
            record_evaluation_decision(
                project_path, evaluation.evaluation_id, "validated", actor="engineer"
            )
            record_evaluation_decision(
                project_path,
                evaluation.evaluation_id,
                "approved",
                actor="engineer",
                notes="approved for patch detection",
            )
            profile = create_detection_profile(
                project_path,
                evaluation_id=evaluation.evaluation_id,
                window_size=(2, 2),
                stride=(1, 1),
                thresholds={"scratch": 0.5},
                center_weighting="uniform",
                profile_id="profile-v3",
            )
            source_path = root / "wafer.png"
            image = QImage(2, 2, QImage.Format.Format_Grayscale8)
            image.fill(80)
            image.save(str(source_path), "PNG")
            asset = ImageAsset("image-1", source_path, 2, 2, "uint8", "PNG", "fp-1")
            window = MainWindow(
                settings=QSettings(str(root / "settings-v3.ini"), QSettings.Format.IniFormat)
            )
            window.show()
            app.processEvents()
            started = []

            def launcher(request):
                started.append(request)
                artifact = CamDetectionArtifact(
                    maps=np.full((2, 2, 1), 0.8),
                    coverage=np.ones((2, 2), dtype=np.int32),
                    class_names=("scratch",),
                    source_width=2,
                    source_height=2,
                    source_transform={"coordinate_system": "source-image pixels"},
                    window_settings={"map_method": "patch_classification_sigmoid"},
                    provenance={"run_id": request.run_id, "profile_id": request.profile_id},
                )
                stage = Path(request.staging_path)
                stage.mkdir(parents=True)
                artifact.write_json(stage / "maps.json")
                (stage / "provenance.json").write_text(
                    json.dumps(artifact.provenance), encoding="utf-8"
                )
                handle = _FakeHandle()
                handle.queue.put(
                    DetectionTerminal(
                        request.request_id,
                        "completed",
                        "Patch Detection completed.",
                        artifact_staging_path=stage,
                    )
                )
                return handle

            try:
                with patch.object(QFileDialog, "getExistingDirectory", return_value=str(project_path)):
                    window.open_project_action.trigger()
                window.show_wafer_image(asset)
                window.workspace_actions["Detect"].trigger()
                window.configure_project_detection(launcher=launcher)
                app.processEvents()
                start = window.findChild(QPushButton, "startDetectionButton")
                start.click()
                deadline = time.monotonic() + 2
                while "Completed" not in window.findChild(QLabel, "detectionStatusLabel").text() and time.monotonic() < deadline:
                    QTest.qWait(20)
                self.assertEqual(len(started), 1)
                self.assertEqual((started[0].window_size, started[0].stride), ((2, 2), (1, 1)))
                self.assertEqual(profile.profile_id, "profile-v3")
                self.assertEqual(window.findChild(QComboBox, "detectionClassSelector").currentText(), "scratch")
                self.assertIn("source 2×2 px", window.findChild(QLabel, "detectionMapPreviewLabel").text())
                self.assertIn("0.800", window.findChild(QLabel, "detectionConfidenceLabel").text())
                self.assertTrue(window.findChild(QComboBox, "detectionRegionModeComboBox").isEnabled())
                disclaimer = window.findChild(QLabel, "detectionDisclaimerLabel").text()
                self.assertIn("Approximate localization", disclaimer)
                self.assertIn("not a segmentation mask", disclaimer)
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()

    def test_project_detection_launch_persists_cancelled_run_without_blocking(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = detection_run_test.DetectionRunTest()._project_with_approved_evaluation(root)
            _complete_checkpoint(project_path)
            profile = create_detection_profile(
                project_path,
                evaluation_id="evaluation-1",
                window_size=(2, 2),
                stride=(2, 2),
                reflect_padding=True,
                thresholds={"scratch": 0.6},
                center_weighting="uniform",
                map_generation={},
                profile_id="profile-1",
            )
            source_path = root / "wafer.png"
            image = QImage(2, 2, QImage.Format.Format_Grayscale8)
            image.bits()[:4] = bytes((0, 64, 128, 255))
            image.save(str(source_path), "PNG")
            asset = ImageAsset(
                "image-1",
                source_path,
                2,
                2,
                "uint8",
                "PNG",
                "fp-1",
            )
            database = project_path / "project.sqlite"
            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
            window = MainWindow(settings=settings)
            window.show()
            app.processEvents()
            handle = _FakeHandle()
            started = []

            def launcher(request):
                started.append(request)
                return handle

            try:
                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(project_path),
                ):
                    window.open_project_action.trigger()
                window.show_wafer_image(asset)
                window.workspace_actions["Detect"].trigger()
                window.configure_project_detection(launcher=launcher)
                app.processEvents()

                start = window.findChild(QPushButton, "startDetectionButton")
                start.click()
                self.assertEqual(
                    len(started),
                    1,
                    f"enabled={start.isEnabled()} status={window.findChild(QLabel, 'detectionStatusLabel').text()} profile={window.findChild(QComboBox, 'detectionProfileComboBox').currentData()}",
                )
                request = started[0]
                self.assertEqual(request.profile_id, profile.profile_id)
                self.assertEqual(request.approved_evaluation_id, "evaluation-1")
                self.assertEqual(request.source.shape, (2, 2))
                self.assertTrue(request.run_id)
                handle.queue.put(
                    DetectionTerminal(
                        request.request_id,
                        "cancelled",
                        "Detection cancelled; no map set was published.",
                    )
                )
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    QTest.qWait(25)
                    connection = sqlite3.connect(database)
                    try:
                        row = connection.execute(
                            "SELECT status FROM detection_runs WHERE detection_run_id = ?",
                            (request.run_id,),
                        ).fetchone()
                    finally:
                        connection.close()
                    if row is not None:
                        break
                self.assertEqual(row, ("cancelled",))
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


def _complete_checkpoint(project_path: Path) -> None:
    run = load_training_run(project_path, "run-1")
    checkpoint = run.staging_path / "model.pt"
    torch.save(
        {
            "checkpoint_format": "wafer_defect_studio.resnet18.v1",
            "architecture": "resnet18",
            "class_count": 1,
            "class_codes": ["scratch"],
            "normalization_bounds": [
                {
                    "dtype": "uint8",
                    "source_min": 0,
                    "source_max": 255,
                    "low": 0.0,
                    "high": 255.0,
                    "low_percentile": 1.0,
                    "high_percentile": 99.0,
                }
            ],
            "input_size": {"width": 2, "height": 2},
            "state_dict": {"dummy": torch.zeros(1)},
        },
        checkpoint,
    )
    (run.staging_path / "manifest.json").write_text(
        json.dumps(
            {
                "required_files": ["model.pt"],
                "files": [
                    {
                        "path": "model.pt",
                        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    update_training_run_terminal(project_path, run.run_id, "completed")


def _write_v3_checkpoint(path: Path) -> None:
    torch.save(
        {
            "checkpoint_format": "wafer_defect_studio.resnet18.v3",
            "architecture": "resnet18",
            "feature_stride": 16,
            "class_count": 1,
            "class_codes": ["scratch"],
            "normalization_bounds": [
                {
                    "dtype": "uint8",
                    "source_min": 0,
                    "source_max": 255,
                    "low": 0.0,
                    "high": 255.0,
                    "low_percentile": 1.0,
                    "high_percentile": 99.0,
                }
            ],
            "input_size": {"width": 2, "height": 2},
            "patch_size": 2,
            "patch_stride": 1,
            "bag_pooling": "max",
            "state_dict": {"dummy": torch.zeros(1)},
        },
        path,
    )


def _write_manifest(staging: Path, checkpoint: Path) -> None:
    (staging / "manifest.json").write_text(
        json.dumps(
            {
                "required_files": ["model.pt"],
                "files": [
                    {
                        "path": "model.pt",
                        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
