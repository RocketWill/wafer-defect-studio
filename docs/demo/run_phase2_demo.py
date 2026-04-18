"""Run the smallest reproducible model-backed Wafer Defect Studio demo.

The demo creates a temporary project, drives the real PySide6 shell and
background workers, and keeps only screenshots plus a small JSON summary in
the requested output directory.  The source images are synthetic, but the
Training, Evaluation, and Detection stages use the real checkpoint path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from PySide6.QtCore import QSettings, QPointF, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QApplication, QComboBox, QFileDialog, QLabel, QPushButton, QSpinBox

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.annotation import GridAnnotation, save_grid_annotation
from wafer_defect_studio.dataset_split import create_dataset_split
from wafer_defect_studio.dataset_workflow import create_project_dataset_snapshot
from wafer_defect_studio.defect_class import DefectClass, save_defect_classes
from wafer_defect_studio.detection_run import (
    create_detection_profile,
    load_detection_run,
)
from wafer_defect_studio.evaluation_run import (
    load_evaluation,
    record_evaluation_decision,
)
from wafer_defect_studio.evaluation_worker import (
    EvaluationTerminal,
    build_checkpoint_evaluation_request,
    create_evaluation_from_staged,
    decode_message as decode_evaluation_message,
    start_evaluation_worker,
)
from wafer_defect_studio.effective_area import (
    confirmed_participating_grids,
    confirm_effective_wafer_area,
    load_effective_wafer_area,
    set_effective_ellipse,
)
from wafer_defect_studio.detection_validation import compare_detection_proposals
from wafer_defect_studio.grid_geometry import annotation_grids
from wafer_defect_studio.grid_profile import load_grid_profiles, save_grid_profile
from wafer_defect_studio.image_grid_placement import set_image_grid_origin
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.proposal_generation import (
    generate_confidence_regions,
    generate_proposals,
)
from wafer_defect_studio.result_export import export_proposals_png
from wafer_defect_studio.review import mark_image_reviewed
from wafer_defect_studio.training_run import load_training_run, validate_project_checkpoint
from wafer_defect_studio.training_scope import DataGroup, assign_image_to_data_group, save_data_groups


def main() -> int:
    args = _arguments()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    with TemporaryDirectory(prefix="wafer-demo-") as temporary:
        project_path, asset, source_path, grid_profile = _seed_project(Path(temporary), app)
        settings = QSettings(
            str(Path(temporary) / "demo-settings.ini"),
            QSettings.Format.IniFormat,
        )
        window = MainWindow(settings=settings)
        window.resize(1800, 1100)
        window.show()
        app.processEvents()
        try:
            _activate_image(window, project_path, asset, grid_profile, "Project created")
            _capture(window, output / "01-import.png", app)

            annotations = _prepare_annotation(window, project_path, asset, grid_profile, app)
            window.workspace_actions["Annotate"].trigger()
            _capture(window, output / "02-annotation-two-classes.png", app)

            snapshot_id, split_id = _create_dataset(project_path)
            _activate_image(window, project_path, asset, grid_profile, "Dataset Snapshot created")
            window.workspace_actions["Dataset"].trigger()
            _select_dataset_scope(window, app)
            _capture(window, output / "03-dataset-snapshot.png", app)

            _activate_image(window, project_path, asset, grid_profile, "Training inputs ready")
            window.workspace_actions["Train"].trigger()
            training = _run_training(window, snapshot_id, split_id, app)
            if training.artifact_path is None:
                raise RuntimeError("demo Training completed without a published checkpoint")
            checkpoint_path = training.artifact_path / "model.pt"
            validate_project_checkpoint(checkpoint_path)
            if not (training.artifact_path / "training_input_bundle.json").is_file():
                raise RuntimeError("demo Training artifact is missing the real input bundle")
            _capture(window, output / "04-training-complete.png", app)

            evaluation = _run_evaluation(project_path, training.run_id, snapshot_id, split_id, Path(temporary))
            record_evaluation_decision(
                project_path,
                evaluation.evaluation_id,
                "validated",
                actor="Demo Engineer",
                notes="Synthetic demo validation.",
            )
            record_evaluation_decision(
                project_path,
                evaluation.evaluation_id,
                "approved",
                actor="Demo Engineer",
                notes="Synthetic demo approved for heatmap visualization.",
            )
            detection_profile = create_detection_profile(
                project_path,
                evaluation_id=evaluation.evaluation_id,
                training_run_id=training.run_id,
                window_size=(32, 32),
                stride=(16, 16),
                reflect_padding=True,
                thresholds={"scratch": 0.0, "particle": 0.0},
                center_weighting="linear",
                map_generation={"smoothing": 1, "minimum_area": 1},
                profile_id="demo-profile",
            )
            _activate_image(window, project_path, asset, grid_profile, "Evaluation approved")
            window.workspace_actions["Evaluate"].trigger()
            _select_combo(window, "evaluationInventoryComboBox", evaluation.evaluation_id)
            _capture(window, output / "05-evaluation-approved.png", app)

            _activate_image(window, project_path, asset, grid_profile, "Detection profile ready")
            window.workspace_actions["Detect"].trigger()
            _select_combo(window, "detectionProfileComboBox", detection_profile.profile_id)
            artifact, detection_run_id = _run_detection(
                window,
                project_path,
                asset,
                detection_profile,
                app,
            )
            if artifact.provenance.get("map_method") != "all_convolutional_sigmoid":
                raise RuntimeError("demo refused a non-checkpoint-backed Detection artifact")
            _activate_image(window, project_path, asset, grid_profile, "Detection completed")
            window.workspace_actions["Detect"].trigger()
            _select_combo(window, "detectionProfileComboBox", detection_profile.profile_id)
            _select_combo(window, "detectionRunComboBox", detection_run_id)
            window.set_detection_artifact(artifact)
            _capture(window, output / "06-detection-controls.png", app)

            window._detection_controls.region_mode_combo.setCurrentText("Heatmap")
            _capture(window, output / "06-heatmap.png", app)
            window._detection_controls.region_mode_combo.setCurrentText("Regions")
            _capture(window, output / "07-regions.png", app)
            window._detection_controls.region_mode_combo.setCurrentText("Both")
            _capture(window, output / "08-both.png", app)

            source_image = QImage(str(source_path))
            proposals = generate_proposals(artifact, detection_profile)
            area = load_effective_wafer_area(project_path, asset.image_asset_id)
            if area is None:
                raise RuntimeError("demo Effective Wafer Area is missing")
            participating = confirmed_participating_grids(
                annotation_grids(
                    source_image.width(),
                    source_image.height(),
                    grid_profile.cell_width,
                    grid_profile.cell_height,
                    0,
                    0,
                ),
                area,
            )
            validation = compare_detection_proposals(
                annotations,
                proposals,
                participating,
                class_names=("scratch", "particle"),
            )
            heatmap_path = output / "06-heatmap-export.png"
            grid_rects = tuple(
                (grid.x, grid.y, grid.width, grid.height)
                for grid in annotation_grids(
                    source_image.width(),
                    source_image.height(),
                    grid_profile.cell_width,
                    grid_profile.cell_height,
                    0,
                    0,
                )
            )
            export_proposals_png(
                heatmap_path,
                source_image,
                (),
                selected_class="scratch",
                confidence_map=artifact.class_map("scratch"),
                region_mask=generate_confidence_regions(artifact, detection_profile)["scratch"],
                region_opacity=window._detection_controls.opacity_slider.value() / 100.0,
                grid_rects=grid_rects,
                overwrite=True,
            )
            summary = _summary(
                project_path,
                source_image,
                training.run_id,
                evaluation.evaluation_id,
                detection_run_id,
                artifact,
                heatmap_path,
                validation,
                detection_profile,
            )
            (output / "demo-summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        finally:
            window.close()
            window.deleteLater()
            app.processEvents()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "docs" / "demo" / "screenshots",
        help="directory for screenshots and demo-summary.json",
    )
    return parser.parse_args()


def _seed_project(root: Path, app: QApplication):
    project_path = root / "project"
    project.create_project(project_path)
    source_path = root / "synthetic-wafer.png"
    source = _synthetic_wafer()
    if not source.save(str(source_path), "PNG"):
        raise RuntimeError("failed to write synthetic wafer")

    settings = QSettings(str(root / "seed-settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    window.show()
    app.processEvents()
    try:
        window._activate_project(project.open_project(project_path), "Project created")
        with patch.object(
            QFileDialog,
            "getOpenFileName",
            return_value=(str(source_path), "PNG"),
        ):
            window.import_wafer_image_action.trigger()
        _wait(app, lambda: window.current_image_asset is not None)
        asset = window.current_image_asset
        if asset is None:
            raise RuntimeError("GUI import did not produce an Image Asset")
        profile = save_grid_profile(project_path, 32, 32)
        set_image_grid_origin(project_path, asset.image_asset_id, profile.grid_profile_id, 0, 0)
        set_effective_ellipse(
            project_path,
            asset.image_asset_id,
            64,
            48,
            42,
            42,
        )
        confirm_effective_wafer_area(project_path, asset.image_asset_id)
        for index in range(1, 10):
            extra_path = root / f"synthetic-wafer-{index:02d}.png"
            extra = _synthetic_wafer(index)
            if not extra.save(str(extra_path), "PNG"):
                raise RuntimeError(f"failed to write {extra_path}")
            extra_asset = image_asset.register_wafer_image(project_path, extra_path)
            set_image_grid_origin(
                project_path,
                extra_asset.image_asset_id,
                profile.grid_profile_id,
                0,
                0,
            )
            set_effective_ellipse(
                project_path,
                extra_asset.image_asset_id,
                64,
                48,
                42,
                42,
            )
            confirm_effective_wafer_area(project_path, extra_asset.image_asset_id)
        save_defect_classes(
            project_path,
            (
                DefectClass("scratch", "Scratch", "#cc4444", order=0),
                DefectClass("particle", "Particle", "#4488cc", order=1),
            ),
        )
        save_data_groups(project_path, (DataGroup("demo-line", "Demo Line"),))
        for item in image_asset.load_image_assets(project_path):
            assign_image_to_data_group(project_path, item.asset.image_asset_id, "demo-line")
        _activate_image(window, project_path, asset, profile, "Image imported")
        return project_path, asset, source_path, profile
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def _synthetic_wafer(variant: int = 0) -> QImage:
    image = QImage(128, 96, QImage.Format.Format_Grayscale8)
    image.fill(14)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    gradient = QRadialGradient(QPointF(64, 48), 42)
    gradient.setColorAt(0.0, QColor(225, 225, 225))
    gradient.setColorAt(0.72, QColor(150, 150, 150))
    gradient.setColorAt(1.0, QColor(48, 48, 48))
    painter.setBrush(gradient)
    painter.setPen(QPen(QColor(245, 245, 245), 4))
    painter.drawEllipse(QRectF(22, 6, 84, 84))
    painter.setBrush(QColor(0, 0, 0, 0))
    painter.setPen(QPen(QColor(255, 230, 70), 2))
    painter.drawEllipse(QRectF(25, 9, 78, 78))
    painter.setPen(QPen(QColor(35, 35, 35), 9))
    offset = (variant % 3) - 1
    painter.drawLine(QPointF(38 + offset, 37), QPointF(58 + offset, 57))
    painter.setPen(QPen(QColor(250, 250, 250), 3))
    painter.setBrush(QColor(245, 245, 245))
    for x, y, radius in ((80 + offset, 80, 5), (88, 32, 3), (96, 56, 4), (48, 72, 2)):
        painter.drawEllipse(QPointF(x, y), radius, radius)
    painter.end()
    return image


def _activate_image(window, project_path, asset, profile, status: str) -> None:
    window._activate_project(project.open_project(project_path), status)
    window.show_wafer_image(asset)
    window.set_grid_profile(project_path, profile)


def _prepare_annotation(window, project_path, asset, profile, app):
    annotations = {
        (1, 1): ("scratch",),
        (2, 2): ("particle",),
    }
    assets = [asset]
    assets.extend(
        item.asset
        for item in image_asset.load_image_assets(project_path)
        if item.asset.image_asset_id != asset.image_asset_id
    )
    for current in assets:
        for (row, column), class_codes in annotations.items():
            save_grid_annotation(
                project_path,
                GridAnnotation(current.image_asset_id, row, column, class_codes),
            )
        mark_image_reviewed(project_path, current.image_asset_id)
    window._image_view.set_annotations(annotations)
    window.set_selected_defect_classes(("scratch", "particle"))
    window.set_annotation_mode("Annotate")
    window._refresh_review_controls()
    app.processEvents()
    return tuple(
        GridAnnotation(asset.image_asset_id, row, column, class_codes)
        for (row, column), class_codes in annotations.items()
    )


def _create_dataset(project_path: Path) -> tuple[str, str]:
    snapshot_id, _preview = create_project_dataset_snapshot(
        project_path,
        ("demo-line",),
        ("scratch", "particle"),
    )
    split = create_dataset_split(project_path, snapshot_id, 42)
    if not split.train_image_ids or not split.validation_image_ids or not split.test_image_ids:
        raise RuntimeError("demo Dataset Split must contain train, validation, and test images")
    return snapshot_id, split.split_id


def _select_dataset_scope(window: MainWindow, app: QApplication) -> None:
    controls = window._training_scope_controls
    controls.set_selected_data_groups(("demo-line",))
    controls.set_selected_classes(("scratch", "particle"))
    window._refresh_dataset_preview()
    app.processEvents()


def _run_training(window: MainWindow, snapshot_id: str, split_id: str, app: QApplication):
    _select_combo(window, "trainingSnapshotComboBox", snapshot_id)
    _select_combo(window, "trainingSplitComboBox", split_id)
    window.findChild(QSpinBox, "trainingEpochsSpinBox").setValue(1)
    window.findChild(QSpinBox, "trainingBatchSizeSpinBox").setValue(2)
    window.findChild(QSpinBox, "trainingSeedSpinBox").setValue(42)
    window.findChild(QComboBox, "trainingDeviceComboBox").setCurrentText("cpu")
    window.findChild(QComboBox, "trainingWeightsPolicyComboBox").setCurrentText("none")
    start = window.findChild(QPushButton, "startTrainingButton")
    if start is None or not start.isEnabled():
        raise RuntimeError("Training controls are not ready")
    start.click()
    status = window.findChild(QLabel, "trainingStatusLabel")
    if status is None:
        raise RuntimeError("Training status label is missing")
    _wait(
        app,
        lambda: "Completed" in status.text() or "Failed" in status.text(),
        timeout=120,
    )
    if "Failed" in status.text():
        raise RuntimeError(f"Training failed: {status.text()}")
    connection = sqlite3.connect(window.active_project_path / "project.sqlite")
    try:
        run_id = connection.execute(
            "SELECT run_id FROM training_runs ORDER BY created_at DESC, rowid DESC LIMIT 1"
        ).fetchone()[0]
    finally:
        connection.close()
    return load_training_run(window.active_project_path, run_id)


def _run_evaluation(
    project_path: Path,
    training_run_id: str,
    snapshot_id: str,
    split_id: str,
    root: Path,
):
    request = build_checkpoint_evaluation_request(
        project_path,
        training_run_id,
        root / "evaluation-stage",
        request_id="demo-evaluation-worker",
        split="test",
        thresholds=(0.5, 0.5),
        criteria={"minimum_recall_target": 0.0},
        notes="Checkpoint-backed two-class demo evaluation.",
        device="cpu",
    )
    handle = start_evaluation_worker(request)
    terminal = _collect_worker(handle, decode_evaluation_message)
    if not isinstance(terminal, EvaluationTerminal) or terminal.status != "completed":
        raise RuntimeError(f"Evaluation worker failed: {terminal}")
    return create_evaluation_from_staged(
        project_path,
        request.staging_path,
        training_run_id=training_run_id,
        actor="Demo Engineer",
        evaluation_id="demo-evaluation",
    )


def _run_detection(window, project_path, asset, profile, app):
    start = window.findChild(QPushButton, "startDetectionButton")
    if start is None or not start.isEnabled():
        raise RuntimeError("Detection controls are not ready")
    start.click()
    status = window.findChild(QLabel, "detectionStatusLabel")
    if status is None:
        raise RuntimeError("Detection status label is missing")
    _wait(
        app,
        lambda: "Completed" in status.text() or "Failed" in status.text(),
        timeout=120,
    )
    if "Failed" in status.text():
        raise RuntimeError(f"Detection failed: {status.text()}")
    artifact = window._detection_controls.artifact
    if artifact is None:
        raise RuntimeError("Detection worker completed without a CAM artifact")
    connection = sqlite3.connect(project_path / "project.sqlite")
    try:
        run_id = connection.execute(
            "SELECT detection_run_id FROM detection_runs ORDER BY created_at DESC, rowid DESC LIMIT 1"
        ).fetchone()[0]
    finally:
        connection.close()
    load_detection_run(project_path, run_id)
    return artifact, run_id


def _collect_worker(handle, decoder):
    messages = []
    while True:
        messages.append(decoder(handle.queue.get(timeout=120)))
        if isinstance(messages[-1], (EvaluationTerminal,)):
            handle.join(timeout=30)
            return messages[-1]


def _select_combo(window, object_name: str, value: str) -> None:
    combo = window.findChild(QComboBox, object_name)
    if combo is None:
        raise RuntimeError(f"Missing combo: {object_name}")
    index = combo.findData(value)
    if index < 0:
        raise RuntimeError(f"Value {value!r} is unavailable in {object_name}")
    combo.setCurrentIndex(index)


def _capture(window: MainWindow, path: Path, app: QApplication) -> None:
    app.processEvents()
    time.sleep(0.05)
    app.processEvents()
    if not window.grab().save(str(path), "PNG"):
        raise RuntimeError(f"failed to capture {path}")


def _wait(app: QApplication, predicate, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.05)
    raise TimeoutError("demo stage timed out")


def _summary(
    project_path,
    source_image,
    training_id,
    evaluation_id,
    detection_id,
    artifact,
    heatmap_path,
    validation,
    detection_profile,
):
    training = load_training_run(project_path, training_id)
    evaluation = load_evaluation(project_path, evaluation_id)
    detection = load_detection_run(project_path, detection_id)
    checkpoint_path = training.artifact_path / "model.pt"
    checkpoint = validate_project_checkpoint(checkpoint_path)
    checkpoint_checksum = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    finite = np.asarray(artifact.class_map("scratch"))[np.isfinite(artifact.class_map("scratch"))]
    return {
        "synthetic_source": True,
        "model_backed": True,
        "source": {"width": source_image.width(), "height": source_image.height(), "format": "uint8 PNG"},
        "classes": ["scratch", "particle"],
        "training": {
            "run_id": training.run_id,
            "status": training.status,
            "architecture": training.config.architecture,
            "epochs": training.config.epochs,
            "device": training.config.device,
            "metrics": training.metrics,
            "checkpoint_sha256": checkpoint_checksum,
            "checkpoint_format": checkpoint["checkpoint_format"],
            "class_order": list(checkpoint["class_codes"]),
            "input_size": checkpoint["input_size"],
        },
        "evaluation": {
            "evaluation_id": evaluation.evaluation_id,
            "split_id": evaluation.split_id,
            "macro_f1": evaluation.metrics.get("macro_f1"),
            "per_class": evaluation.metrics.get("per_class", []),
            "target_satisfied": evaluation.target_satisfied,
            "decisions": ["candidate", "validated", "approved"],
        },
        "detection": {
            "run_id": detection.run_id,
            "status": detection.status,
            "class_names": list(artifact.class_names),
            "map_shape": list(artifact.maps.shape),
            "map_method": artifact.provenance.get("map_method"),
            "profile": {
                "profile_id": detection_profile.profile_id,
                "window_size": list(detection_profile.window_size),
                "stride": list(detection_profile.stride),
                "thresholds": dict(detection_profile.thresholds),
                "map_generation": dict(detection_profile.map_generation),
            },
            "scratch_min": float(np.min(finite)) if finite.size else None,
            "scratch_max": float(np.max(finite)) if finite.size else None,
            "heatmap": str(heatmap_path.name),
        },
        "annotation_validation": validation.to_dict(),
        "validation_limitations": [
            "Synthetic source images and fixed grid labels are for pipeline validation only.",
            "Metrics are observed on one held-out image and are not production accuracy claims.",
            "Confidence regions are approximate localization, not a segmentation mask.",
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
