"""Run the smallest reproducible model-backed Wafer Defect Studio demo.

The demo creates a temporary project, drives the real PySide6 shell and
background workers, and keeps only screenshots plus a small JSON summary in
the requested output directory.  It accepts a small synthetic source or a
generated realistic source; either way Training, Evaluation, and Detection use
the real checkpoint path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
REALISTIC_PROCESSING_SIZE = 1536
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch
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


def calibrated_profile_thresholds(
    staged: Mapping[str, object],
    class_names: Sequence[str],
) -> dict[str, float]:
    """Return positive, class-ordered thresholds from Evaluation output."""

    try:
        rows = staged["thresholds"]["per_class"]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise ValueError("Evaluation threshold output is missing per-class rows") from error
    if not isinstance(rows, (list, tuple)):
        raise ValueError("Evaluation threshold output per_class must be a sequence")
    by_class = {
        row.get("class_name"): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("class_name"), str)
    }
    thresholds: dict[str, float] = {}
    for class_name in class_names:
        row = by_class.get(class_name)
        if row is None or "threshold" not in row:
            raise ValueError(f"Evaluation threshold output is missing class: {class_name}")
        threshold = float(row["threshold"])
        if not np.isfinite(threshold) or threshold <= 0.0 or threshold > 1.0:
            raise ValueError(
                f"Evaluation threshold for {class_name} must be positive and at most 1"
            )
        thresholds[class_name] = threshold
    return thresholds


def validate_demo_regions(
    region_masks: Mapping[str, np.ndarray],
    coverage: np.ndarray,
) -> None:
    """Reject Demo output that cannot show a separated class region."""

    coverage_values = np.asarray(coverage, dtype=bool)
    covered_count = int(np.count_nonzero(coverage_values))
    if covered_count == 0:
        raise RuntimeError("Demo Detection has no covered source pixels")
    for class_name, region_mask in region_masks.items():
        values = np.asarray(region_mask, dtype=bool)
        if values.shape != coverage_values.shape:
            raise ValueError(f"Demo region shape mismatch for class: {class_name}")
        active_count = int(np.count_nonzero(values & coverage_values))
        if active_count == 0:
            raise RuntimeError(f"Demo region is empty for class: {class_name}")
        if active_count == covered_count:
            raise RuntimeError(f"Demo region covers all covered pixels for class: {class_name}")


def source_configuration(
    source_kind: str,
    width: int,
    height: int,
) -> dict[str, tuple[int, int]]:
    """Return the source/grid/window contract used by one Demo source."""

    if source_kind not in {"synthetic", "realistic"}:
        raise ValueError(f"unsupported Demo source kind: {source_kind}")
    if width < 1 or height < 1:
        raise ValueError("Demo source dimensions must be positive")
    cell_size = 32 if source_kind == "synthetic" else 512
    return {
        "source_size": (width, height),
        "grid_size": (cell_size, cell_size),
        "window_size": (cell_size, cell_size),
        "stride": (
            cell_size if source_kind == "realistic" else cell_size // 2,
            cell_size if source_kind == "realistic" else cell_size // 2,
        ),
    }


_REALISTIC_TRAIN_CENTER_GRAYS: tuple[int, ...] = (132, 138, 144, 150, 156, 162, 168, 174)
_REALISTIC_TRAIN_SCRATCH_CELLS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 1),
    (1, 0),
    (1, 1),
)


def build_realistic_corpus(
    root: Path,
    source: QImage,
) -> tuple[tuple[Path, dict[tuple[int, int], tuple[str, ...]], str, str], ...]:
    """Write three isolated generated base families and their Grid class labels."""

    if source.isNull():
        raise ValueError("realistic corpus source image is null")
    if source.width() < 3 * 512 or source.height() < 3 * 512:
        raise ValueError("realistic corpus source must contain at least a 3x3 512px grid")
    root.mkdir(parents=True, exist_ok=True)
    grid_columns = source.width() // 512
    grid_rows = source.height() // 512
    if grid_columns != grid_rows:
        raise ValueError("realistic corpus source must use a square grid")

    entries: list[tuple[Path, dict[tuple[int, int], tuple[str, ...]], str, str]] = []
    for index, center_gray in enumerate(_REALISTIC_TRAIN_CENTER_GRAYS):
        variant = _smooth_generated_wafer(source.width(), source.height(), center_gray)
        painter = QPainter(variant)
        inset = 82 + index * 17
        ring_gray = 94 + index * 3
        painter.setPen(QPen(QColor(ring_gray, ring_gray, ring_gray), 2 + index % 3))
        painter.drawEllipse(
            QRectF(inset, inset + index * 4, source.width() - inset * 2, source.height() - inset * 2)
        )
        painter.setPen(QPen(QColor(232, 232, 232), 4 + index % 2))
        painter.drawPoint(QPointF(610 + index * 29, 350 + index * 31))
        painter.end()
        annotations = {(1, 1): ("particle",), (2, 1): ("particle",)}
        if index < len(_REALISTIC_TRAIN_CENTER_GRAYS) - 1:
            scratch_row, scratch_column = _REALISTIC_TRAIN_SCRATCH_CELLS[index]
            _draw_realistic_scratch(variant, scratch_row, scratch_column, "train-a", index)
            existing_codes = annotations.get((scratch_row, scratch_column), ())
            annotations[(scratch_row, scratch_column)] = (*existing_codes, "scratch")
        output_path = root / f"realistic-train-base-{index:02d}.png"
        if not variant.save(str(output_path), "PNG"):
            raise RuntimeError(f"failed to write realistic corpus image: {output_path}")
        entries.append((output_path, annotations, "train-a", "train"))

    validation_base = _smooth_generated_wafer(source.width(), source.height(), 178)
    validation_scratch_cell = (1, 0)
    _draw_realistic_scratch(validation_base, *validation_scratch_cell, "validation-b", 0)
    validation_path = root / "realistic-validation-b.png"
    if not validation_base.save(str(validation_path), "PNG"):
        raise RuntimeError(f"failed to write realistic corpus image: {validation_path}")
    entries.append(
        (
            validation_path,
            {
                validation_scratch_cell: ("scratch",),
                (1, 1): ("particle",),
                (2, 1): ("particle",),
            },
            "validation-b",
            "validation",
        )
    )

    test_base = _smooth_generated_wafer(source.width(), source.height(), 146)
    painter = QPainter(test_base)
    painter.setPen(QPen(QColor(112, 112, 112), 3))
    painter.drawEllipse(QRectF(180, 220, source.width() - 360, source.height() - 440))
    painter.end()
    test_scratch_cell = (1, 2)
    _draw_realistic_scratch(test_base, *test_scratch_cell, "test-c", 0)
    test_annotations = {
        test_scratch_cell: ("scratch",),
        (1, 1): ("particle",),
        (2, 1): ("particle",),
    }
    test_path = root / "realistic-test-c.png"
    if not test_base.save(str(test_path), "PNG"):
        raise RuntimeError(f"failed to write realistic corpus image: {test_path}")
    entries.append((test_path, test_annotations, "test-c", "test"))
    return tuple(entries)


def _draw_realistic_scratch(
    image: QImage,
    row: int,
    column: int,
    family: str,
    variant: int,
) -> None:
    """Add one deterministic dark stroke inside a non-corner Annotation Grid."""

    left = column * 512
    top = row * 512
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QPen(QColor(18, 18, 18), 9))
    if family == "train-a":
        train_lines = (
            ((72, 96), (408, 416)),
            ((100, 410), (380, 140)),
            ((60, 240), (440, 280)),
            ((250, 60), (280, 450)),
            ((130, 150), (370, 360)),
            ((140, 360), (390, 190)),
            ((70, 330), (430, 240)),
        )
        start, end = train_lines[variant]
        painter.drawLine(
            QPointF(left + start[0], top + start[1]),
            QPointF(left + end[0], top + end[1]),
        )
    elif family == "validation-b":
        painter.drawLine(QPointF(left + 84, top + 120), QPointF(left + 354, top + 354))
    elif family == "test-c":
        painter.drawLine(QPointF(left + 118, top + 382), QPointF(left + 438, top + 154))
    else:
        raise ValueError(f"unknown scratch base family: {family}")
    painter.end()


def _smooth_generated_wafer(width: int, height: int, center_gray: int) -> QImage:
    image = QImage(width, height, QImage.Format.Format_Grayscale8)
    image.fill(24)
    painter = QPainter(image)
    gradient = QRadialGradient(QPointF(width / 2, height / 2), min(width, height) * 0.48)
    gradient.setColorAt(0.0, QColor(center_gray, center_gray, center_gray))
    gradient.setColorAt(1.0, QColor(72, 72, 72))
    painter.setBrush(gradient)
    painter.setPen(QPen(QColor(205, 205, 205), 3))
    painter.drawEllipse(QRectF(width * 0.02, height * 0.02, width * 0.96, height * 0.96))
    painter.setPen(QPen(QColor(238, 238, 238), 7))
    painter.drawPoint(QPointF(width * 0.58, height * 0.48))
    painter.drawPoint(QPointF(width * 0.42, height * 0.72))
    painter.end()
    return image


def main() -> int:
    args = _arguments()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    with TemporaryDirectory(prefix="wafer-demo-") as temporary:
        (
            project_path,
            asset,
            source_path,
            grid_profile,
            source_kind,
            original_source_size,
            realistic_corpus,
        ) = _seed_project(
            Path(temporary),
            app,
            args.source_image,
        )
        device = "cuda" if source_kind == "realistic" else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("realistic Demo requires an available CUDA device")
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

            annotations = _prepare_annotation(
                window,
                project_path,
                asset,
                grid_profile,
                app,
                source_kind=source_kind,
                realistic_corpus=realistic_corpus,
            )
            window.workspace_actions["Annotate"].trigger()
            _capture(window, output / "02-annotation-two-classes.png", app)

            snapshot_id, split_id = _create_dataset(
                project_path,
                realistic_corpus=realistic_corpus,
            )
            _activate_image(window, project_path, asset, grid_profile, "Dataset Snapshot created")
            window.workspace_actions["Dataset"].trigger()
            _select_dataset_scope(window, app)
            _capture(window, output / "03-dataset-snapshot.png", app)

            _activate_image(window, project_path, asset, grid_profile, "Training inputs ready")
            window.workspace_actions["Train"].trigger()
            training = _run_training(
                window,
                snapshot_id,
                split_id,
                app,
                source_kind=source_kind,
            )
            if training.artifact_path is None:
                raise RuntimeError("demo Training completed without a published checkpoint")
            checkpoint_path = training.artifact_path / "model.pt"
            validate_project_checkpoint(checkpoint_path)
            if not (training.artifact_path / "training_input_bundle.json").is_file():
                raise RuntimeError("demo Training artifact is missing the real input bundle")
            _capture(window, output / "04-training-complete.png", app)

            evaluation = _run_evaluation(
                project_path,
                training.run_id,
                snapshot_id,
                split_id,
                Path(temporary),
                device=device,
            )
            record_evaluation_decision(
                project_path,
                evaluation.evaluation_id,
                "validated",
                actor="Demo Engineer",
                notes=f"{source_kind.capitalize()} demo validation.",
            )
            record_evaluation_decision(
                project_path,
                evaluation.evaluation_id,
                "approved",
                actor="Demo Engineer",
                notes=f"{source_kind.capitalize()} demo approved for heatmap visualization.",
            )
            profile_thresholds = calibrated_profile_thresholds(
                {"thresholds": evaluation.thresholds},
                ("scratch", "particle"),
            )
            profile_thresholds = {
                class_name: max(threshold, 0.5)
                for class_name, threshold in profile_thresholds.items()
            }
            detection_profile = create_detection_profile(
                project_path,
                evaluation_id=evaluation.evaluation_id,
                training_run_id=training.run_id,
                window_size=(grid_profile.cell_width, grid_profile.cell_height),
                stride=(
                    (grid_profile.cell_width, grid_profile.cell_height)
                    if source_kind == "realistic"
                    else (grid_profile.cell_width // 2, grid_profile.cell_height // 2)
                ),
                reflect_padding=True,
                thresholds=profile_thresholds,
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
                timeout=600 if source_kind == "realistic" else 120,
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
            if source_kind == "realistic":
                validate_demo_regions(
                    generate_confidence_regions(artifact, detection_profile),
                    artifact.coverage,
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
                source_kind,
                original_source_size,
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
    parser.add_argument(
        "--source-image",
        type=Path,
        default=None,
        help="optional grayscale wafer source; omitted uses the small synthetic source",
    )
    return parser.parse_args()


def _seed_project(root: Path, app: QApplication, source_image: Path | None = None):
    project_path = root / "project"
    project.create_project(project_path)
    realistic_corpus = ()
    if source_image is None:
        source_kind = "synthetic"
        source_path = root / "synthetic-wafer.png"
        source = _synthetic_wafer()
        if not source.save(str(source_path), "PNG"):
            raise RuntimeError("failed to write synthetic wafer")
    else:
        source_kind = "realistic"
        source_path = source_image.expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source = QImage(str(source_path))
        if source.isNull():
            raise RuntimeError(f"failed to read source image: {source_path}")
        original_source_size = (source.width(), source.height())
        if max(original_source_size) > REALISTIC_PROCESSING_SIZE:
            source = source.scaled(REALISTIC_PROCESSING_SIZE, REALISTIC_PROCESSING_SIZE)
        realistic_corpus = build_realistic_corpus(root, source)
        source_path = realistic_corpus[0][0]
    if source_kind == "synthetic":
        original_source_size = (source.width(), source.height())
    configuration = source_configuration(source_kind, source.width(), source.height())

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
        profile = save_grid_profile(project_path, *configuration["grid_size"])
        set_image_grid_origin(project_path, asset.image_asset_id, profile.grid_profile_id, 0, 0)
        center_x = source.width() // 2
        center_y = source.height() // 2
        radius = 42 if source_kind == "synthetic" else int(min(source.width(), source.height()) * 0.49)
        set_effective_ellipse(
            project_path,
            asset.image_asset_id,
            center_x,
            center_y,
            radius,
            radius,
        )
        confirm_effective_wafer_area(project_path, asset.image_asset_id)
        for index in range(1, 10):
            if source_kind == "synthetic":
                extra_path = root / f"synthetic-wafer-{index:02d}.png"
                extra = _synthetic_wafer(index)
                if not extra.save(str(extra_path), "PNG"):
                    raise RuntimeError(f"failed to write {extra_path}")
            else:
                extra_path = realistic_corpus[index][0]
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
                center_x,
                center_y,
                radius,
                radius,
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
        return (
            project_path,
            asset,
            source_path,
            profile,
            source_kind,
            original_source_size,
            realistic_corpus,
        )
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


def _prepare_annotation(
    window,
    project_path,
    asset,
    profile,
    app,
    *,
    source_kind="synthetic",
    realistic_corpus=(),
):
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
    if source_kind == "realistic":
        if len(realistic_corpus) != len(assets):
            raise RuntimeError("realistic corpus and registered image count differ")
        annotations_by_path = {
            path.resolve(): current_annotations
            for path, current_annotations, _family, _split in realistic_corpus
        }
        per_asset_annotations = []
        for current in assets:
            try:
                per_asset_annotations.append(annotations_by_path[current.path.resolve()])
            except KeyError as error:
                raise RuntimeError(
                    f"realistic corpus is missing annotations for image: {current.path}"
                ) from error
    else:
        per_asset_annotations = [annotations] * len(assets)
    for current, current_annotations in zip(assets, per_asset_annotations, strict=True):
        for (row, column), class_codes in current_annotations.items():
            save_grid_annotation(
                project_path,
                GridAnnotation(current.image_asset_id, row, column, class_codes),
            )
        mark_image_reviewed(project_path, current.image_asset_id)
    if source_kind == "realistic":
        annotations = annotations_by_path[asset.path.resolve()]
    window._image_view.set_annotations(annotations)
    window.set_selected_defect_classes(("scratch", "particle"))
    window.set_annotation_mode("Annotate")
    window._refresh_review_controls()
    app.processEvents()
    return tuple(
        GridAnnotation(asset.image_asset_id, row, column, class_codes)
        for (row, column), class_codes in annotations.items()
    )


def select_family_isolated_split_seed(
    image_ids_by_family: Mapping[str, Sequence[str]],
    expected_split_by_family: Mapping[str, str],
) -> int:
    """Find the first seed whose image-level split keeps each Demo family isolated."""

    entries = tuple(
        (image_id, family)
        for family, image_ids in image_ids_by_family.items()
        for image_id in image_ids
    )
    validation_index = len(entries) - (len(entries) // 10) * 2
    test_index = len(entries) - len(entries) // 10
    for seed in range(100_000):
        ranked = sorted(
            entries,
            key=lambda item: hashlib.sha256(f"{seed}\0{item[0]}".encode("utf-8")).digest(),
        )
        actual: dict[str, set[str]] = {}
        for index, (_image_id, family) in enumerate(ranked):
            split = (
                "train"
                if index < validation_index
                else "validation"
                if index < test_index
                else "test"
            )
            actual.setdefault(family, set()).add(split)
        if actual == {
            family: {split} for family, split in expected_split_by_family.items()
        }:
            return seed
    raise RuntimeError("unable to isolate Demo base families in Dataset Split")


def _create_dataset(
    project_path: Path,
    *,
    realistic_corpus=(),
) -> tuple[str, str]:
    snapshot_id, _preview = create_project_dataset_snapshot(
        project_path,
        ("demo-line",),
        ("scratch", "particle"),
    )
    seed = 42
    if realistic_corpus:
        family_by_path = {
            path.resolve(): (family, expected_split)
            for path, _annotations, family, expected_split in realistic_corpus
        }
        image_ids_by_family: dict[str, list[str]] = {}
        expected_split_by_family: dict[str, str] = {}
        for reopened in image_asset.load_image_assets(project_path):
            family, expected_split = family_by_path[reopened.asset.path.resolve()]
            image_ids_by_family.setdefault(family, []).append(reopened.asset.image_asset_id)
            expected_split_by_family[family] = expected_split
        seed = select_family_isolated_split_seed(
            image_ids_by_family,
            expected_split_by_family,
        )
    split = create_dataset_split(project_path, snapshot_id, seed)
    if not split.train_image_ids or not split.validation_image_ids or not split.test_image_ids:
        raise RuntimeError("demo Dataset Split must contain train, validation, and test images")
    return snapshot_id, split.split_id


def _select_dataset_scope(window: MainWindow, app: QApplication) -> None:
    controls = window._training_scope_controls
    controls.set_selected_data_groups(("demo-line",))
    controls.set_selected_classes(("scratch", "particle"))
    window._refresh_dataset_preview()
    app.processEvents()


def configure_training_weights_policy(window, source_kind: str) -> str:
    """Select the Demo's explicit model-weight policy through the Training UI."""

    if source_kind not in {"synthetic", "realistic"}:
        raise ValueError(f"unsupported Demo source kind: {source_kind}")
    policy = "imagenet" if source_kind == "realistic" else "none"
    combo = window.findChild(QComboBox, "trainingWeightsPolicyComboBox")
    if combo is None:
        raise RuntimeError("Training weights policy control is missing")
    combo.setCurrentText(policy)
    return policy


def configure_training_epochs(window, source_kind: str) -> int:
    """Select the Demo's source-specific epoch count through the Training UI."""

    if source_kind not in {"synthetic", "realistic"}:
        raise ValueError(f"unsupported Demo source kind: {source_kind}")
    epochs = 20 if source_kind == "realistic" else 3
    spinbox = window.findChild(QSpinBox, "trainingEpochsSpinBox")
    if spinbox is None:
        raise RuntimeError("Training epochs control is missing")
    spinbox.setValue(epochs)
    return epochs


def _run_training(
    window: MainWindow,
    snapshot_id: str,
    split_id: str,
    app: QApplication,
    *,
    source_kind: str = "synthetic",
):
    _select_combo(window, "trainingSnapshotComboBox", snapshot_id)
    _select_combo(window, "trainingSplitComboBox", split_id)
    configure_training_epochs(window, source_kind)
    window.findChild(QSpinBox, "trainingBatchSizeSpinBox").setValue(
        4 if source_kind == "realistic" else 2
    )
    window.findChild(QSpinBox, "trainingSeedSpinBox").setValue(42)
    window.findChild(QComboBox, "trainingDeviceComboBox").setCurrentText(
        "cuda" if source_kind == "realistic" else "cpu"
    )
    configure_training_weights_policy(window, source_kind)
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
        timeout=600 if source_kind == "realistic" else 120,
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
    *,
    device: str = "cpu",
):
    calibration_request = build_checkpoint_evaluation_request(
        project_path,
        training_run_id,
        root / "evaluation-calibration-stage",
        request_id="demo-evaluation-calibration-worker",
        split="validation",
        policies="max_f1",
        criteria={"minimum_recall_target": 0.0},
        notes="Checkpoint-backed validation threshold calibration.",
        device=device,
    )
    calibration_handle = start_evaluation_worker(calibration_request)
    calibration_terminal = _collect_worker(calibration_handle, decode_evaluation_message)
    if not isinstance(calibration_terminal, EvaluationTerminal) or calibration_terminal.status != "completed":
        raise RuntimeError(f"Evaluation threshold calibration failed: {calibration_terminal}")
    calibration_payload = json.loads(
        (Path(calibration_request.staging_path) / "metrics.json").read_text(encoding="utf-8")
    )
    profile_thresholds = calibrated_profile_thresholds(
        calibration_payload,
        ("scratch", "particle"),
    )
    request = build_checkpoint_evaluation_request(
        project_path,
        training_run_id,
        root / "evaluation-stage",
        request_id="demo-evaluation-worker",
        split="test",
        thresholds=tuple(profile_thresholds.values()),
        criteria={"minimum_recall_target": 0.0},
        notes="Checkpoint-backed two-class demo evaluation.",
        device=device,
    )
    handle = start_evaluation_worker(request)
    terminal = _collect_worker(handle, decode_evaluation_message)
    if not isinstance(terminal, EvaluationTerminal) or terminal.status != "completed":
        raise RuntimeError(f"Evaluation worker failed: {terminal}")
    evaluation = create_evaluation_from_staged(
        project_path,
        request.staging_path,
        training_run_id=training_run_id,
        actor="Demo Engineer",
        evaluation_id="demo-evaluation",
    )
    return evaluation


def _run_detection(window, project_path, asset, profile, app, *, timeout: float = 120):
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
        timeout=timeout,
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
    source_kind,
    original_source_size,
):
    training = load_training_run(project_path, training_id)
    evaluation = load_evaluation(project_path, evaluation_id)
    detection = load_detection_run(project_path, detection_id)
    checkpoint_path = training.artifact_path / "model.pt"
    checkpoint = validate_project_checkpoint(checkpoint_path)
    checkpoint_checksum = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    finite = np.asarray(artifact.class_map("scratch"))[np.isfinite(artifact.class_map("scratch"))]
    return {
        "synthetic_source": source_kind == "synthetic",
        "source_kind": source_kind,
        "model_backed": True,
        "source": {
            "width": source_image.width(),
            "height": source_image.height(),
            "format": "uint8 PNG",
            "processing_proxy": source_kind == "realistic" and original_source_size != (source_image.width(), source_image.height()),
        },
        "source_original": (
            {"width": original_source_size[0], "height": original_source_size[1], "format": "uint8 PNG"}
            if source_kind == "realistic"
            else None
        ),
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
            (
                "Generated 20MP source was resized to a 1536px processing proxy; fixed grid labels are for pipeline validation only."
                if source_kind == "realistic"
                else "Synthetic source images and fixed grid labels are for pipeline validation only."
            ),
            "Metrics are observed on one held-out image and are not production accuracy claims.",
            "Confidence regions are approximate localization, not a segmentation mask.",
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
