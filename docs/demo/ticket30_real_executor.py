"""Real project/model adapter for the frozen Ticket 30 CUDA evidence runner."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from docs.demo.hard_negative_selection import (
    hard_negative_selection_json,
    resolve_hard_negative_selection,
    select_hard_negative_bags,
)
from docs.demo.ticket30_evidence_corpus import (
    CLASS_CODES,
    build_ticket30_evidence_corpus,
    render_ticket30_evidence_pixels,
)
from docs.demo.ticket30_map_thresholds import calibrate_map_thresholds, canonical_json
from docs.demo.wafer_quality_evidence import WaferEvidenceCase, compute_wafer_quality_evidence
from wafer_defect_studio.grid_geometry import annotation_grids


def create_ticket30_real_executor(backend=None):
    """Return one stateful executor; one prepared project is shared by a seed."""

    return _Ticket30Executor(backend or _ProjectBackend())


class _Ticket30Executor:
    def __init__(self, backend):
        self.backend = backend
        self.states = {}

    def __call__(self, stage: str, run: Mapping[str, object], run_dir: Path, source: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        key = (int(run["seed"]), str(source.resolve()))
        if key not in self.states:
            self.states[key] = {"context": self.backend.prepare(key[0], source, run_dir.parent)}
        state = self.states[key]
        context = state["context"]
        model = str(run["model"])

        if stage == "train":
            checkpoint = self.backend.train(context, run, run_dir / "probe", priority_ids=())
            produced = {}
            if model == "spatial_mil_v4":
                validation = self.backend.detect(context, checkpoint, "validation", None, run, run_dir / "probe-validation")
                thresholds_payload = calibrate_map_thresholds(validation, CLASS_CODES, str(run["score_domain"]))
                thresholds = _threshold_values(thresholds_payload)
                training = self.backend.detect(context, checkpoint, "train", None, run, run_dir / "probe-train")
                selection = select_hard_negative_bags(
                    training, self.backend.training_bundle(context), CLASS_CODES, thresholds,
                    max_bags=int(run["hard_negative_max_bags"]),
                )
                selection_text = hard_negative_selection_json(selection)
                selection_path = run_dir / "hard-negative-selection.json"
                selection_path.write_text(selection_text, encoding="utf-8")
                selection_sha = hashlib.sha256(selection_text.encode("utf-8")).hexdigest()
                priority_ids = resolve_hard_negative_selection(
                    selection, CLASS_CODES, int(run["hard_negative_max_bags"])
                )
                # This is a fresh 20+5 request, never a resumed Training Run.
                checkpoint = self.backend.train(
                    context, run, run_dir / "refined", priority_ids=priority_ids,
                    selection_sha256=selection_sha,
                )
                produced["hard_negative_selection"] = selection_path
            destination = run_dir / "checkpoint.pt"
            shutil.copyfile(checkpoint, destination)
            state[model] = {"checkpoint": destination}
            return {"checkpoint": destination, **produced}

        checkpoint = state.get(model, {}).get("checkpoint")
        if checkpoint is None:
            raise RuntimeError(f"train stage has not completed for model={model}")
        if stage == "threshold":
            cases = self.backend.detect(context, checkpoint, "validation", None, run, run_dir / "validation")
            payload = calibrate_map_thresholds(cases, CLASS_CODES, str(run["score_domain"]))
            path = run_dir / "threshold.json"
            path.write_text(canonical_json(payload), encoding="utf-8")
            state[model]["thresholds"] = _threshold_values(payload)
            return {"threshold": path}
        if stage == "map":
            heldout = tuple(case for case in build_ticket30_evidence_corpus() if case.seed == int(run["seed"]))
            cases = self.backend.detect(context, checkpoint, "test", heldout, run, run_dir / "test")
            path = run_dir / "maps.npz"
            np.savez_compressed(
                path,
                maps=np.stack([case.absolute_maps for case in cases]),
                filenames=np.asarray([case.filename for case in cases]),
                class_codes=np.asarray(CLASS_CODES),
                pixel_sha256=np.asarray([case.pixel_sha256 for case in heldout]),
            )
            return {"map": path}
        if stage == "metrics":
            thresholds = state.get(model, {}).get("thresholds")
            if thresholds is None:
                raise RuntimeError(f"threshold stage has not completed for model={model}")
            corpus = {case.filename: case for case in build_ticket30_evidence_corpus() if case.seed == int(run["seed"])}
            with np.load(run_dir / "maps.npz", allow_pickle=False) as artifact:
                codes = tuple(str(value) for value in artifact["class_codes"])
                filenames = tuple(str(value) for value in artifact["filenames"])
                pixel_hashes = tuple(str(value) for value in artifact["pixel_sha256"])
                maps = artifact["maps"]
            if codes != CLASS_CODES:
                raise ValueError(f"map artifact class order drift: {codes!r}")
            cases = []
            for filename, pixel_hash, absolute_maps in zip(filenames, pixel_hashes, maps, strict=True):
                truth = corpus.get(filename)
                if truth is None or truth.pixel_sha256 != pixel_hash:
                    raise ValueError(f"map artifact held-out identity drift: {filename}")
                cases.append(WaferEvidenceCase(
                    filename, "test", truth.oracle,
                    annotation_grids(truth.oracle.image_width, truth.oracle.image_height, 512, 512),
                    absolute_maps,
                ))
            payload = compute_wafer_quality_evidence(cases, CLASS_CODES, thresholds)
            path = run_dir / "metrics.json"
            path.write_text(canonical_json(payload), encoding="utf-8")
            return {"metrics": path}
        raise ValueError(f"unsupported Ticket 30 stage: {stage!r}")


def _threshold_values(payload: Mapping[str, object]) -> dict[str, float]:
    selected = payload["selected"]
    return {code: float(selected[code]["threshold"]) for code in CLASS_CODES}


def _phase2_project_helpers():
    demo_dir = str(Path(__file__).resolve().parent)
    if demo_dir not in sys.path:
        sys.path.insert(0, demo_dir)
    from docs.demo.run_phase2_demo import _create_dataset, _prepare_annotation, _seed_project

    return _seed_project, _prepare_annotation, _create_dataset


@dataclass
class _ProjectContext:
    project_path: Path
    snapshot_id: str
    split_id: str
    realistic_corpus: tuple
    latest_bundle: object | None = None


class _ProjectBackend:
    """Thin production calls into the already-tested desktop services."""

    def prepare(self, seed: int, source_image: Path, work_dir: Path) -> _ProjectContext:
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
        from wafer_defect_studio.main_window import MainWindow

        _seed_project, _prepare_annotation, _create_dataset = _phase2_project_helpers()
        from docs.demo.run_phase2_demo import _activate_image
        app = QApplication.instance() or QApplication([])
        root = work_dir / f"seed-{seed}-project"
        project_path, asset, _source, profile, source_kind, _size, corpus = _seed_project(root, app, source_image)
        window = MainWindow(settings=QSettings(str(root / "annotation.ini"), QSettings.Format.IniFormat))
        window.show()
        try:
            _activate_image(window, project_path, asset, profile, "Ticket 30 annotation")
            _prepare_annotation(
                window, project_path, asset, profile, app,
                source_kind=source_kind, realistic_corpus=corpus,
            )
        finally:
            window.close()
            window.deleteLater()
            app.processEvents()
        snapshot_id, split_id = _create_dataset(project_path, realistic_corpus=corpus)
        return _ProjectContext(project_path, snapshot_id, split_id, corpus)

    def train(
        self, context: _ProjectContext, run: Mapping[str, object], destination: Path,
        *, priority_ids: Sequence[str] = (), selection_sha256: str | None = None,
    ) -> Path:
        from wafer_defect_studio.training_input_bundle import create_training_input_bundle
        from wafer_defect_studio.training_protocol import TerminalMessage, TrainingConfig, TrainingRequest, decode_message
        from wafer_defect_studio.training_run import RunConfig, create_training_run, update_training_run_terminal, validate_project_checkpoint
        from wafer_defect_studio.training_worker import start_training_worker

        model = str(run["model"])
        patch = model != "cam_v2"
        spatial = model == "spatial_mil_v4"
        config = TrainingConfig(
            context.snapshot_id, context.split_id, len(CLASS_CODES), int(run["epochs"]), 4,
            device="cuda", seed=int(run["seed"]), learning_rate=0.001, weights_policy=str(run["weights"]),
            patch_size=128 if patch else None, patch_stride=64 if patch else None,
            training_policy="spatial_mil_v4" if spatial else "legacy",
            priority_normal_bag_ids=tuple(priority_ids), hard_negative_selection_sha256=selection_sha256,
        )
        run_config = RunConfig(
            context.snapshot_id, context.split_id, class_count=len(CLASS_CODES), batch_size=4,
            device="cuda", epochs=int(run["epochs"]), learning_rate=0.001, seed=int(run["seed"]),
            weights_policy=str(run["weights"]), patch_size=128 if patch else None,
            patch_stride=64 if patch else None, bag_pooling="max" if patch and not spatial else None,
            training_policy="spatial_mil_v4" if spatial else "legacy",
            priority_normal_bag_ids=tuple(priority_ids), hard_negative_selection_sha256=selection_sha256,
        )
        persisted = create_training_run(context.project_path, run_config)
        update_training_run_terminal(context.project_path, persisted.run_id, "running", message="Ticket 30 training started")
        bundle_path = persisted.staging_path / "training_input_bundle.json"
        bundle = create_training_input_bundle(
            context.project_path, context.snapshot_id, context.split_id, bundle_path,
            config=config if patch else None,
        )
        context.latest_bundle = bundle
        handle = start_training_worker(TrainingRequest(persisted.run_id, config, persisted.staging_path, bundle_path))
        while True:
            message = decode_message(handle.queue.get(timeout=7200))
            if isinstance(message, TerminalMessage):
                handle.join(timeout=60)
                update_training_run_terminal(
                    context.project_path, persisted.run_id, message.status, message=message.message,
                    error_code=message.error_code, staging_path=message.artifact_staging_path,
                )
                if message.status != "completed":
                    raise RuntimeError(f"Training failed: {message.status}: {message.message}")
                checkpoint = context.project_path / "runs" / persisted.run_id / "model.pt"
                validate_project_checkpoint(checkpoint, expected_class_codes=CLASS_CODES)
                return checkpoint

    def training_bundle(self, context: _ProjectContext):
        if context.latest_bundle is None:
            raise RuntimeError("training bundle is unavailable before training")
        return context.latest_bundle

    def detect(self, context, checkpoint, split, cases, run, destination):
        from PySide6.QtGui import QImage
        from docs.demo.run_phase2_demo import REALISTIC_CORPUS_MANIFEST, _collect_detection_worker
        from wafer_defect_studio.detection_controls import _load_staged_artifact
        from wafer_defect_studio.detection_worker import DetectionRequest, start_detection_worker
        from wafer_defect_studio.training_run import validate_project_checkpoint

        validate_project_checkpoint(checkpoint, expected_class_codes=CLASS_CODES)
        if cases is None:
            truth_by_name = {case.filename: case for case in REALISTIC_CORPUS_MANIFEST}
            selected = []
            for path, _labels, _family, current_split in context.realistic_corpus:
                if current_split == split:
                    truth = truth_by_name[path.name]
                    selected.append((truth.filename, truth.oracle, _qimage_pixels(QImage(str(path)))))
        else:
            selected = [(case.filename, case.oracle, render_ticket30_evidence_pixels(case.oracle)) for case in cases]
        geometry = run["detection_geometry"]
        result = []
        for index, (filename, oracle, pixels) in enumerate(selected):
            stage = destination / f"{index:03d}"
            request = DetectionRequest(
                request_id=f"ticket30-{run['seed']}-{run['model']}-{split}-{index}",
                source=pixels, staging_path=stage, run_id=f"ticket30-{run['seed']}-{run['model']}",
                profile_id=f"ticket30-{run['model']}", class_names=CLASS_CODES,
                window_size=int(geometry["window_size"]), stride=int(geometry["window_stride"]),
                reflect_padding=True, center_weighting="linear", device="cuda",
                checkpoint_path=checkpoint, batch_size=4,
                model_id=f"ticket30:{run['model']}",
            )
            terminal = _collect_detection_worker(start_detection_worker(request))
            if terminal.status != "completed":
                raise RuntimeError(f"Detection failed: filename={filename}: {terminal.status}: {terminal.message}")
            artifact = _load_staged_artifact(stage)
            result.append(WaferEvidenceCase(
                filename, split, oracle,
                annotation_grids(oracle.image_width, oracle.image_height, 512, 512), artifact.maps,
            ))
        return tuple(result)


def _qimage_pixels(image) -> np.ndarray:
    from PySide6.QtGui import QImage

    gray = image.convertToFormat(QImage.Format.Format_Grayscale8)
    if gray.isNull():
        raise RuntimeError("failed to decode Ticket 30 source image")
    return np.frombuffer(gray.bits(), dtype=np.uint8, count=gray.sizeInBytes()).reshape(
        gray.height(), gray.bytesPerLine()
    )[:, : gray.width()].copy()
