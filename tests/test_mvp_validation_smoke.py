import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import torch
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.annotation import GridAnnotation, save_grid_annotation
from wafer_defect_studio.autosave import AutosaveGuard
from wafer_defect_studio.dataset_snapshot import (
    DatasetSnapshotError,
    SamplingPolicy,
    create_dataset_snapshot,
)
from wafer_defect_studio.dataset_split import create_dataset_split
from wafer_defect_studio.defect_class import DefectClass, save_defect_classes
from wafer_defect_studio.detection_run import create_detection_profile, create_detection_run
from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.effective_area import (
    confirm_effective_wafer_area,
    set_effective_ellipse,
)
from wafer_defect_studio.evaluation_run import create_evaluation, record_evaluation_decision
from wafer_defect_studio.grid_profile import save_grid_profile
from wafer_defect_studio.image_grid_placement import set_image_grid_origin
from wafer_defect_studio.job_actions import FailureKind, diagnose_failure
from wafer_defect_studio.job_state import JobKind, JobSnapshot
from wafer_defect_studio.job_worker_bridge import apply_worker_message
from wafer_defect_studio.normalization import NormalizationBounds
from wafer_defect_studio.proposal_generation import DefectProposal
from wafer_defect_studio.proposal_review import load_proposal_revisions, record_review
from wafer_defect_studio.proposal_store import load_defect_proposal, save_defect_proposal
from wafer_defect_studio.result_export import build_reviewed_rows, export_result_bundle
from wafer_defect_studio.review import mark_image_reviewed
from wafer_defect_studio.training_run import (
    RunConfig,
    create_training_run,
    update_training_run_terminal,
)
from wafer_defect_studio.training_scope import (
    DataGroup,
    TrainingScope,
    assign_image_to_data_group,
    eligible_image_ids,
    save_data_groups,
    save_training_scope,
)


class MvpValidationSmokeTest(unittest.TestCase):
    def test_synthetic_project_walks_public_seams(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "project"
            project.create_project(project_path)

            source_path = workspace / "wafer.png"
            source = QImage(32, 24, QImage.Format_Grayscale8)
            source.fill(80)
            self.assertTrue(source.save(str(source_path), "PNG"))
            original_source = source_path.read_bytes()
            asset = image_asset.register_wafer_image(project_path, source_path)

            profile = save_grid_profile(project_path, 16, 12)
            placement = set_image_grid_origin(
                project_path, asset.image_asset_id, profile.grid_profile_id, 0, 0
            )
            self.assertEqual(placement.grid_profile_version, profile.version)
            set_effective_ellipse(project_path, asset.image_asset_id, 16, 12, 16, 12)
            confirmed_area = confirm_effective_wafer_area(project_path, asset.image_asset_id)
            self.assertTrue(confirmed_area.confirmed)

            save_defect_classes(
                project_path,
                (
                    DefectClass("scratch", "Scratch", "#cc4444", order=0),
                    DefectClass("stain", "Stain", "#4488cc", order=1),
                ),
            )
            annotation = GridAnnotation(
                asset.image_asset_id, 0, 0, ("scratch", "stain")
            )
            save_grid_annotation(project_path, annotation)
            self.assertEqual(mark_image_reviewed(project_path, asset.image_asset_id).reviewed, True)

            save_data_groups(project_path, (DataGroup("line-a", "Line A"),))
            assign_image_to_data_group(project_path, asset.image_asset_id, "line-a")
            save_training_scope(
                project_path,
                TrainingScope(("line-a",), ("scratch", "stain")),
            )
            self.assertEqual(eligible_image_ids(project_path), (asset.image_asset_id,))

            bounds = (NormalizationBounds("uint8", 0, 255, 10.0, 240.0, 1.0, 99.0),)
            snapshot = create_dataset_snapshot(
                project_path, bounds, SamplingPolicy(normal_to_positive_ratio=1.0)
            )
            split = create_dataset_split(project_path, snapshot.snapshot_id, 42)
            self.assertEqual(split.train_image_ids, (asset.image_asset_id,))

            training = create_training_run(
                project_path,
                RunConfig(snapshot.snapshot_id, split.split_id, class_count=2, device="cpu"),
                environment={"device": "cpu", "cuda": "not-used"},
                run_id="training-1",
            )
            model_path = training.staging_path / "model.pt"
            torch.save(
                {
                    "checkpoint_format": "wafer_defect_studio.resnet18.v1",
                    "architecture": "resnet18",
                    "class_count": 2,
                    "class_codes": ["scratch", "stain"],
                    "normalization_bounds": [
                        {
                            "dtype": "uint8",
                            "source_min": 0,
                            "source_max": 255,
                            "low": 10.0,
                            "high": 240.0,
                            "low_percentile": 10.0,
                            "high_percentile": 99.0,
                        }
                    ],
                    "input_size": {"width": 16, "height": 12},
                    "state_dict": {"fixture": torch.zeros(1)},
                },
                model_path,
            )
            manifest = {
                "required_files": ["model.pt"],
                "files": [
                    {
                        "path": "model.pt",
                        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    }
                ],
            }
            (training.staging_path / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            completed_training = update_training_run_terminal(
                project_path, training.run_id, "completed", metrics={"loss": 0.1}
            )
            self.assertEqual(completed_training.status, "completed")
            self.assertTrue(completed_training.artifact_path.is_dir())

            evaluation = create_evaluation(
                project_path,
                training.run_id,
                {"macro_f1": 0.9},
                {"target_satisfied": True},
                {"minimum_recall_target": 0.8},
                evaluation_id="evaluation-1",
            )
            record_evaluation_decision(
                project_path, evaluation.evaluation_id, "validated", actor="engineer", notes="validated"
            )
            record_evaluation_decision(
                project_path,
                evaluation.evaluation_id,
                "approved",
                actor="engineer",
                notes="approved for synthetic detection",
            )
            profile = create_detection_profile(
                project_path,
                evaluation_id=evaluation.evaluation_id,
                window_size=(16, 12),
                stride=(16, 12),
                reflect_padding=True,
                thresholds={"scratch": 0.6, "stain": 0.55},
                center_weighting="hann",
                map_generation={"smoothing": 1, "minimum_area": 1},
                profile_id="profile-1",
            )
            detection = create_detection_run(
                project_path,
                profile_id=profile.profile_id,
                evaluation_id=evaluation.evaluation_id,
                source_fingerprints={asset.image_asset_id: asset.fingerprint},
                provenance={"coordinate_system": "source-image-pixels", "window_count": 4},
                run_id="detection-1",
            )
            self.assertEqual(detection.status, "created")

            source_path.write_bytes(original_source + b"changed")
            with self.assertRaisesRegex(DatasetSnapshotError, "unchanged source"):
                create_dataset_snapshot(
                    project_path, bounds, SamplingPolicy(normal_to_positive_ratio=1.0)
                )
            source_path.write_bytes(original_source)

            proposal = DefectProposal(
                "proposal-1",
                "scratch",
                Rect(2, 3, 4, 5),
                20,
                0.91,
                0.74,
                {
                    "detection_run_id": detection.run_id,
                    "profile_id": profile.profile_id,
                    "image_asset_id": asset.image_asset_id,
                    "source_coordinate_system": "source-image-pixels",
                },
            )
            save_defect_proposal(
                project_path,
                proposal,
                detection_run_id=detection.run_id,
                profile_id=profile.profile_id,
            )
            review = record_review(
                project_path,
                proposal.proposal_id,
                "corrected",
                source_rect=Rect(3, 4, 5, 6),
                provenance={"manual_correction": True},
                actor="reviewer",
            )
            rows = build_reviewed_rows(
                (load_defect_proposal(project_path, proposal.proposal_id),),
                load_proposal_revisions(project_path, proposal.proposal_id),
                project_id="synthetic-project",
                image_asset_id=asset.image_asset_id,
                run_id=detection.run_id,
                profile_id=profile.profile_id,
                thresholds=profile.thresholds,
            )
            self.assertEqual(rows[0].source_rect, review.source_rect)
            self.assertEqual(rows[0].review_status, "corrected")

            source_image = QImage(str(source_path))
            confidence = np.full((24, 32), np.nan, dtype=np.float32)
            confidence[4, 3] = 0.9
            destinations = tuple(workspace / f"result.{suffix}" for suffix in ("csv", "json", "png"))
            bundle = export_result_bundle(
                *destinations,
                source_image,
                rows,
                selected_class="scratch",
                confidence_map=confidence,
            )
            self.assertTrue(bundle.success)
            self.assertEqual(bundle.paths, destinations)
            self.assertTrue(all(path.is_file() for path in destinations))

            failed_destinations = tuple(
                workspace / f"failed.{suffix}" for suffix in ("csv", "json", "png")
            )
            with patch(
                "wafer_defect_studio.result_export.export_proposals_json",
                side_effect=OSError("injected export writer failure"),
            ):
                failed_bundle = export_result_bundle(
                    *failed_destinations,
                    source_image,
                    rows,
                    selected_class="scratch",
                )
            self.assertFalse(failed_bundle.success)
            self.assertEqual(failed_bundle.paths, ())
            self.assertIn("injected export writer failure", failed_bundle.error or "")
            self.assertFalse(any(path.exists() for path in failed_destinations))

            autosave = AutosaveGuard(
                lambda _annotation: (_ for _ in ()).throw(OSError("autosave disk failure"))
            )
            self.assertFalse(autosave.autosave(annotation, ()))
            self.assertTrue(autosave.blocked)
            self.assertFalse(autosave.request_image_switch("other-image"))

            interrupted = apply_worker_message(
                JobSnapshot("job-interrupted", JobKind.TRAINING),
                {
                    "request_id": "job-interrupted",
                    "status": "cancelled",
                    "message": "worker killed",
                },
            )
            oom = apply_worker_message(
                JobSnapshot("job-oom", JobKind.TRAINING),
                {
                    "request_id": "job-oom",
                    "status": "failed",
                    "error_code": "out_of_memory",
                    "message": "CUDA out of memory",
                },
            )
            self.assertEqual(diagnose_failure(interrupted).kind, FailureKind.INTERRUPTED)
            self.assertEqual(diagnose_failure(oom).kind, FailureKind.OUT_OF_MEMORY)
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
