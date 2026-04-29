import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PySide6.QtGui import QImage

from wafer_defect_studio.cam_detection import generate_all_convolutional_artifact
from wafer_defect_studio.detection_windows import Rect, enumerate_inference_windows
from wafer_defect_studio.grid_geometry import annotation_grids
from wafer_defect_studio.proposal_conversion import preview_proposal_conversion
from wafer_defect_studio.proposal_generation import (
    generate_confidence_regions,
    generate_proposals,
)
from wafer_defect_studio.proposal_queue import build_review_queue
from wafer_defect_studio.proposal_review import ProposalReviewRevision
from wafer_defect_studio.result_export import build_reviewed_rows, export_result_bundle


class PatchDetectionServiceFlowTest(unittest.TestCase):
    def test_v3_patch_map_reuses_proposal_review_conversion_and_export_services(self):
        windows = enumerate_inference_windows(4, 4, window_size=2, stride=2)
        artifact = generate_all_convolutional_artifact(
            windows,
            ((0.9,), (0.1,), (0.1,), (0.1,)),
            class_names=("scratch",),
            window_settings={"window_size": [2, 2], "stride": [2, 2]},
            provenance={
                "run_id": "detection-v3",
                "profile_id": "profile-v3",
                "evaluation_id": "evaluation-v3",
                "checkpoint_format": "wafer_defect_studio.resnet18.v3",
            },
            model_id="patch-model",
            profile_id="profile-v3",
            evaluation_id="evaluation-v3",
        )
        settings = {
            "thresholds": {"scratch": 0.5},
            "map_generation": {"closing_radius": 0, "minimum_area": 1},
        }

        regions = generate_confidence_regions(artifact, settings)
        proposals = generate_proposals(artifact, settings)

        self.assertEqual(artifact.window_settings["map_method"], "patch_classification_sigmoid")
        self.assertIn("Approximate localization", artifact.disclaimers)
        self.assertEqual(int(regions["scratch"].sum()), 4)
        self.assertEqual(len(proposals), 1)
        proposal = proposals[0]
        self.assertEqual(tuple(proposal.source_rect), (0, 0, 2, 2))
        self.assertEqual((proposal.area, proposal.peak_confidence), (4, 0.9))
        self.assertAlmostEqual(proposal.mean_confidence, 0.9)
        self.assertEqual(proposal.provenance["run_id"], "detection-v3")
        self.assertEqual(proposal.provenance["checkpoint_format"], "wafer_defect_studio.resnet18.v3")

        grids = annotation_grids(4, 4, 2, 2)
        frozen_grids = tuple(grids)
        revision = ProposalReviewRevision(
            proposal.proposal_id,
            1,
            "accepted",
            proposal.source_rect,
            {"actor": "engineer"},
        )
        queue = build_review_queue(proposals, (revision,))
        preview = preview_proposal_conversion(queue, grids)
        self.assertEqual(grids, frozen_grids)
        self.assertEqual(preview.source_proposal_ids, (proposal.proposal_id,))
        self.assertEqual([(cell.row, cell.column) for cell in preview.cells], [(0, 0)])

        rows = build_reviewed_rows(
            proposals,
            (revision,),
            project_id="project-v3",
            image_asset_id="wafer-v3",
            run_id="detection-v3",
            profile_id="profile-v3",
            thresholds={"scratch": 0.5},
        )
        source = QImage(4, 4, QImage.Format.Format_Grayscale8)
        source.fill(80)
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            destinations = tuple(root / f"result.{suffix}" for suffix in ("csv", "json", "png"))
            result = export_result_bundle(
                *destinations,
                source,
                rows,
                selected_class="scratch",
                confidence_map=artifact.class_map("scratch"),
                region_mask=regions["scratch"],
                grid_rects=tuple(
                    Rect(cell.grid.x, cell.grid.y, cell.grid.width, cell.grid.height)
                    for cell in preview.cells
                ),
            )

            self.assertTrue(result.success, result.error)
            self.assertTrue(all(path.read_bytes() for path in destinations))
            exported = json.loads(destinations[1].read_text(encoding="utf-8"))
            self.assertEqual(exported["metadata"]["run_id"], "detection-v3")
            self.assertEqual(
                exported["proposals"][0]["proposal_provenance"]["checkpoint_format"],
                "wafer_defect_studio.resnet18.v3",
            )
            rendered = QImage(str(destinations[2]))
            self.assertIn("approximate", rendered.text("localization_warning").lower())


if __name__ == "__main__":
    unittest.main()
