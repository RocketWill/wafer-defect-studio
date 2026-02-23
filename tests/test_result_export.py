import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.proposal_generation import DefectProposal
from wafer_defect_studio.proposal_review import ProposalReviewRevision
from wafer_defect_studio.result_export import (
    APPROXIMATE_LOCALIZATION_WARNING,
    CSV_COLUMNS,
    JSON_EXPORT_TYPE,
    JSON_SCHEMA_VERSION,
    build_reviewed_rows,
    export_proposals_csv,
    export_proposals_json,
    export_proposals_png,
)


class ResultExportTest(unittest.TestCase):
    def _reviewed_rows(self):
        proposal = DefectProposal(
            proposal_id="proposal-α",
            class_name="刮痕",
            source_rect=Rect(1, 2, 3, 4),
            area=12,
            peak_confidence=0.91,
            mean_confidence=0.72,
            provenance={"source_coordinate_system": "source-image-pixels"},
        )
        revision = ProposalReviewRevision(
            proposal_id=proposal.proposal_id,
            revision_number=2,
            status="corrected",
            source_rect=Rect(10, 20, 30, 40),
            provenance={"actor": "工程師", "reason": "重新框選"},
        )
        return build_reviewed_rows(
            (proposal,),
            (revision,),
            project_id="專案",
            image_asset_id="晶圓/α.png",
            run_id="run-immutable",
            profile_id="profile-immutable",
            thresholds={"刮痕": 0.65},
        )

    def test_csv_uses_latest_review_geometry_and_round_trips_unicode(self):
        rows = self._reviewed_rows()

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.source_rect, Rect(10, 20, 30, 40))
        self.assertEqual(row.review_status, "corrected")
        self.assertEqual(row.revision_number, 2)
        self.assertEqual(row.revision_provenance["actor"], "工程師")
        self.assertEqual(row.threshold, 0.65)
        self.assertEqual(row.localization_warning, APPROXIMATE_LOCALIZATION_WARNING)

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "結果.csv"
            written = export_proposals_csv(destination, rows)
            self.assertEqual(written, destination)
            with destination.open("r", encoding="utf-8", newline="") as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(tuple(records[0]), CSV_COLUMNS)
            self.assertEqual(records[0]["project_id"], "專案")
            self.assertEqual(records[0]["class_name"], "刮痕")
            self.assertEqual(records[0]["source_x"], "10")
            self.assertEqual(records[0]["source_height"], "40")
            self.assertEqual(records[0]["review_status"], "corrected")
            self.assertIn("工程師", records[0]["revision_provenance"])

    def test_json_is_versioned_deterministic_and_keeps_source_review_provenance(self):
        rows = self._reviewed_rows()
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "結果.json"
            second = Path(temporary) / "結果-反向.json"
            self.assertEqual(export_proposals_json(first, rows), first)
            self.assertEqual(export_proposals_json(second, tuple(reversed(rows))), second)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            raw = first.read_text(encoding="utf-8")
            self.assertIn("刮痕", raw)
            payload = json.loads(raw)
            self.assertEqual(payload["schema_version"], JSON_SCHEMA_VERSION)
            self.assertEqual(payload["export_type"], JSON_EXPORT_TYPE)
            metadata = payload["metadata"]
            self.assertEqual(metadata["schema_version"], JSON_SCHEMA_VERSION)
            self.assertEqual(metadata["project_id"], "專案")
            self.assertEqual(metadata["image_asset_id"], "晶圓/α.png")
            self.assertEqual(metadata["run_id"], "run-immutable")
            self.assertEqual(metadata["profile_id"], "profile-immutable")
            self.assertEqual(metadata["source_coordinate_system"], "source-image-pixels")
            self.assertIn("approximate", metadata["localization_warning"].lower())

            proposal = payload["proposals"][0]
            self.assertEqual(proposal["class_name"], "刮痕")
            self.assertEqual(proposal["threshold"], 0.65)
            self.assertEqual(proposal["source_rect"], {"x": 10, "y": 20, "width": 30, "height": 40})
            self.assertEqual(proposal["review_status"], "corrected")
            self.assertEqual(proposal["revision_number"], 2)
            self.assertEqual(proposal["revision_provenance"]["actor"], "工程師")

    def test_png_preserves_native_size_and_records_legend_without_mutating_source(self):
        app = QApplication.instance() or QApplication([])
        source = QImage(8, 6, QImage.Format_Grayscale8)
        source.fill(80)
        source_before = source.copy()
        confidence_map = np.full((6, 8), np.nan, dtype=np.float32)
        confidence_map[2, 3] = 0.9

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "影像結果.png"
            written = export_proposals_png(
                destination,
                source,
                self._reviewed_rows(),
                selected_class="刮痕",
                confidence_map=confidence_map,
                grid_rects=(Rect(0, 0, 4, 3),),
            )
            self.assertEqual(written, destination)
            self.assertTrue(source == source_before)

            rendered = QImage(str(destination))
            self.assertEqual((rendered.width(), rendered.height()), (8, 6))
            self.assertEqual(rendered.text("selected_class"), "刮痕")
            self.assertIn("confidence", rendered.text("legend").lower())
            self.assertIn("approximate", rendered.text("localization_warning").lower())
            self.assertEqual(rendered.text("source_coordinate_system"), "source-image-pixels")
            self.assertNotEqual(rendered.pixelColor(3, 2), source.pixelColor(3, 2))
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
