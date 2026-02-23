import csv
import tempfile
import unittest
from pathlib import Path

from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.proposal_generation import DefectProposal
from wafer_defect_studio.proposal_review import ProposalReviewRevision
from wafer_defect_studio.result_export import (
    APPROXIMATE_LOCALIZATION_WARNING,
    CSV_COLUMNS,
    build_reviewed_rows,
    export_proposals_csv,
)


class ResultExportTest(unittest.TestCase):
    def test_csv_uses_latest_review_geometry_and_round_trips_unicode(self):
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

        rows = build_reviewed_rows(
            (proposal,),
            (revision,),
            project_id="專案",
            image_asset_id="晶圓/α.png",
            run_id="run-immutable",
            profile_id="profile-immutable",
            thresholds={"刮痕": 0.65},
        )

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


if __name__ == "__main__":
    unittest.main()
