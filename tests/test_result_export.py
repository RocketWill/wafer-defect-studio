import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
    ExportBundleResult,
    build_reviewed_rows,
    export_proposals_csv,
    export_proposals_json,
    export_proposals_png,
    export_result_bundle,
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

    def test_png_region_mask_uses_source_pixels_and_empty_mask_is_transparent(self):
        app = QApplication.instance() or QApplication([])
        source = QImage(64, 96, QImage.Format_Grayscale8)
        source.fill(80)
        region_mask = np.zeros((96, 64), dtype=bool)
        region_mask[60, 30] = True
        region_mask[61, 30] = True

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "regions.png"
            export_proposals_png(
                destination,
                source,
                self._reviewed_rows(),
                selected_class="刮痕",
                region_mask=region_mask,
                region_opacity=0.7,
            )
            rendered = QImage(str(destination))
            self.assertEqual((rendered.width(), rendered.height()), (64, 96))
            self.assertNotEqual(rendered.pixelColor(30, 60), source.pixelColor(30, 60))
            self.assertEqual(rendered.pixelColor(10, 60), source.pixelColor(10, 60))
            self.assertIn("threshold", rendered.text("legend").lower())
            self.assertIn("approximate", rendered.text("legend").lower())
            self.assertEqual(rendered.text("region_threshold"), "0.65")
            self.assertEqual(rendered.text("region_coordinate_system"), "source-image-pixels")

            empty_destination = Path(temporary) / "empty-regions.png"
            export_proposals_png(
                empty_destination,
                source,
                self._reviewed_rows(),
                selected_class="刮痕",
                region_mask=np.zeros((96, 64), dtype=bool),
                region_opacity=0.7,
            )
            empty = QImage(str(empty_destination))
            self.assertEqual(empty.pixelColor(10, 60), source.pixelColor(10, 60))
        app.processEvents()

    def test_bundle_publishes_three_unicode_destinations_atomically(self):
        app = QApplication.instance() or QApplication([])
        source = QImage(8, 6, QImage.Format_Grayscale8)
        source.fill(80)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destinations = (
                root / "輸出" / "結果.csv",
                root / "輸出" / "結果.json",
                root / "輸出" / "結果.png",
            )
            result = export_result_bundle(
                *destinations,
                source,
                self._reviewed_rows(),
                selected_class="刮痕",
            )
            self.assertIsInstance(result, ExportBundleResult)
            self.assertTrue(result.success)
            self.assertEqual(result.paths, destinations)
            self.assertIsNone(result.error)
            self.assertTrue(all(path.is_file() for path in destinations))
        app.processEvents()

    def test_bundle_refuses_existing_destinations_without_overwrite(self):
        app = QApplication.instance() or QApplication([])
        source = QImage(8, 6, QImage.Format_Grayscale8)
        source.fill(80)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destinations = (root / "result.csv", root / "result.json", root / "result.png")
            sentinel = ("CSV original", "JSON original", "PNG original")
            for path, content in zip(destinations, sentinel):
                path.write_text(content, encoding="utf-8")
            result = export_result_bundle(
                *destinations,
                source,
                self._reviewed_rows(),
                selected_class="刮痕",
            )
            self.assertFalse(result.success)
            self.assertEqual(result.paths, ())
            self.assertIsNotNone(result.error)
            self.assertEqual(
                [path.read_text(encoding="utf-8") for path in destinations],
                list(sentinel),
            )
        app.processEvents()

    def test_bundle_writer_failure_leaves_no_partial_outputs(self):
        app = QApplication.instance() or QApplication([])
        source = QImage(8, 6, QImage.Format_Grayscale8)
        source.fill(80)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destinations = (root / "result.csv", root / "result.json", root / "result.png")
            with patch(
                "wafer_defect_studio.result_export.export_proposals_json",
                side_effect=OSError("injected JSON failure"),
            ):
                result = export_result_bundle(
                    *destinations,
                    source,
                    self._reviewed_rows(),
                    selected_class="刮痕",
                )
            self.assertFalse(result.success)
            self.assertEqual(result.paths, ())
            self.assertIn("injected JSON failure", result.error or "")
            self.assertFalse(any(path.exists() for path in destinations))
            self.assertEqual(tuple(root.glob(".*")), ())
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
