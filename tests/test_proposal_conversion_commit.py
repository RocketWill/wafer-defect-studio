import json
import os
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QPushButton, QWidget

from wafer_defect_studio import project
from wafer_defect_studio.annotation import load_grid_annotation
from wafer_defect_studio.grid_geometry import AnnotationGrid
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.proposal_conversion import ConversionCell, ConversionPreview
from wafer_defect_studio.proposal_conversion_store import (
    ConversionStoreError,
    confirm_proposal_conversion,
    load_proposal_conversion,
)


class ProposalConversionCommitTest(unittest.TestCase):
    def _project(self, root: Path) -> Path:
        project_path = root / "project"
        project.create_project(project_path)
        connection = sqlite3.connect(project_path / "project.sqlite")
        try:
            for statement in (
                project._GRID_PROFILES_TABLE_SQL,
                project._IMAGE_GRID_PLACEMENTS_TABLE_SQL,
                project._EFFECTIVE_WAFER_AREAS_TABLE_SQL,
                project._DEFECT_CLASSES_TABLE_SQL,
                project._GRID_ANNOTATIONS_TABLE_SQL,
                project._IMAGE_REVIEWS_TABLE_SQL,
                project._DATA_GROUPS_TABLE_SQL,
                project._IMAGE_DATA_GROUPS_TABLE_SQL,
                project._TRAINING_SCOPE_TABLE_SQL,
                project._DATASET_SNAPSHOTS_TABLE_SQL,
                project._DATASET_SPLITS_TABLE_SQL,
                project._TRAINING_RUNS_TABLE_SQL,
                project._EVALUATION_RUNS_TABLE_SQL,
                project._EVALUATION_DECISIONS_TABLE_SQL,
                project._DETECTION_PROFILES_TABLE_SQL,
                project._DETECTION_RUNS_TABLE_SQL,
                project._DEFECT_PROPOSALS_TABLE_SQL,
                project._PROPOSAL_REVIEW_REVISIONS_TABLE_SQL,
            ):
                connection.execute(statement)
            connection.executemany(
                "INSERT INTO defect_classes "
                "(code, name, color, icon, description, display_order, enabled) "
                "VALUES (?, ?, ?, '', '', ?, 1)",
                (
                    ("scratch", "Scratch", "#cc4444", 0),
                    ("particle", "Particle", "#4488cc", 1),
                ),
            )
            connection.execute(
                "INSERT INTO image_assets "
                "(image_asset_id, path, width, height, dtype, format, fingerprint) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("image-1", "wafer.tif", 64, 64, "uint8", "TIFF", "fp-1"),
            )
            connection.execute(
                "INSERT INTO grid_annotations VALUES (?, ?, ?, ?)",
                ("image-1", 0, 0, json.dumps(["scratch"], separators=(",", ":"))),
            )
            connection.execute(
                "UPDATE project_metadata SET schema_version = 17 WHERE schema_version = 6"
            )
            connection.execute("PRAGMA user_version = 17")
            connection.commit()
        finally:
            connection.close()
        return project_path

    @staticmethod
    def _preview(class_codes=("particle",), proposal_ids=("proposal-1",)) -> ConversionPreview:
        grid = AnnotationGrid(0, 0, 0, 0, 8, 8)
        return ConversionPreview(
            (ConversionCell(grid, tuple(class_codes), tuple(proposal_ids)),),
            tuple(proposal_ids),
            {
                "source_coordinate_system": "source-image-pixels",
                "run_id": "run-1",
            },
        )

    def test_unconfirmed_conversion_is_a_zero_write_cancel_path(self):
        with TemporaryDirectory() as temporary_directory:
            project_path = self._project(Path(temporary_directory))
            preview = self._preview()
            with self.assertRaises(ConversionStoreError):
                confirm_proposal_conversion(
                    project_path,
                    preview,
                    image_asset_id="image-1",
                    confirmed=False,
                    conversion_id="cancelled",
                )
            self.assertEqual(project.open_project(project_path).schema_version, 17)
            self.assertEqual(
                load_grid_annotation(project_path, "image-1", 0, 0).class_codes,
                ("scratch",),
            )
            connection = sqlite3.connect(project_path / "project.sqlite")
            try:
                self.assertIsNone(
                    connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' "
                        "AND name = 'proposal_conversions'"
                    ).fetchone()
                )
            finally:
                connection.close()

    def test_confirm_merges_without_removing_labels_and_round_trips_provenance(self):
        with TemporaryDirectory() as temporary_directory:
            project_path = self._project(Path(temporary_directory))
            record = confirm_proposal_conversion(
                project_path,
                self._preview(),
                image_asset_id="image-1",
                confirmed=True,
                conversion_id="conversion-1",
                actor="engineer",
                converted_at="2026-08-22T12:00:00+00:00",
            )
            self.assertEqual(record.conversion_id, "conversion-1")
            self.assertEqual(record.image_asset_id, "image-1")
            self.assertEqual(record.affected_cells[0].class_codes, ("particle",))
            self.assertEqual(record.proposal_ids, ("proposal-1",))
            self.assertEqual(record.provenance["source_coordinate_system"], "source-image-pixels")
            self.assertEqual(record.actor, "engineer")
            self.assertEqual(record.converted_at, "2026-08-22T12:00:00+00:00")
            self.assertEqual(
                load_grid_annotation(project_path, "image-1", 0, 0).class_codes,
                ("scratch", "particle"),
            )
            self.assertEqual(load_proposal_conversion(project_path, "conversion-1"), record)
            with self.assertRaises(ConversionStoreError):
                confirm_proposal_conversion(
                    project_path,
                    self._preview(),
                    image_asset_id="image-1",
                    confirmed=True,
                    conversion_id="conversion-1",
                )

            connection = sqlite3.connect(project_path / "project.sqlite")
            try:
                with self.assertRaises(sqlite3.DatabaseError):
                    connection.execute(
                        "UPDATE proposal_conversions SET actor = 'other' "
                        "WHERE conversion_id = 'conversion-1'"
                    )
                with self.assertRaises(sqlite3.DatabaseError):
                    connection.execute(
                        "DELETE FROM proposal_conversions WHERE conversion_id = 'conversion-1'"
                    )
            finally:
                connection.close()

    def test_invalid_class_rolls_back_annotations_and_conversion_record(self):
        with TemporaryDirectory() as temporary_directory:
            project_path = self._project(Path(temporary_directory))
            with self.assertRaises(ConversionStoreError):
                confirm_proposal_conversion(
                    project_path,
                    self._preview(("unknown",)),
                    image_asset_id="image-1",
                    confirmed=True,
                    conversion_id="invalid-class",
                )
            self.assertEqual(
                load_grid_annotation(project_path, "image-1", 0, 0).class_codes,
                ("scratch",),
            )
            connection = sqlite3.connect(project_path / "project.sqlite")
            try:
                self.assertIsNone(
                    connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' "
                        "AND name = 'proposal_conversions'"
                    ).fetchone()
                )
            finally:
                connection.close()

    def test_confirmation_controls_gate_callback_and_show_preview(self):
        app = QApplication.instance() or QApplication([])
        preview = self._preview()
        calls = []
        window = MainWindow()
        window.configure_proposal_conversion(preview, lambda value: calls.append(value))
        window.show()
        app.processEvents()
        controls = window.findChild(QWidget, "proposalConversionControls")
        self.assertIsNotNone(controls)
        preview_label = window.findChild(QLabel, "proposalConversionPreviewLabel")
        self.assertIn("(0, 0)", preview_label.text())
        self.assertIn("particle", preview_label.text())
        self.assertIn("proposal-1", preview_label.text())
        checkbox = window.findChild(QCheckBox, "proposalConversionConfirmCheckBox")
        button = window.findChild(QPushButton, "convertToGridAnnotationsButton")
        self.assertFalse(button.isEnabled())
        checkbox.click()
        self.assertTrue(button.isEnabled())
        button.click()
        self.assertEqual(calls, [preview])
        self.assertIn("Converted", window.findChild(QLabel, "proposalConversionStatusLabel").text())
        window.close()
if __name__ == "__main__":
    unittest.main()
