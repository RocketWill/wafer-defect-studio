import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QSettings, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QLabel,
    QPushButton,
    QSpinBox,
)

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.annotation import load_grid_annotation
from wafer_defect_studio.defect_class import DefectClass, save_defect_classes
from wafer_defect_studio.effective_area import (
    confirm_effective_wafer_area,
    load_effective_wafer_area,
    set_effective_ellipse,
)
from wafer_defect_studio.grid_profile import load_grid_profiles, save_grid_profile
from wafer_defect_studio.image_grid_placement import load_image_grid_placement, set_image_grid_origin
from wafer_defect_studio.main_window import MainWindow
from wafer_defect_studio.review import load_review_state
from wafer_defect_studio.review_counts import load_review_counts


class GuiValidationSmokeTest(unittest.TestCase):
    def test_file_actions_create_import_and_display_wafer_image(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
            settings.clear()
            settings.sync()
            project_path = root / "gui-project"
            project_path.mkdir()
            source_path = root / "wafer.png"
            _write_grayscale_png(source_path, width=5, height=4, value=80)
            source_before = source_path.read_bytes()

            window = MainWindow(settings=settings)
            window.show()
            app.processEvents()
            try:
                file_menu = next(
                    action.menu()
                    for action in window.menuBar().actions()
                    if action.text() == "File" and action.menu() is not None
                )
                create_action = next(
                    action
                    for action in file_menu.actions()
                    if action.objectName() == "createProjectAction"
                )
                import_action = next(
                    action
                    for action in file_menu.actions()
                    if action.objectName() == "importWaferImageAction"
                )
                self.assertFalse(import_action.isEnabled())

                with patch.object(
                    QFileDialog,
                    "getExistingDirectory",
                    return_value=str(project_path),
                ):
                    create_action.trigger()
                self.assertEqual(window.active_project_path, project_path.resolve())
                self.assertTrue(import_action.isEnabled())

                with patch.object(
                    QFileDialog,
                    "getOpenFileName",
                    return_value=(str(source_path), "PNG"),
                ):
                    import_action.trigger()

                deadline = time.monotonic() + 5
                while (
                    window.statusBar().currentMessage() != "Ready"
                    and time.monotonic() < deadline
                ):
                    app.processEvents()
                    QTest.qWait(10)
                self.assertEqual(window.statusBar().currentMessage(), "Ready")
                self.assertIsNotNone(window.current_image_asset)
                self.assertIsNotNone(window.loaded_wafer_image)
                self.assertEqual(window.current_image_asset.path, source_path.resolve())
                loaded = window.loaded_wafer_image
                self.assertEqual((loaded.width, loaded.height, loaded.dtype), (5, 4, "uint8"))
                self.assertEqual(source_path.read_bytes(), source_before)
            finally:
                deadline = time.monotonic() + 5
                while window._load_threads and time.monotonic() < deadline:
                    app.processEvents()
                    QTest.qWait(10)
                window.close()
                window.deleteLater()
                app.processEvents()

    def test_grid_profile_origin_and_area_controls_persist_through_visible_actions(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            project_path = root / "gui-project"
            project.create_project(project_path)
            source_path = root / "wafer.png"
            _write_grayscale_png(source_path, width=100, height=80, value=80)
            asset = image_asset.register_wafer_image(project_path, source_path)
            profile = save_grid_profile(project_path, 20, 20)
            set_image_grid_origin(
                project_path,
                asset.image_asset_id,
                profile.grid_profile_id,
                1,
                2,
            )
            area = set_effective_ellipse(
                project_path,
                asset.image_asset_id,
                50,
                40,
                30,
                20,
            )

            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
            settings.clear()
            settings.sync()
            window = MainWindow(settings=settings)
            window.resize(500, 400)
            window.show()
            app.processEvents()
            try:
                window.show_wafer_image(asset)
                window.set_grid_profile(project_path, profile)
                window.set_effective_wafer_area(area)

                width_spin = window.findChild(QSpinBox, "gridWidthSpinBox")
                height_spin = window.findChild(QSpinBox, "gridHeightSpinBox")
                apply_profile = window.findChild(QPushButton, "applyGridProfileButton")
                origin_x_spin = window.findChild(QSpinBox, "gridOriginXSpinBox")
                origin_y_spin = window.findChild(QSpinBox, "gridOriginYSpinBox")
                apply_origin = window.findChild(QPushButton, "applyGridOriginButton")
                confirm_area = window.findChild(QPushButton, "confirmEffectiveWaferAreaButton")
                confirmation_label = window.findChild(QLabel, "effectiveAreaConfirmationLabel")
                participating_label = window.findChild(QLabel, "participatingGridCountLabel")

                self.assertEqual((width_spin.value(), height_spin.value()), (20, 20))
                self.assertEqual((origin_x_spin.value(), origin_y_spin.value()), (1, 2))
                self.assertEqual(confirmation_label.text(), "Unconfirmed")
                self.assertTrue(confirm_area.isEnabled())
                self.assertEqual(participating_label.text(), "Participating: 6")

                width_spin.setValue(25)
                height_spin.setValue(16)
                self.assertTrue(apply_profile.isEnabled())
                apply_profile.click()
                app.processEvents()

                profiles = load_grid_profiles(project_path)
                self.assertEqual(len(profiles), 2)
                latest_profile = profiles[-1]
                self.assertEqual(
                    (latest_profile.version, latest_profile.cell_width, latest_profile.cell_height),
                    (2, 25, 16),
                )

                origin_x_spin.setValue(3)
                origin_y_spin.setValue(4)
                self.assertTrue(apply_origin.isEnabled())
                apply_origin.click()
                app.processEvents()

                placement = load_image_grid_placement(project_path, asset.image_asset_id)
                self.assertIsNotNone(placement)
                self.assertEqual(
                    (placement.grid_profile_id, placement.grid_profile_version,
                     placement.origin_x, placement.origin_y),
                    (latest_profile.grid_profile_id, latest_profile.version, 3, 4),
                )
                self.assertEqual((origin_x_spin.value(), origin_y_spin.value()), (3, 4))

                confirm_area.click()
                app.processEvents()
                persisted_area = load_effective_wafer_area(project_path, asset.image_asset_id)
                self.assertIsNotNone(persisted_area)
                self.assertTrue(persisted_area.confirmed)
                self.assertEqual(confirmation_label.text(), "Confirmed")
                self.assertFalse(confirm_area.isEnabled())
                self.assertEqual(participating_label.text(), "Participating: 4")
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()

    def test_annotation_and_review_controls_persist_visible_multilabel_workflow(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            project_path = root / "gui-project"
            project.create_project(project_path)
            source_path = root / "wafer.png"
            _write_grayscale_png(source_path, width=32, height=24, value=80)
            asset = image_asset.register_wafer_image(project_path, source_path)
            profile = save_grid_profile(project_path, 16, 12)
            set_image_grid_origin(
                project_path,
                asset.image_asset_id,
                profile.grid_profile_id,
                0,
                0,
            )
            set_effective_ellipse(
                project_path,
                asset.image_asset_id,
                16,
                12,
                16,
                12,
            )
            confirmed_area = confirm_effective_wafer_area(
                project_path,
                asset.image_asset_id,
            )
            save_defect_classes(
                project_path,
                (
                    DefectClass("scratch", "Scratch", "#cc4444", order=0),
                    DefectClass("stain", "Stain", "#4488cc", order=1),
                ),
            )

            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
            settings.clear()
            settings.sync()
            window = MainWindow(settings=settings)
            window.resize(900, 700)
            window.show()
            app.processEvents()
            try:
                window.show_wafer_image(asset)
                window.set_grid_profile(project_path, profile)
                window.set_effective_wafer_area(confirmed_area)
                app.processEvents()

                scratch_box = window.findChild(QCheckBox, "defectClass_scratchCheckBox")
                stain_box = window.findChild(QCheckBox, "defectClass_stainCheckBox")
                annotate_button = window.findChild(QPushButton, "annotateToolButton")
                if annotate_button is None:
                    annotate_button = window.findChild(QPushButton, "annotatetoolbutton")
                mark_reviewed_button = window.findChild(QPushButton, "markImageReviewedButton")
                labeled_label = window.findChild(QLabel, "labeledGridCountLabel")
                unreviewed_label = window.findChild(QLabel, "unreviewedGridCountLabel")
                derived_normal_label = window.findChild(QLabel, "derivedNormalGridCountLabel")
                excluded_label = window.findChild(QLabel, "excludedGridCountLabel")

                self.assertIsNotNone(scratch_box)
                self.assertIsNotNone(stain_box)
                self.assertIsNotNone(annotate_button)
                self.assertTrue(scratch_box.isVisible())
                self.assertTrue(stain_box.isVisible())
                self.assertTrue(annotate_button.isVisible())
                self.assertFalse(mark_reviewed_button.isEnabled())

                scratch_box.click()
                stain_box.click()
                self.assertTrue(scratch_box.isChecked())
                self.assertTrue(stain_box.isChecked())
                annotate_button.click()
                app.processEvents()

                view = window.centralWidget()
                target = view.mapFromScene(QPointF(8, 6))
                self.assertTrue(view.viewport().rect().contains(target))
                QTest.mouseClick(
                    view.viewport(),
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                    target,
                )
                app.processEvents()

                annotation = load_grid_annotation(
                    project_path,
                    asset.image_asset_id,
                    0,
                    0,
                )
                self.assertIsNotNone(annotation)
                self.assertEqual(annotation.class_codes, ("scratch", "stain"))
                self.assertTrue(mark_reviewed_button.isEnabled())
                self.assertEqual(labeled_label.text(), "Labeled: 1")
                self.assertEqual(unreviewed_label.text(), "Unreviewed: 3")
                self.assertEqual(derived_normal_label.text(), "Derived Normal: 0")
                self.assertEqual(excluded_label.text(), "Excluded: 0")

                mark_reviewed_button.click()
                app.processEvents()
                self.assertTrue(
                    load_review_state(project_path, asset.image_asset_id).reviewed
                )
                self.assertEqual(
                    load_review_counts(project_path, asset.image_asset_id).labeled,
                    1,
                )
                self.assertEqual(labeled_label.text(), "Labeled: 1")
                self.assertEqual(unreviewed_label.text(), "Unreviewed: 0")
                self.assertEqual(derived_normal_label.text(), "Derived Normal: 3")
                self.assertEqual(excluded_label.text(), "Excluded: 0")
                self.assertFalse(mark_reviewed_button.isEnabled())
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


def _write_grayscale_png(path: Path, *, width: int, height: int, value: int) -> None:
    image = QImage(width, height, QImage.Format_Grayscale8)
    image.fill(value)
    if not image.save(str(path), "PNG"):
        raise AssertionError(f"Unable to write grayscale source: {path}")
    del image


if __name__ == "__main__":
    unittest.main()
