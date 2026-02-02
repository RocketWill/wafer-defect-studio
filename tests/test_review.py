import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wafer_defect_studio import project
from wafer_defect_studio.annotation import (
    GridAnnotation,
    load_grid_annotation,
    save_grid_annotation,
)
from wafer_defect_studio.defect_class import (
    DefectClass,
    load_defect_classes,
    save_defect_classes,
)
from wafer_defect_studio.effective_area import set_effective_polygon
from wafer_defect_studio.grid_profile import save_grid_profile
from wafer_defect_studio.image_grid_placement import set_image_grid_origin
from wafer_defect_studio.review import (
    ReviewError,
    derived_normal_grids,
    load_review_state,
    mark_image_reviewed,
    reopen_image,
)


class ReviewTest(unittest.TestCase):
    def test_review_reopen_derives_only_in_area_and_preserves_annotations(self):
        with TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project"
            image_asset_id = _seed_project(project_path, confirmed=True)

            self.assertEqual(load_review_state(project_path, image_asset_id).reviewed, False)
            reviewed = mark_image_reviewed(project_path, image_asset_id)
            self.assertTrue(reviewed.reviewed)
            self.assertEqual(project.open_project(project_path).schema_version, 9)

            normal = derived_normal_grids(project_path, image_asset_id)
            self.assertEqual(
                {(grid.row, grid.column) for grid in normal},
                {(1, 2), (1, 3), (2, 1), (2, 2), (2, 3)},
            )
            self.assertNotIn((-1, -1), {(grid.row, grid.column) for grid in normal})

            classes = load_defect_classes(project_path)
            save_defect_classes(project_path, classes)
            save_grid_annotation(
                project_path,
                GridAnnotation(image_asset_id, 1, 1, ("scratch",)),
            )
            self.assertEqual(
                load_grid_annotation(project_path, image_asset_id, 1, 1).class_codes,
                ("scratch",),
            )

            reopened = reopen_image(project_path, image_asset_id)
            self.assertFalse(reopened.reviewed)
            self.assertEqual(derived_normal_grids(project_path, image_asset_id), ())
            self.assertEqual(
                load_grid_annotation(project_path, image_asset_id, 1, 1).class_codes,
                ("scratch",),
            )

    def test_unconfirmed_area_blocks_first_review_without_migration(self):
        with TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project"
            image_asset_id = _seed_project(project_path, confirmed=False)
            database_path = project_path / "project.sqlite"
            before = database_path.read_bytes()

            with self.assertRaises(ReviewError):
                mark_image_reviewed(project_path, image_asset_id)

            self.assertEqual(project.open_project(project_path).schema_version, 8)
            self.assertEqual(database_path.read_bytes(), before)
            connection = sqlite3.connect(database_path)
            try:
                self.assertEqual(
                    connection.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type = 'table' AND name = 'image_reviews'"
                    ).fetchone(),
                    None,
                )
            finally:
                connection.close()


def _seed_project(project_path: Path, *, confirmed: bool) -> str:
    project.create_project(project_path)
    database_path = project_path / "project.sqlite"
    image_asset_id = "image-1"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "INSERT INTO image_assets "
            "(image_asset_id, path, width, height, dtype, format, fingerprint) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (image_asset_id, "source.tif", 100, 80, "uint8", "TIFF", "fingerprint"),
        )
        connection.commit()
    finally:
        connection.close()

    profile = save_grid_profile(project_path, 20, 20)
    set_image_grid_origin(project_path, image_asset_id, profile.grid_profile_id, 0, 0)
    set_effective_polygon(
        project_path,
        image_asset_id,
        [(20, 20), (80, 20), (80, 60), (20, 60)],
    )

    connection = sqlite3.connect(database_path)
    try:
        if confirmed:
            connection.execute(
                "UPDATE effective_wafer_areas SET confirmed = 1 WHERE image_asset_id = ?",
                (image_asset_id,),
            )
        connection.commit()
    finally:
        connection.close()

    save_defect_classes(
        project_path,
        (DefectClass("scratch", "Scratch", "#cc4444", order=0),),
    )
    save_grid_annotation(
        project_path,
        GridAnnotation(image_asset_id, 1, 1, ("scratch",)),
    )
    save_grid_annotation(
        project_path,
        GridAnnotation(image_asset_id, -1, -1, ("scratch",)),
    )
    return image_asset_id


if __name__ == "__main__":
    unittest.main()
