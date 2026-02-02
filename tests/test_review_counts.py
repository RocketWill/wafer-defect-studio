import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wafer_defect_studio import project
from wafer_defect_studio.annotation import GridAnnotation, save_grid_annotation
from wafer_defect_studio.defect_class import DefectClass, save_defect_classes
from wafer_defect_studio.effective_area import (
    confirm_effective_wafer_area,
    set_effective_polygon,
)
from wafer_defect_studio.grid_profile import save_grid_profile
from wafer_defect_studio.image_grid_placement import set_image_grid_origin
from wafer_defect_studio.review import mark_image_reviewed
from wafer_defect_studio.review_counts import ReviewCounts, load_review_counts


class ReviewCountsTest(unittest.TestCase):
    def test_counts_split_participating_labels_review_state_and_excluded_grids(self):
        with TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project"
            image_asset_id = _seed_project(project_path)

            self.assertEqual(
                load_review_counts(project_path, image_asset_id),
                ReviewCounts(labeled=1, unreviewed=5, derived_normal=0, excluded=14),
            )

            mark_image_reviewed(project_path, image_asset_id)
            self.assertEqual(
                load_review_counts(project_path, image_asset_id),
                ReviewCounts(labeled=1, unreviewed=0, derived_normal=5, excluded=14),
            )


def _seed_project(project_path: Path) -> str:
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
    confirm_effective_wafer_area(project_path, image_asset_id)
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
