import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtGui import QImage

from wafer_defect_studio import image_asset, project


class SourceHealthTest(unittest.TestCase):
    def test_reopen_available_missing_changed_without_database_mutation(self):
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "project"
            project.create_project(project_path)
            source_path = workspace / "wafer.tiff"
            _write_source(source_path, 0x1234)
            registered = image_asset.register_wafer_image(project_path, source_path)
            database_path = project_path / "project.sqlite"
            database_before = database_path.read_bytes()

            reopened = image_asset.load_image_assets(project_path)
            self.assertEqual(len(reopened), 1)
            self.assertEqual(reopened[0].asset, registered)
            self.assertEqual(reopened[0].source_health, image_asset.SourceHealth.AVAILABLE)
            self.assertEqual(database_path.read_bytes(), database_before)

            source_path.unlink()
            missing = image_asset.load_image_assets(project_path)
            self.assertEqual(missing[0].asset, registered)
            self.assertEqual(missing[0].source_health, image_asset.SourceHealth.MISSING)
            self.assertEqual(database_path.read_bytes(), database_before)

            source_path.write_bytes(b"recreated with different bytes")
            changed = image_asset.load_image_assets(project_path)
            self.assertEqual(changed[0].asset, registered)
            self.assertEqual(changed[0].source_health, image_asset.SourceHealth.CHANGED)
            self.assertEqual(database_path.read_bytes(), database_before)

            legacy_path = workspace / "legacy"
            project.create_project(legacy_path)
            _make_v1(legacy_path)
            self.assertEqual(image_asset.load_image_assets(legacy_path), ())


def _write_source(path: Path, value: int) -> None:
    image = QImage(5000, 4000, QImage.Format_Grayscale16)
    image.fill(value)
    if not image.save(str(path), "TIFF"):
        raise AssertionError(f"Unable to write source: {path}")
    del image


def _make_v1(project_path: Path) -> None:
    connection = sqlite3.connect(project_path / "project.sqlite")
    try:
        connection.execute("DROP TABLE image_assets")
        connection.execute("UPDATE project_metadata SET schema_version = 1")
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    finally:
        connection.close()


if __name__ == "__main__":
    unittest.main()
