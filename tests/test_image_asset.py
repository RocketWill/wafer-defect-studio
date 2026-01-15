import hashlib
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from PySide6.QtGui import QImage

from wafer_defect_studio import image_asset, project


class ImageAssetTest(unittest.TestCase):
    def test_register_grayscale16_tiff_and_reject_rgb_without_mutation(self):
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "project"
            project.create_project(project_path)
            self._make_v1(project_path)

            database_path = project_path / "project.sqlite"
            database_before_open = database_path.read_bytes()
            self.assertEqual(project.open_project(project_path).schema_version, 1)
            self.assertEqual(database_path.read_bytes(), database_before_open)

            grayscale_path = workspace / "wafer.tiff"
            grayscale = QImage(5000, 4000, QImage.Format_Grayscale16)
            grayscale.fill(0x1234)
            self.assertTrue(grayscale.save(str(grayscale_path), "TIFF"))
            fingerprint = _sha256(grayscale_path)

            asset = image_asset.register_wafer_image(project_path, grayscale_path)

            self.assertTrue(asset.image_asset_id)
            self.assertEqual(asset.path, grayscale_path.resolve())
            self.assertEqual(asset.width, 5000)
            self.assertEqual(asset.height, 4000)
            self.assertEqual(asset.dtype, "uint16")
            self.assertEqual(asset.format, "TIFF")
            self.assertEqual(asset.fingerprint, fingerprint)
            self.assertEqual(_sha256(grayscale_path), fingerprint)
            self.assertEqual(project.open_project(project_path).schema_version, 2)

            connection = sqlite3.connect(database_path)
            try:
                row = connection.execute(
                    "SELECT image_asset_id, path, width, height, dtype, format, fingerprint "
                    "FROM image_assets"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(
                row,
                (
                    asset.image_asset_id,
                    str(grayscale_path.resolve()),
                    5000,
                    4000,
                    "uint16",
                    "TIFF",
                    fingerprint,
                ),
            )

            rgb_path = workspace / "color.tiff"
            rgb = QImage(8, 8, QImage.Format_RGB32)
            rgb.fill(0xFF102030)
            self.assertTrue(rgb.save(str(rgb_path), "TIFF"))
            database_before_rgb = database_path.read_bytes()
            with self.assertRaisesRegex(image_asset.ImageAssetError, "grayscale"):
                image_asset.register_wafer_image(project_path, rgb_path)
            self.assertEqual(database_path.read_bytes(), database_before_rgb)

    @staticmethod
    def _make_v1(project_path: Path) -> None:
        with sqlite3.connect(project_path / "project.sqlite") as connection:
            connection.execute("DROP TABLE image_assets")
            connection.execute("UPDATE project_metadata SET schema_version = 1")
            connection.execute("PRAGMA user_version = 1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
