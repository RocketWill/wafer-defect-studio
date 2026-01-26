import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import time
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QEventLoop
from PySide6.QtGui import QImage, qRgb
from PySide6.QtWidgets import QApplication

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.main_window import MainWindow


class JpegWarningTest(unittest.TestCase):
    def test_migrate_v2_persist_lossy_warning_and_reject_color_without_row(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "project"
            project.create_project(project_path)
            _make_v2(project_path)
            self.assertEqual(project.open_project(project_path).schema_version, 2)

            grayscale_path = workspace / "wafer.jpg"
            _write_grayscale_jpeg(grayscale_path)
            source_before = grayscale_path.read_bytes()

            asset = image_asset.register_wafer_image(project_path, grayscale_path)

            self.assertEqual((asset.width, asset.height), (3, 2))
            self.assertEqual((asset.dtype, asset.format, asset.lossy_source), ("uint8", "JPEG", True))
            self.assertEqual(grayscale_path.read_bytes(), source_before)
            self.assertEqual(project.open_project(project_path).schema_version, 3)
            connection = sqlite3.connect(project_path / "project.sqlite")
            try:
                row = connection.execute(
                    "SELECT path, width, height, dtype, format, lossy_source "
                    "FROM image_assets WHERE image_asset_id = ?",
                    (asset.image_asset_id,),
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(row, (str(asset.path), 3, 2, "uint8", "JPEG", 1))

            reopened = image_asset.load_image_assets(project_path)
            self.assertEqual(len(reopened), 1)
            self.assertEqual(reopened[0].asset, asset)
            self.assertTrue(reopened[0].asset.lossy_source)

            window = MainWindow()
            window.resize(320, 240)
            window.show()
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)
            try:
                window.load_wafer_image(reopened[0])
                self.assertEqual(window.statusBar().currentMessage(), "Loading - Lossy JPEG Source")
                deadline = time.monotonic() + 5
                while window.statusBar().currentMessage() != "Ready - Lossy JPEG Source" and time.monotonic() < deadline:
                    app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)
                self.assertEqual(window.statusBar().currentMessage(), "Ready - Lossy JPEG Source")
                self.assertEqual((window._loaded_wafer_image.width, window._loaded_wafer_image.height), (3, 2))
                self.assertEqual(window._loaded_wafer_image.dtype, "uint8")
                while window._load_threads and time.monotonic() < deadline:
                    app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)
            finally:
                window.close()
                del window
                app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)

            reject_project = workspace / "reject-project"
            project.create_project(reject_project)
            _make_v2(reject_project)
            color_path = workspace / "color.jpg"
            _write_color_jpeg(color_path)
            with self.assertRaisesRegex(
                image_asset.ImageAssetError,
                r"^Color images are not supported; provide a grayscale TIFF, PNG, BMP, or JPEG\.$",
            ):
                image_asset.register_wafer_image(reject_project, color_path)
            connection = sqlite3.connect(reject_project / "project.sqlite")
            try:
                row_count = connection.execute("SELECT COUNT(*) FROM image_assets").fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(row_count, 0)


def _make_v2(project_path: Path) -> None:
    connection = sqlite3.connect(project_path / "project.sqlite")
    try:
        connection.execute("DROP TABLE image_assets")
        connection.execute(
            "CREATE TABLE image_assets ("
            "image_asset_id TEXT NOT NULL PRIMARY KEY, "
            "path TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, "
            "dtype TEXT NOT NULL, format TEXT NOT NULL, fingerprint TEXT NOT NULL"
            ")"
        )
        connection.execute("UPDATE project_metadata SET schema_version = 2")
        connection.execute("PRAGMA user_version = 2")
        connection.commit()
    finally:
        connection.close()


def _write_grayscale_jpeg(path: Path) -> None:
    image = QImage(3, 2, QImage.Format_Grayscale8)
    image.fill(120)
    if not image.save(str(path), "JPEG"):
        raise AssertionError(f"Unable to write grayscale JPEG: {path}")
    del image


def _write_color_jpeg(path: Path) -> None:
    image = QImage(3, 2, QImage.Format_RGB32)
    image.fill(qRgb(42, 42, 42))
    if not image.save(str(path), "JPEG"):
        raise AssertionError(f"Unable to write color JPEG: {path}")
    del image


if __name__ == "__main__":
    unittest.main()
