import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.grid_profile import GridProfile
from wafer_defect_studio.main_window import MainWindow


class GridOverlayTest(unittest.TestCase):
    def test_rendered_lod_lines_replace_and_survive_image_rebuild(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "project"
            project.create_project(project_path)
            source_path = workspace / "wafer.png"
            _write_fixture(source_path)
            asset = image_asset.register_wafer_image(project_path, source_path)

            window = MainWindow()
            window.resize(500, 400)
            window.show()
            app.processEvents()
            window.show_wafer_image(asset)
            app.processEvents()
            view = window.centralWidget()
            anchor = view.viewport().rect().center()
            source_before = view.source_pixel_at(anchor)

            profile = GridProfile("profile", 1, 10, 8)
            view.set_annotation_grid(profile, QPoint(3, 2))
            app.processEvents()
            self.assertEqual(view.source_pixel_at(anchor), source_before)
            self.assertEqual(len(view.scene().items()), 2)

            view.resetTransform()
            view.scale(0.25, 0.25)
            far = _render(view)
            self.assertGreater(_magenta_pixels(far), 0)

            view.resetTransform()
            view.scale(1.0, 1.0)
            mid = _render(view)
            self.assertEqual(
                _vertical_hits(view, mid, (3, 13, 23, 33, 43, 53, 63, 73, 83, 93), 40),
                (3, 43, 83),
            )
            self.assertEqual(
                _horizontal_hits(view, mid, (2, 10, 18, 26, 34, 42, 50, 58, 66, 74), 50),
                (2, 34, 66),
            )

            view.resetTransform()
            view.scale(3.0, 3.0)
            near = _render(view)
            self.assertGreater(_magenta_pixels(near), _magenta_pixels(mid))
            self.assertEqual(
                _vertical_hits(view, near, (3, 13, 23, 33, 43, 53, 63, 73, 83, 93), 40),
                (3, 13, 23, 33, 43, 53, 63, 73, 83, 93),
            )

            view.set_annotation_grid(GridProfile("replacement", 1, 20, 16), QPoint(0, 0))
            self.assertEqual(len(view.scene().items()), 2)
            window.show_wafer_image(asset)
            app.processEvents()
            self.assertEqual(len(view.scene().items()), 2)

            window.close()
            del view, window
            app.processEvents()


def _write_fixture(path: Path) -> None:
    image = QImage(100, 80, QImage.Format_Grayscale8)
    image.fill(80)
    if not image.save(str(path), "PNG"):
        raise AssertionError(f"Unable to write fixture: {path}")
    del image


def _render(view) -> QImage:
    image = QImage(view.viewport().size(), QImage.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    view.render(painter)
    painter.end()
    return image


def _magenta_pixels(image: QImage) -> int:
    magenta = (255, 0, 255, 255)
    return sum(
        image.pixelColor(x, y).getRgb() == magenta
        for y in range(image.height())
        for x in range(image.width())
    )


def _vertical_hits(view, image: QImage, source_xs: tuple[int, ...], source_y: int) -> tuple[int, ...]:
    hits = []
    for source_x in source_xs:
        point = view.mapFromScene(QPointF(source_x, source_y))
        if any(
            0 <= point.x() + dx < image.width()
            and 0 <= point.y() + dy < image.height()
            and image.pixelColor(point.x() + dx, point.y() + dy).getRgb() == (255, 0, 255, 255)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        ):
            hits.append(source_x)
    return tuple(hits)


def _horizontal_hits(view, image: QImage, source_ys: tuple[int, ...], source_x: int) -> tuple[int, ...]:
    hits = []
    for source_y in source_ys:
        point = view.mapFromScene(QPointF(source_x, source_y))
        if any(
            0 <= point.x() + dx < image.width()
            and 0 <= point.y() + dy < image.height()
            and image.pixelColor(point.x() + dx, point.y() + dy).getRgb() == (255, 0, 255, 255)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        ):
            hits.append(source_y)
    return tuple(hits)


if __name__ == "__main__":
    unittest.main()
