"""Run the repeatable WaferView interaction feedback benchmark."""

from __future__ import annotations

import argparse
import os
import sys
from array import array
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from wafer_defect_studio.interaction_benchmark import measure_interactions
from wafer_defect_studio.wafer_view import LoadedWaferImage, WaferView


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--target-ms", type=float, default=100.0)
    options = parser.parse_args(argv)

    app = QApplication.instance() or QApplication([])
    view = WaferView()
    view.resize(320, 240)
    loaded = LoadedWaferImage(64, 48, "uint8", array("B", [80] * (64 * 48)))
    image = QImage(64, 48, QImage.Format_Grayscale8)
    image.fill(80)
    view._set_loaded_image(loaded, image)
    view.show()
    app.processEvents()
    report = measure_interactions(view, samples=options.samples, target_ms=options.target_ms)
    print(report.to_markdown(), end="")
    view.close()
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
