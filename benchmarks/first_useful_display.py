"""Measure first useful display time for the 20 MP happy path."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import dataclass
from datetime import datetime
import platform
from pathlib import Path
import sys
import tempfile
import time

import PySide6
from PySide6.QtCore import QEventLoop, qVersion
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QGraphicsPixmapItem

from wafer_defect_studio import image_asset, project
from wafer_defect_studio.main_window import MainWindow


WIDTH = 5000
HEIGHT = 4000
PIXEL_VALUE = 0x1234
TARGET_SECONDS = 3.0
TIMEOUT_SECONDS = 30.0
REPORT_PATH = Path(__file__).resolve().parents[1] / "docs" / "benchmarks" / "01-6c-first-useful-display.md"


@dataclass(frozen=True)
class Measurement:
    measured_at: str
    elapsed_seconds: float | None
    outcome: str
    status: str
    loaded_dimensions: str
    loaded_dtype: str
    pixmap_ready: bool
    threads_cleaned: bool
    error: str | None


def _write_fixture(path: Path) -> None:
    image = QImage(WIDTH, HEIGHT, QImage.Format_Grayscale16)
    image.fill(PIXEL_VALUE)
    if not image.save(str(path), "TIFF"):
        raise RuntimeError(f"Unable to write deterministic TIFF fixture: {path}")
    del image


def _pump_until(app: QApplication, predicate, deadline: float) -> bool:
    while time.monotonic() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)
        if predicate():
            return True
        time.sleep(0.001)
    app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)
    return bool(predicate())


def _pixmap_ready(window: MainWindow) -> bool:
    view = window.centralWidget()
    if view is None or view.scene() is None:
        return False
    return any(
        isinstance(item, QGraphicsPixmapItem) and not item.pixmap().isNull()
        for item in view.scene().items()
    )


def _display_ready(window: MainWindow) -> bool:
    loaded = window._loaded_wafer_image
    return (
        window.statusBar().currentMessage() == "Ready"
        and loaded is not None
        and loaded.width == WIDTH
        and loaded.height == HEIGHT
        and loaded.dtype == "uint16"
        and len(loaded.pixels) == WIDTH * HEIGHT
        and _pixmap_ready(window)
    )


def _measure() -> Measurement:
    app = QApplication.instance() or QApplication([])
    window: MainWindow | None = None
    measured_at = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        with tempfile.TemporaryDirectory(prefix="wafer-first-display-") as temporary_directory:
            workspace = Path(temporary_directory)
            project_path = workspace / "project"
            source_path = workspace / "wafer.tiff"

            _write_fixture(source_path)
            project.create_project(project_path)
            image_asset.register_wafer_image(project_path, source_path)
            reopened = image_asset.load_image_assets(project_path)
            if len(reopened) != 1:
                raise RuntimeError("Expected one reopened image asset")

            window = MainWindow()
            window.resize(1000, 700)
            window.show()
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)

            started = time.perf_counter()
            window.load_wafer_image(reopened[0])
            deadline = time.monotonic() + TIMEOUT_SECONDS
            ready = _pump_until(app, lambda: _display_ready(window), deadline)
            elapsed = time.perf_counter() - started
            if not ready:
                raise RuntimeError(
                    "Timed out waiting for Ready, native 5000x4000 uint16 pixels, and a pixmap"
                )

            threads_cleaned = _pump_until(app, lambda: not window._load_threads, deadline)
            if not threads_cleaned:
                raise RuntimeError("Timed out waiting for decode thread cleanup")

            loaded = window._loaded_wafer_image
            return Measurement(
                measured_at=measured_at,
                elapsed_seconds=elapsed,
                outcome="MET" if elapsed <= TARGET_SECONDS else "MISSED",
                status=window.statusBar().currentMessage(),
                loaded_dimensions=f"{loaded.width}x{loaded.height}" if loaded else "unknown",
                loaded_dtype=loaded.dtype if loaded else "unknown",
                pixmap_ready=_pixmap_ready(window),
                threads_cleaned=threads_cleaned,
                error=None,
            )
    except Exception as error:
        return Measurement(
            measured_at=measured_at,
            elapsed_seconds=None,
            outcome="MISSED",
            status=window.statusBar().currentMessage() if window is not None else "not started",
            loaded_dimensions="unknown",
            loaded_dtype="unknown",
            pixmap_ready=False,
            threads_cleaned=not window or not window._load_threads,
            error=str(error),
        )
    finally:
        if window is not None:
            window.close()
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)
            window.deleteLater()
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)


def _environment_lines() -> list[str]:
    return [
        f"- Python: `{sys.version.split()[0]}` ({platform.python_implementation()})",
        f"- PySide6: `{getattr(PySide6, '__version__', 'unknown')}`; Qt: `{qVersion()}`",
        f"- Platform: `{platform.platform()}`; machine: `{platform.machine() or 'unknown'}`",
        f"- `QT_QPA_PLATFORM`: `{os.environ.get('QT_QPA_PLATFORM', 'unset')}`",
    ]


def _render_report(measurement: Measurement) -> str:
    elapsed = (
        f"{measurement.elapsed_seconds:.3f} s" if measurement.elapsed_seconds is not None else "unavailable"
    )
    error = f"\n- Error: `{measurement.error}`" if measurement.error else ""
    return "\n".join(
        [
            "# Slice 01.6c — 20 MP first-useful-display measurement",
            "",
            f"Measured at: `{measurement.measured_at}`",
            "",
            "## Local environment",
            "",
            *_environment_lines(),
            "",
            "## Measurement",
            "",
            f"- Fixture: deterministic `5000 × 4000` `Grayscale16` TIFF, filled with `0x{PIXEL_VALUE:04x}`.",
            "- Setup (project creation, TIFF write, registration, and reopen) completed before timing.",
            "- Timed interval: immediately before `MainWindow.load_wafer_image(...)` through source-health hashing, background decode, native array copy, and the first `Ready` state with a non-null scene pixmap.",
            f"- First useful display: `{elapsed}`.",
            f"- Target: `≤ {TARGET_SECONDS:.3f} s`; result: **{measurement.outcome}**.",
            f"- Final status: `{measurement.status}`; native payload: `{measurement.loaded_dimensions}` `{measurement.loaded_dtype}`; pixmap ready: `{measurement.pixmap_ready}`; decode threads cleaned: `{measurement.threads_cleaned}`.{error}",
            "",
            "## Limitations",
            "",
            "- This is one local offscreen run on one deterministic fixture; it is not a cold-start, multi-run distribution, or production-display benchmark.",
            "- The result includes the current source-health SHA-256 pass and native copy, but excludes project/TIFF setup by design.",
            "- No optimization, caching, image pyramid, or benchmark framework was added for this measurement.",
            "",
        ]
    )


def main() -> int:
    measurement = _measure()
    report = _render_report(measurement)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Environment: Python {sys.version.split()[0]} ({platform.python_implementation()}), "
          f"PySide6 {getattr(PySide6, '__version__', 'unknown')}, Qt {qVersion()}, "
          f"{platform.platform()}, QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM', 'unset')}")
    elapsed = (
        f"{measurement.elapsed_seconds:.3f}s" if measurement.elapsed_seconds is not None else "unavailable"
    )
    print(f"First useful display: {elapsed}; target <= {TARGET_SECONDS:.3f}s; {measurement.outcome}")
    print(
        f"Ready={measurement.status == 'Ready'} dimensions={measurement.loaded_dimensions} "
        f"dtype={measurement.loaded_dtype} pixmap={measurement.pixmap_ready} "
        f"threads_cleaned={measurement.threads_cleaned}"
    )
    if measurement.error:
        print(f"Error: {measurement.error}")
    print(f"Report: {REPORT_PATH}")
    return 0 if measurement.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
