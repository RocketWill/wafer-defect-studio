"""Repeatable native WaferView interaction feedback measurements."""

from __future__ import annotations

import platform
import statistics
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from PySide6.QtCore import QPoint


@dataclass(frozen=True, slots=True)
class InteractionMeasurement:
    """One measured interaction latency and its target result."""

    name: str
    latency_ms: float
    target_ms: float
    passed: bool

    @property
    def result(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass(frozen=True, slots=True)
class InteractionBenchmarkReport:
    """Immutable measurements with the environment used to obtain them."""

    environment: Mapping[str, str]
    target_ms: float
    measurements: tuple[InteractionMeasurement, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))
        object.__setattr__(self, "measurements", tuple(self.measurements))

    @property
    def passed(self) -> bool:
        return all(measurement.passed for measurement in self.measurements)

    @property
    def all_passed(self) -> bool:
        return self.passed

    def to_markdown(self) -> str:
        lines = [
            "# Interaction Feedback Benchmark",
            "",
            f"Target: <= {self.target_ms:.1f} ms",
            "",
            "## Environment",
        ]
        for key in sorted(self.environment):
            lines.append(f"- `{key}`: {self.environment[key]}")
        lines.extend(("", "## Measurements", "| Interaction | Latency (ms) | Target (ms) | Result |", "| --- | ---: | ---: | --- |"))
        for measurement in self.measurements:
            lines.append(
                f"| {measurement.name} | {measurement.latency_ms:.3f} | "
                f"{measurement.target_ms:.1f} | {measurement.result} |"
            )
        return "\n".join(lines) + "\n"

    @property
    def markdown(self) -> str:
        return self.to_markdown()


def measure_interactions(
    view: object,
    *,
    samples: int = 3,
    target_ms: float = 100.0,
    position: QPoint | None = None,
) -> InteractionBenchmarkReport:
    """Measure pan, zoom, source lookup, and selection on an existing view.

    Setup and warm-up calls happen before timing.  The operations only alter
    view transforms/selection and query source pixels; they never write source
    image data.
    """

    _validate_options(samples, target_ms)
    _require_view(view)
    point = position or view.viewport().rect().center()
    operations: tuple[tuple[str, Callable[[], None]], ...] = (
        ("pan", lambda: _pan(view)),
        ("zoom", lambda: _zoom(view)),
        ("hover/source lookup", lambda: view.source_pixel_at(point)),
        ("selection", lambda: view.set_selected_class_codes(("benchmark",))),
    )
    measurements: list[InteractionMeasurement] = []
    for name, operation in operations:
        operation()  # warm-up/setup is excluded from recorded latency
        elapsed = []
        for _ in range(samples):
            started = time.perf_counter_ns()
            operation()
            elapsed.append((time.perf_counter_ns() - started) / 1_000_000.0)
        latency = float(statistics.median(elapsed))
        measurements.append(InteractionMeasurement(name, latency, target_ms, latency <= target_ms))
    return InteractionBenchmarkReport(_environment(), target_ms, tuple(measurements))


def render_markdown(report: InteractionBenchmarkReport) -> str:
    if not isinstance(report, InteractionBenchmarkReport):
        raise TypeError("report must be an InteractionBenchmarkReport")
    return report.to_markdown()


def _pan(view: object) -> None:
    horizontal = view.horizontalScrollBar()
    vertical = view.verticalScrollBar()
    horizontal_value = horizontal.value()
    vertical_value = vertical.value()
    horizontal.setValue(horizontal_value + 1)
    vertical.setValue(vertical_value + 1)
    horizontal.setValue(horizontal_value)
    vertical.setValue(vertical_value)


def _zoom(view: object) -> None:
    transform = view.transform()
    view.scale(1.01, 1.01)
    view.setTransform(transform)


def _require_view(view: object) -> None:
    required = (
        "viewport",
        "horizontalScrollBar",
        "verticalScrollBar",
        "transform",
        "scale",
        "setTransform",
        "source_pixel_at",
        "set_selected_class_codes",
    )
    missing = [name for name in required if not callable(getattr(view, name, None))]
    if missing:
        raise TypeError(f"view does not expose WaferView benchmark seam: {', '.join(missing)}")


def _validate_options(samples: int, target_ms: float) -> None:
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if isinstance(target_ms, bool) or not isinstance(target_ms, (int, float)) or target_ms <= 0:
        raise ValueError("target_ms must be a positive number")


def _environment() -> dict[str, str]:
    try:
        import PySide6

        qt_version = str(PySide6.__version__)
    except Exception:  # pragma: no cover - optional metadata only
        qt_version = "unknown"
    return {
        "platform": platform.platform(),
        "python": platform.python_version() or sys.version.split()[0],
        "qt": qt_version,
    }


InteractionBenchmark = InteractionBenchmarkReport
InteractionResult = InteractionMeasurement


__all__ = [
    "InteractionBenchmark",
    "InteractionBenchmarkReport",
    "InteractionMeasurement",
    "InteractionResult",
    "measure_interactions",
    "render_markdown",
]
