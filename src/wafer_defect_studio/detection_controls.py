"""Small, value-only Widgets for reviewing one CAM Detection result.

The controls deliberately keep persistence and worker orchestration outside the
GUI.  A :class:`CamDetectionArtifact` is rendered as a source-coordinate map
summary; an injected launcher supplies the non-blocking Detection worker.
"""

from __future__ import annotations

import json
import queue
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .cam_detection import CamDetectionArtifact
from .detection_worker import (
    DetectionProgress,
    DetectionRequest,
    DetectionTerminal,
    decode_message,
    start_detection_worker,
)


DetectionRequestSource = DetectionRequest | Callable[[], DetectionRequest]
DetectionLauncher = Callable[[DetectionRequest], Any]
DetectionTerminalCallback = Callable[[DetectionTerminal], Any]


class DetectionControls(QWidget):
    """Expose Detection layers and a truthful worker state without SQLite IO."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._artifact: CamDetectionArtifact | None = None
        self._request_source: DetectionRequestSource | None = None
        self._launcher: DetectionLauncher = start_detection_worker
        self._terminal_callback: DetectionTerminalCallback | None = None
        self._handle: Any | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)

        self.image_layer_checkbox = QCheckBox("Image", self)
        self.image_layer_checkbox.setObjectName("detectionImageLayerCheckBox")
        self.image_layer_checkbox.setChecked(True)
        self.grid_layer_checkbox = QCheckBox("Annotation Grid", self)
        self.grid_layer_checkbox.setObjectName("detectionGridLayerCheckBox")
        self.grid_layer_checkbox.setChecked(True)
        self.map_layer_checkbox = QCheckBox("Defect Confidence Map", self)
        self.map_layer_checkbox.setObjectName("detectionMapLayerCheckBox")
        self.map_layer_checkbox.setChecked(True)

        layer_group = QGroupBox("Layers", self)
        layer_layout = QVBoxLayout(layer_group)
        layer_layout.addWidget(self.image_layer_checkbox)
        layer_layout.addWidget(self.grid_layer_checkbox)
        layer_layout.addWidget(self.map_layer_checkbox)

        self.class_selector = QComboBox(self)
        self.class_selector.setObjectName("detectionClassSelector")
        self.class_selector.currentIndexChanged.connect(self._class_changed)
        self.legend_label = QLabel("Legend: —", self)
        self.legend_label.setObjectName("detectionLegendLabel")
        self.legend_label.setWordWrap(True)
        self.map_preview_label = QLabel("Map: —", self)
        self.map_preview_label.setObjectName("detectionMapPreviewLabel")
        self.map_preview_label.setWordWrap(True)
        self.map_preview_label.setMinimumHeight(36)

        class_group = QGroupBox("Class Map", self)
        class_form = QFormLayout(class_group)
        class_form.addRow("Class", self.class_selector)
        class_form.addRow(self.legend_label)
        class_form.addRow(self.map_preview_label)

        self.opacity_slider = QSlider(self)
        self.opacity_slider.setObjectName("detectionOpacitySlider")
        self.opacity_slider.setOrientation(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(70)
        self.opacity_slider.valueChanged.connect(self._opacity_changed)
        self.opacity_label = QLabel("Opacity: 70%", self)
        self.opacity_label.setObjectName("detectionOpacityLabel")
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(self.opacity_slider)
        opacity_row.addWidget(self.opacity_label)

        self.x_spin = QSpinBox(self)
        self.x_spin.setObjectName("detectionXSpinBox")
        self.y_spin = QSpinBox(self)
        self.y_spin.setObjectName("detectionYSpinBox")
        self.inspect_button = QPushButton("Inspect Confidence", self)
        self.inspect_button.setObjectName("inspectDetectionConfidenceButton")
        self.inspect_button.clicked.connect(self._inspect_from_controls)
        self.confidence_label = QLabel("Confidence: —", self)
        self.confidence_label.setObjectName("detectionConfidenceLabel")
        self.confidence_label.setWordWrap(True)
        inspect_group = QGroupBox("Local Confidence (source pixels)", self)
        inspect_form = QFormLayout(inspect_group)
        inspect_form.addRow("Source x", self.x_spin)
        inspect_form.addRow("Source y", self.y_spin)
        inspect_form.addRow(self.inspect_button)
        inspect_form.addRow(self.confidence_label)

        self.disclaimer_label = QLabel(
            "Approximate localization; not a segmentation mask.", self
        )
        self.disclaimer_label.setObjectName("detectionDisclaimerLabel")
        self.disclaimer_label.setWordWrap(True)

        self.status_label = QLabel("Status: Not configured", self)
        self.status_label.setObjectName("detectionStatusLabel")
        self.status_label.setWordWrap(True)
        self.progress_label = QLabel("Progress: —", self)
        self.progress_label.setObjectName("detectionProgressLabel")
        self.progress_label.setWordWrap(True)
        self.start_button = QPushButton("Run Detection", self)
        self.start_button.setObjectName("startDetectionButton")
        self.start_button.clicked.connect(self.start_detection)
        self.cancel_button = QPushButton("Cancel Detection", self)
        self.cancel_button.setObjectName("cancelDetectionButton")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_detection)

        layout = QVBoxLayout(self)
        layout.addWidget(layer_group)
        layout.addWidget(class_group)
        layout.addWidget(QLabel("Map opacity", self))
        layout.addLayout(opacity_row)
        layout.addWidget(inspect_group)
        layout.addWidget(self.disclaimer_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.cancel_button)
        layout.addStretch(1)

        self._set_controls_enabled(False)
        self.map_layer_checkbox.setEnabled(False)
        self.image_layer_checkbox.toggled.connect(self._layers_changed)
        self.grid_layer_checkbox.toggled.connect(self._layers_changed)
        self.map_layer_checkbox.toggled.connect(self._layers_changed)

    @property
    def artifact(self) -> CamDetectionArtifact | None:
        return self._artifact

    def configure(
        self,
        artifact: CamDetectionArtifact | None = None,
        request: DetectionRequestSource | None = None,
        *,
        launcher: DetectionLauncher | None = None,
        request_source: DetectionRequestSource | None = None,
        terminal_callback: DetectionTerminalCallback | None = None,
    ) -> None:
        """Bind an immutable artifact and optional value-only worker request."""

        if request is not None and request_source is not None:
            raise ValueError("pass either request or request_source, not both")
        chosen_request = request if request is not None else request_source
        if artifact is not None and not isinstance(artifact, CamDetectionArtifact):
            raise TypeError("artifact must be a CamDetectionArtifact or None")
        if chosen_request is not None and not isinstance(chosen_request, DetectionRequest) and not callable(chosen_request):
            raise TypeError("request must be a DetectionRequest or request factory")
        self.set_artifact(artifact)
        self._request_source = chosen_request
        self._launcher = launcher or start_detection_worker
        self._terminal_callback = terminal_callback
        self._handle = None
        self._timer.stop()
        self.start_button.setEnabled(chosen_request is not None)
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Status: Ready" if chosen_request is not None else "Status: Artifact ready")
        self.progress_label.setText("Progress: —")

    def set_artifact(self, artifact: CamDetectionArtifact | None) -> None:
        """Replace the displayed result without writing project state."""

        if artifact is not None and not isinstance(artifact, CamDetectionArtifact):
            raise TypeError("artifact must be a CamDetectionArtifact or None")
        self._artifact = artifact
        self.class_selector.blockSignals(True)
        try:
            self.class_selector.clear()
            if artifact is not None:
                self.class_selector.addItems(list(artifact.class_names))
        finally:
            self.class_selector.blockSignals(False)
        has_artifact = artifact is not None and bool(artifact.class_names)
        self.map_layer_checkbox.setEnabled(has_artifact)
        self._set_controls_enabled(has_artifact)
        if artifact is None:
            self.legend_label.setText("Legend: —")
            self.map_preview_label.setText("Map: —")
            self.confidence_label.setText("Confidence: —")
            return
        width, height = int(artifact.source_width), int(artifact.source_height)
        self.x_spin.setRange(0, max(width - 1, 0))
        self.y_spin.setRange(0, max(height - 1, 0))
        self._render_selected_class()
        self._inspect_from_controls()

    def selected_class(self) -> str | None:
        """Return the class currently selected in the map and legend."""

        value = self.class_selector.currentText().strip()
        return value or None

    def selected_class_map(self) -> np.ndarray | None:
        artifact = self._artifact
        selected = self.selected_class()
        if artifact is None or selected is None:
            return None
        return artifact.class_map(selected)

    def inspect_local_confidence(self, x: int, y: int) -> float | None:
        """Inspect one source pixel in the selected class map."""

        artifact = self._artifact
        selected = self.selected_class()
        if artifact is None or selected is None:
            self.confidence_label.setText("Confidence: —")
            return None
        x_value, y_value = int(x), int(y)
        values = np.asarray(artifact.class_map(selected), dtype=float)
        if x_value < 0 or y_value < 0 or y_value >= values.shape[0] or x_value >= values.shape[1]:
            self.confidence_label.setText(f"Confidence at ({x_value}, {y_value}): outside source image")
            return None
        confidence = values[y_value, x_value]
        if not np.isfinite(confidence):
            self.confidence_label.setText(
                f"Confidence at ({x_value}, {y_value}) for {selected}: uncovered"
            )
            return None
        result = float(confidence)
        self.confidence_label.setText(
            f"Confidence at ({x_value}, {y_value}) for {selected}: {result:.3f}"
        )
        return result

    def start_detection(self) -> None:
        if self._handle is not None:
            return
        source = self._request_source
        if source is None:
            self.status_label.setText("Status: No Detection request configured")
            return
        try:
            request = source() if callable(source) else source
            if not isinstance(request, DetectionRequest):
                raise TypeError("request factory must return a DetectionRequest")
            handle = self._launcher(request)
        except Exception as error:
            self.status_label.setText(f"Status: Failed to start — {error}")
            return
        self._handle = handle
        self._timer.start()
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status_label.setText("Status: Running")
        self.progress_label.setText("Progress: 0/—")

    def cancel_detection(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            result = handle.cancel("user")
            if result is False:
                self.status_label.setText("Status: Cancellation already requested")
                return
        except Exception as error:
            self.status_label.setText(f"Status: Failed to cancel — {error}")
            return
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Status: Cancelling…")

    def _poll(self) -> None:
        handle = self._handle
        if handle is None:
            self._timer.stop()
            return
        output = getattr(handle, "queue", None)
        if output is None:
            self._finish("failed", "worker has no output queue")
            return
        while True:
            try:
                raw = output.get_nowait()
            except (queue.Empty, OSError, EOFError):
                break
            try:
                message = decode_message(raw) if isinstance(raw, str) else raw
            except Exception as error:
                self._finish("failed", f"invalid worker message: {error}")
                return
            if isinstance(message, DetectionProgress):
                self.progress_label.setText(
                    f"Progress: {message.completed}/{message.total} — {message.message}"
                )
            elif isinstance(message, DetectionTerminal):
                self._finish_message(message)
                return
        is_alive = getattr(handle, "is_alive", None)
        if callable(is_alive) and not is_alive():
            self._finish("interrupted", "Detection worker exited without a terminal status")

    def _finish_message(self, message: DetectionTerminal) -> None:
        if self._terminal_callback is not None:
            try:
                self._terminal_callback(message)
            except Exception as error:
                self._finish("failed", f"Failed to persist Detection Run — {error}")
                return
        if message.status == "completed" and message.artifact_staging_path:
            try:
                artifact = _load_staged_artifact(message.artifact_staging_path)
            except Exception as error:
                self._finish("failed", f"completed artifact could not be loaded: {error}")
                return
            self.set_artifact(artifact)
        self._finish(message.status, message.message)

    def _finish(self, status: str, detail: str) -> None:
        self._timer.stop()
        self._handle = None
        self.start_button.setEnabled(self._request_source is not None)
        self.cancel_button.setEnabled(False)
        text = status.capitalize()
        self.status_label.setText(f"Status: {text}" + (f" — {detail}" if detail else ""))

    def _class_changed(self, _index: int) -> None:
        self._render_selected_class()
        self._inspect_from_controls()

    def _render_selected_class(self) -> None:
        artifact = self._artifact
        selected = self.selected_class()
        if artifact is None or selected is None:
            self.legend_label.setText("Legend: —")
            self.map_preview_label.setText("Map: —")
            return
        values = np.asarray(artifact.class_map(selected), dtype=float)
        finite = values[np.isfinite(values)]
        maximum = float(np.max(finite)) if finite.size else 0.0
        coverage = np.asarray(artifact.coverage)
        covered = int(np.count_nonzero(coverage))
        self.legend_label.setText(f"Legend: {selected} | confidence 0.000–{maximum:.3f}")
        self.map_preview_label.setText(
            f"Map: {selected} | source {artifact.source_width}×{artifact.source_height} px | covered pixels {covered}"
        )
        self.map_preview_label.setVisible(self.map_layer_checkbox.isChecked())

    def _inspect_from_controls(self) -> None:
        self.inspect_local_confidence(self.x_spin.value(), self.y_spin.value())

    def _opacity_changed(self, value: int) -> None:
        self.opacity_label.setText(f"Opacity: {value}%")
        self.map_preview_label.setWindowOpacity(value / 100.0)

    def _layers_changed(self, _checked: bool) -> None:
        self._render_selected_class()

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.class_selector,
            self.opacity_slider,
            self.x_spin,
            self.y_spin,
            self.inspect_button,
        ):
            widget.setEnabled(enabled)

    def closeEvent(self, event) -> None:  # pragma: no cover - Qt teardown
        if self._handle is not None:
            try:
                self._handle.cancel("window closed")
            except Exception:
                pass
        self._timer.stop()
        super().closeEvent(event)


def _load_staged_artifact(staging_path: str | Path) -> CamDetectionArtifact:
    """Load the worker's JSON map stage into the value-only UI model."""

    root = Path(staging_path).expanduser().resolve()
    maps = json.loads((root / "maps.json").read_text(encoding="utf-8"))
    provenance_path = root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else maps.get("provenance", {})
    source = maps.get("source", {})
    values = np.asarray(maps["maps"], dtype=object)
    values = np.where(values == None, np.nan, values).astype(np.float64)  # noqa: E711
    coverage = np.asarray(maps.get("coverage", []), dtype=np.float64)
    return CamDetectionArtifact(
        maps=values,
        coverage=coverage,
        class_names=tuple(maps["class_names"]),
        source_width=int(source["width"]),
        source_height=int(source["height"]),
        source_transform=dict(maps.get("source_transform", {})),
        window_settings=dict(maps.get("window_settings", {})),
        provenance=dict(provenance),
        disclaimers=tuple(maps.get("disclaimers", ("Approximate localization", "not a segmentation mask"))),
    )


__all__ = ["DetectionControls", "DetectionLauncher", "DetectionRequestSource"]
