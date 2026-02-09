"""Small non-blocking Widgets for one background Training Run."""

from __future__ import annotations

import queue
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QPlainTextEdit, QLabel, QPushButton, QVBoxLayout, QWidget

from .training_protocol import ProgressMessage, TerminalMessage, TrainingRequest, decode_message
from .training_worker import start_training_worker


TrainingRequestSource = TrainingRequest | Callable[[], TrainingRequest]
TrainingLauncher = Callable[[TrainingRequest], Any]
CloneCallback = Callable[[], Any]


class TrainingControls(QWidget):
    """Start and observe one value-only worker without writing project state."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._request_source: TrainingRequestSource | None = None
        self._launcher: TrainingLauncher = start_training_worker
        self._clone_callback: CloneCallback | None = None
        self._handle: Any | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)

        self.status_label = QLabel("Status: Not configured", self)
        self.status_label.setObjectName("trainingStatusLabel")
        self.status_label.setWordWrap(True)
        self.progress_label = QLabel("Phase: — | Epoch: — | ETA: — | Loss: —", self)
        self.progress_label.setObjectName("trainingProgressLabel")
        self.phase_label = QLabel("Phase: —", self)
        self.phase_label.setObjectName("trainingPhaseLabel")
        self.epoch_label = QLabel("Epoch: —", self)
        self.epoch_label.setObjectName("trainingEpochLabel")
        self.eta_label = QLabel("ETA: —", self)
        self.eta_label.setObjectName("trainingEtaLabel")
        self.loss_label = QLabel("Loss: —", self)
        self.loss_label.setObjectName("trainingLossLabel")
        self.resource_label = QLabel("Resource: —", self)
        self.resource_label.setObjectName("trainingResourceLabel")
        self.log_text = QPlainTextEdit(self)
        self.log_text.setObjectName("trainingLogText")
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(500)

        self.start_button = QPushButton("Start Training", self)
        self.start_button.setObjectName("startTrainingButton")
        self.start_button.clicked.connect(self.start_training)
        self.cancel_button = QPushButton("Cancel Training", self)
        self.cancel_button.setObjectName("cancelTrainingButton")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_training)
        self.clone_button = QPushButton("Clone with Smaller Batch", self)
        self.clone_button.setObjectName("cloneOomTrainingButton")
        self.clone_button.setEnabled(False)
        self.clone_button.clicked.connect(self.clone_after_oom)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.phase_label)
        layout.addWidget(self.epoch_label)
        layout.addWidget(self.eta_label)
        layout.addWidget(self.loss_label)
        layout.addWidget(self.resource_label)
        layout.addWidget(self.log_text)
        layout.addWidget(self.start_button)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.clone_button)

    def configure(
        self,
        request: TrainingRequestSource,
        *,
        launcher: TrainingLauncher | None = None,
        clone_callback: CloneCallback | None = None,
    ) -> None:
        """Bind a request and service actions; no database handle crosses this boundary."""

        if not isinstance(request, TrainingRequest) and not callable(request):
            raise TypeError("request must be a TrainingRequest or a request factory")
        self._request_source = request
        self._launcher = launcher or start_training_worker
        self._clone_callback = clone_callback
        self._handle = None
        self._timer.stop()
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.clone_button.setEnabled(False)
        self.status_label.setText("Status: Ready to start")

    def start_training(self) -> None:
        if self._handle is not None:
            return
        if self._request_source is None:
            self.status_label.setText("Status: Not configured")
            return
        try:
            request = (
                self._request_source()
                if callable(self._request_source)
                else self._request_source
            )
            if not isinstance(request, TrainingRequest):
                raise TypeError("request factory must return a TrainingRequest")
            handle = self._launcher(request)
        except Exception as error:
            self.status_label.setText(f"Status: Failed to start — {error}")
            return
        self._handle = handle
        self._timer.start()
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.clone_button.setEnabled(False)
        self.status_label.setText("Status: Running")
        self.phase_label.setText("Phase: —")
        self.epoch_label.setText("Epoch: —")
        self.eta_label.setText("ETA: —")
        self.loss_label.setText("Loss: —")
        self.progress_label.setText("Phase: — | Epoch: — | ETA: — | Loss: —")
        device = request.config.device
        resource = "auto (CUDA when available)" if device == "auto" else device
        self.resource_label.setText(f"Resource: {resource}")
        self.log_text.clear()
        self._append_log(f"Started {request.request_id}; device={device}")

    def cancel_training(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.cancel("user")
        except Exception as error:
            self.status_label.setText(f"Status: Failed to cancel — {error}")
            return
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Status: Cancelling…")
        self._append_log("Cancellation requested")

    def clone_after_oom(self) -> None:
        if self._clone_callback is None or self.clone_button.isEnabled() is False:
            return
        try:
            result = self._clone_callback()
        except Exception as error:
            self.status_label.setText(f"Status: OOM clone failed — {error}")
            return
        self.clone_button.setEnabled(False)
        detail = getattr(result, "run_id", "")
        suffix = f" ({detail})" if detail else ""
        self.status_label.setText(f"Status: OOM clone requested{suffix}")
        self._append_log("OOM clone requested with an explicitly smaller batch size")

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
            if isinstance(message, ProgressMessage):
                self._show_progress(message)
            elif isinstance(message, TerminalMessage):
                self._finish_message(message)
                return
        is_alive = getattr(handle, "is_alive", None)
        if callable(is_alive) and not is_alive():
            self._finish("failed", "worker exited without a terminal status")

    def _show_progress(self, message: ProgressMessage) -> None:
        self.phase_label.setText(f"Phase: {message.phase}")
        self.epoch_label.setText(f"Epoch: {message.epoch}/{message.total_epochs}")
        eta = "—" if message.eta_seconds is None else f"{message.eta_seconds:.1f}s"
        self.eta_label.setText(f"ETA: {eta}")
        loss = "—" if message.loss is None else f"{message.loss:.6f}"
        self.loss_label.setText(f"Loss: {loss}")
        self.progress_label.setText(
            f"Phase: {message.phase} | Epoch: {message.epoch}/{message.total_epochs} "
            f"| ETA: {eta} | Loss: {loss}"
        )
        if "device=" in message.message:
            self.resource_label.setText(f"Resource: {message.message}")
        self._append_log(
            f"{message.phase} epoch={message.epoch}/{message.total_epochs} "
            f"step={message.step}/{message.total_steps} loss={loss}"
        )

    def _finish_message(self, message: TerminalMessage) -> None:
        self._finish(message.status, message.message, message.error_code)

    def _finish(self, status: str, detail: str, error_code: str | None = None) -> None:
        self._timer.stop()
        self._handle = None
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        oom = status == "failed" and (
            error_code in {"out_of_memory", "oom"}
            or "out of memory" in detail.lower()
            or "oom" in detail.lower()
        )
        self.clone_button.setEnabled(bool(oom and self._clone_callback is not None))
        text = status.capitalize()
        self.status_label.setText(f"Status: {text}" + (f" — {detail}" if detail else ""))
        if detail:
            self._append_log(detail)

    def _append_log(self, line: str) -> None:
        self.log_text.appendPlainText(line)

    def closeEvent(self, event) -> None:  # pragma: no cover - exercised by Qt teardown
        if self._handle is not None:
            try:
                self._handle.cancel("window closed")
            except Exception:
                pass
        self._timer.stop()
        super().closeEvent(event)


__all__ = ["TrainingControls"]
