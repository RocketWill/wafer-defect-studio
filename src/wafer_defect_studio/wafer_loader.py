"""Background image decoding with request-token result ordering."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot

from .wafer_view import LoadedWaferImage, _decode_wafer_image


class _DecodeWorker(QObject):
    loaded = Signal(object, object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    @Slot()
    def run(self) -> None:
        try:
            loaded, image = _decode_wafer_image(self._path)
        except Exception as error:
            self.failed.emit(str(error))
        else:
            self.loaded.emit(loaded, image)
        finally:
            self.finished.emit()


class WaferLoader(QObject):
    """Run one decode worker per request and report tokenized outcomes."""

    loaded = Signal(int, object, object)
    failed = Signal(int, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._next_token = 0
        self._threads: dict[int, QThread] = {}
        self._workers: dict[int, _DecodeWorker] = {}

    def request(self, path: str | Path) -> int:
        self._next_token += 1
        token = self._next_token
        thread = QThread(self)
        worker = _DecodeWorker(Path(path))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.loaded.connect(lambda loaded, image, token=token: self.loaded.emit(token, loaded, image))
        worker.failed.connect(lambda message, token=token: self.failed.emit(token, message))
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda token=token: self._cleanup(token))
        self._threads[token] = thread
        self._workers[token] = worker
        thread.start()
        return token

    def _cleanup(self, token: int) -> None:
        self._workers.pop(token, None)
        self._threads.pop(token, None)
