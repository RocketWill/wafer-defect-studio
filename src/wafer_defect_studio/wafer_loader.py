"""Background image decoding with request-token result ordering."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot

from .wafer_view import LoadedWaferImage, _decode_wafer_image


class _DecodeWorker(QObject):
    loaded = Signal(int, object, object)
    failed = Signal(int, str)
    finished = Signal()

    def __init__(self, token: int, path: Path) -> None:
        super().__init__()
        self._token = token
        self._path = path

    @Slot()
    def run(self) -> None:
        try:
            loaded, image = _decode_wafer_image(self._path)
        except Exception as error:
            self.failed.emit(self._token, str(error))
        else:
            self.loaded.emit(self._token, loaded, image)
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
        worker = _DecodeWorker(token, Path(path))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.loaded.connect(self.loaded)
        worker.failed.connect(self.failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._cleanup_finished)
        self._threads[token] = thread
        self._workers[token] = worker
        thread.start()
        return token

    def _cleanup(self, token: int) -> None:
        self._threads.pop(token, None)

    @Slot()
    def _cleanup_finished(self) -> None:
        thread = self.sender()
        for token, candidate in tuple(self._threads.items()):
            if candidate is thread:
                self._cleanup(token)
                return
