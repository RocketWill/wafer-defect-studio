"""Small, synchronous autosave failure guard for annotation edits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from .annotation import GridAnnotation


SaveCallback = Callable[[GridAnnotation], None]
RestoreCallback = Callable[[GridAnnotation, tuple[str, ...]], None]
ApplyCallback = Callable[[GridAnnotation], None]
SaveAsCallback = Callable[[Path, GridAnnotation], None]


@dataclass(frozen=True)
class SaveFailureState:
    """The failed edit and the value that was visible before it was attempted."""

    annotation: GridAnnotation
    previous_class_codes: tuple[str, ...]
    error_message: str

    @property
    def image_asset_id(self) -> str:
        return self.annotation.image_asset_id

    @property
    def row(self) -> int:
        return self.annotation.row

    @property
    def column(self) -> int:
        return self.annotation.column

    @property
    def class_codes(self) -> tuple[str, ...]:
        return self.annotation.class_codes


class AutosaveGuard:
    """Attempt annotation saves and hold a failed edit until it is resolved.

    The callbacks deliberately keep this object independent from Qt.  The caller
    owns the in-memory cell and supplies callbacks to restore it after a failed
    write and to apply it after Retry or Save As succeeds.
    """

    def __init__(
        self,
        save_callback: SaveCallback,
        *,
        restore_callback: RestoreCallback | None = None,
        apply_callback: ApplyCallback | None = None,
        save_as_callback: SaveAsCallback | None = None,
    ) -> None:
        if not callable(save_callback):
            raise ValueError("save_callback must be callable")
        self._save_callback = save_callback
        self._restore_callback = restore_callback
        self._apply_callback = apply_callback
        self._save_as_callback = save_as_callback
        self._pending: SaveFailureState | None = None

    @property
    def pending(self) -> SaveFailureState | None:
        """Return the pending failure, if a failed edit is awaiting an action."""

        return self._pending

    @property
    def failure(self) -> SaveFailureState | None:
        """Alias for :attr:`pending` used by UI callers."""

        return self._pending

    @property
    def blocked(self) -> bool:
        """Whether image switching is currently unsafe."""

        return self._pending is not None

    def autosave(
        self,
        annotation: GridAnnotation,
        previous_class_codes: tuple[str, ...],
    ) -> bool:
        """Persist one edit, rolling it back and recording failure on exception."""

        if not isinstance(annotation, GridAnnotation):
            raise ValueError("annotation must be a GridAnnotation")
        previous = tuple(previous_class_codes)
        if self._pending is not None:
            # Do not replace the first failed edit with a later unsaved edit.
            self._restore(annotation, previous)
            return False
        try:
            self._save_callback(annotation)
        except Exception as error:
            self._pending = SaveFailureState(
                annotation,
                previous,
                _error_message(error),
            )
            self._restore(annotation, previous)
            return False
        return True

    # ``save`` and ``save_change`` make the pure seam convenient to call from
    # small non-Qt services while keeping ``autosave`` the descriptive name.
    save = autosave
    save_change = autosave

    def request_image_switch(self, _target=None) -> bool:
        """Return whether an image switch may proceed right now."""

        return self._pending is None

    request_switch = request_image_switch

    def retry(self) -> bool:
        """Retry the original persistence operation; keep the guard on failure."""

        pending = self._pending
        if pending is None:
            return False
        try:
            self._save_callback(pending.annotation)
        except Exception as error:
            self._pending = replace(pending, error_message=_error_message(error))
            return False
        self._apply(pending.annotation)
        self._pending = None
        return True

    def save_as(self, target_path: str | Path) -> bool:
        """Persist the pending edit through the caller's Save As callback."""

        pending = self._pending
        if pending is None:
            return False
        if self._save_as_callback is None:
            self._pending = replace(
                pending,
                error_message="Save As is unavailable for this project",
            )
            return False
        path = Path(target_path).expanduser()
        try:
            self._save_as_callback(path, pending.annotation)
        except Exception as error:
            self._pending = replace(pending, error_message=_error_message(error))
            return False
        self._apply(pending.annotation)
        self._pending = None
        return True

    def cancel(self) -> bool:
        """Discard the failed edit and unblock switching while keeping prior state."""

        if self._pending is None:
            return False
        self._pending = None
        return True

    def _restore(self, annotation: GridAnnotation, previous: tuple[str, ...]) -> None:
        if self._restore_callback is not None:
            self._restore_callback(annotation, previous)

    def _apply(self, annotation: GridAnnotation) -> None:
        if self._apply_callback is not None:
            self._apply_callback(annotation)


def _error_message(error: Exception) -> str:
    message = str(error).strip()
    return message or error.__class__.__name__
