"""Deterministic native-coordinate inference-window enumeration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


class Rect(NamedTuple):
    """An ``x, y, width, height`` rectangle in source-image coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


class Padding(NamedTuple):
    """Reflect padding in ``left, top, right, bottom`` order."""

    left: int
    top: int
    right: int
    bottom: int


class Point(NamedTuple):
    """An ``x, y`` point in source-image coordinates."""

    x: int | float
    y: int | float


@dataclass(frozen=True)
class Window:
    """One full-size inference window and its source-coordinate transform.

    ``read_rect`` is the requested model input rectangle.  It always has the
    configured window dimensions and may extend past the source image.  The
    clipped ``source_rect`` identifies source pixels actually covered by the
    request; out-of-bounds parts are supplied by reflect padding.
    """

    index: int
    source_rect: Rect
    read_rect: Rect
    padding: Padding
    center: Point
    source_width: int
    source_height: int

    @property
    def padded_rect(self) -> Rect:
        """Alias for the full requested/read rectangle."""

        return self.read_rect

    @property
    def window_rect(self) -> Rect:
        """Alias used by callers that call the model input a window."""

        return self.read_rect

    @property
    def x(self) -> int:
        return self.read_rect.x

    @property
    def y(self) -> int:
        return self.read_rect.y

    @property
    def width(self) -> int:
        return self.read_rect.width

    @property
    def height(self) -> int:
        return self.read_rect.height

    @property
    def center_x(self) -> int | float:
        return self.center.x

    @property
    def center_y(self) -> int | float:
        return self.center.y

    @property
    def window_center(self) -> Point:
        """Geometric center before clipping (may lie in reflected padding)."""

        return Point(
            self.read_rect.x + self.read_rect.width / 2,
            self.read_rect.y + self.read_rect.height / 2,
        )

    def window_to_source(self, x: int, y: int) -> Point:
        """Map a local model-input pixel through reflect padding to the source."""

        _local_coordinate(x, self.read_rect.width, "x")
        _local_coordinate(y, self.read_rect.height, "y")
        return Point(
            _reflect_index(self.read_rect.x + x, self.source_width),
            _reflect_index(self.read_rect.y + y, self.source_height),
        )

    def source_to_window(self, x: int, y: int) -> Point:
        """Return the canonical local coordinate for a covered source pixel.

        A source pixel is canonicalized to its first occurrence in the
        unpadded portion of this window.  Pixels outside ``source_rect`` are
        not part of this window and are rejected.
        """

        _local_coordinate(x, self.source_width, "x", allow_zero=True)
        _local_coordinate(y, self.source_height, "y", allow_zero=True)
        if not (
            self.source_rect.x <= x < self.source_rect.right
            and self.source_rect.y <= y < self.source_rect.bottom
        ):
            raise ValueError("source coordinate is outside this inference window")
        return Point(x - self.read_rect.x, y - self.read_rect.y)

    # Explicit aliases make the transform readable at call sites.
    local_to_source = window_to_source
    source_to_local = source_to_window


InferenceWindow = Window


def enumerate_inference_windows(
    source_width: int,
    source_height: int,
    window_size: int | tuple[int, int],
    stride: int | tuple[int, int],
) -> tuple[Window, ...]:
    """Enumerate overlapping windows over a native-resolution source.

    Starts are placed at ``0, stride, ...`` while they remain inside the
    source.  This guarantees edge coverage and leaves the final window
    reflect-padded when its full model input extends past the source boundary.
    No source pixels are resized or otherwise modified by this geometry-only
    operation.
    """

    _positive_integer("source_width", source_width)
    _positive_integer("source_height", source_height)
    window_width, window_height = _pair(window_size, "window_size")
    stride_x, stride_y = _pair(stride, "stride")
    if stride_x > window_width or stride_y > window_height:
        raise ValueError("stride must not exceed window_size so windows overlap or touch")

    windows: list[Window] = []
    index = 0
    for top in range(0, source_height, stride_y):
        for left in range(0, source_width, stride_x):
            source_right = min(left + window_width, source_width)
            source_bottom = min(top + window_height, source_height)
            source_rect = Rect(
                left,
                top,
                source_right - left,
                source_bottom - top,
            )
            read_rect = Rect(left, top, window_width, window_height)
            padding = Padding(
                left=0,
                top=0,
                right=read_rect.right - source_right,
                bottom=read_rect.bottom - source_bottom,
            )
            windows.append(
                Window(
                    index=index,
                    source_rect=source_rect,
                    read_rect=read_rect,
                    padding=padding,
                    center=Point(
                        source_rect.x + source_rect.width / 2,
                        source_rect.y + source_rect.height / 2,
                    ),
                    source_width=source_width,
                    source_height=source_height,
                )
            )
            index += 1
    return tuple(windows)


def enumerate_windows(
    source_width: int,
    source_height: int,
    window_size: int | tuple[int, int],
    stride: int | tuple[int, int],
) -> tuple[Window, ...]:
    """Short alias for :func:`enumerate_inference_windows`."""

    return enumerate_inference_windows(source_width, source_height, window_size, stride)


def _pair(value: object, name: str) -> tuple[int, int]:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer or a pair")
    if isinstance(value, int):
        _positive_integer(name, value)
        return value, value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        first, second = value
        _positive_integer(f"{name}[0]", first)
        _positive_integer(f"{name}[1]", second)
        return first, second
    raise ValueError(f"{name} must be a positive integer or a pair")


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _local_coordinate(value: object, length: int, name: str, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 0
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value < length:
        raise ValueError(f"{name} coordinate must be an integer inside the rectangle")


def _reflect_index(value: int, length: int) -> int:
    if length == 1:
        return 0
    period = 2 * (length - 1)
    folded = value % period
    return folded if folded < length else period - folded


__all__ = [
    "InferenceWindow",
    "Padding",
    "Point",
    "Rect",
    "Window",
    "enumerate_inference_windows",
    "enumerate_windows",
]
