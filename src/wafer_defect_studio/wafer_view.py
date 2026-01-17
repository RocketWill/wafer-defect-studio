"""Decode a registered wafer image while preserving its native pixels."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from pathlib import Path
import sys

from PySide6.QtGui import QImage, QImageReader


@dataclass(frozen=True)
class LoadedWaferImage:
    """Native grayscale pixels decoded for display and coordinate lookup."""

    width: int
    height: int
    dtype: str
    pixels: array


def _decode_wafer_image(path: Path) -> tuple[LoadedWaferImage, QImage]:
    reader = QImageReader(str(path))
    image = reader.read()
    if image.isNull():
        raise ValueError(f"Unable to decode wafer image: {path}")
    if image.format() != QImage.Format_Grayscale16:
        raise ValueError(f"Wafer image must be grayscale16: {path}")

    width = image.width()
    height = image.height()
    row_bytes = width * 2
    stride = image.bytesPerLine()
    source = image.constBits()
    pixels = array("H")
    for row_index in range(height):
        row = array("H")
        row.frombytes(bytes(source[row_index * stride : row_index * stride + row_bytes]))
        if sys.byteorder == "big":
            row.byteswap()
        pixels.extend(row)

    return LoadedWaferImage(width, height, "uint16", pixels), image
