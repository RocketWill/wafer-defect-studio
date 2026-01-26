"""Register external grayscale TIFF, PNG, and BMP image assets."""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from PySide6.QtGui import QImage, QImageReader

from .project import (
    ProjectError,
    _IMAGE_ASSETS_TABLE_SQL,
    _SCHEMA_VERSION,
    open_project,
)


class ImageAssetError(ProjectError):
    """Raised when an external source cannot be registered as a wafer image."""


class SourceHealth(Enum):
    AVAILABLE = "available"
    MISSING = "missing"
    CHANGED = "changed"


@dataclass(frozen=True)
class ImageAsset:
    """Immutable metadata for an externally referenced wafer image."""

    image_asset_id: str
    path: Path
    width: int
    height: int
    dtype: str
    format: str
    fingerprint: str


@dataclass(frozen=True)
class ReopenedWaferImage:
    """Persisted image metadata paired with its current source health."""

    asset: ImageAsset
    source_health: SourceHealth


_COLOR_FORMATS = {
    getattr(QImage, name)
    for name in (
        "Format_RGB16",
        "Format_RGB32",
        "Format_ARGB32",
        "Format_ARGB32_Premultiplied",
        "Format_RGB555",
        "Format_RGB565",
        "Format_RGB666",
        "Format_RGB888",
        "Format_RGBX8888",
        "Format_RGBA8888",
        "Format_RGBA8888_Premultiplied",
        "Format_BGR30",
        "Format_A2BGR30_Premultiplied",
        "Format_RGB30",
        "Format_A2RGB30_Premultiplied",
    )
    if hasattr(QImage, name)
}
_SUPPORTED_FORMATS = {"TIFF", "PNG", "BMP"}
_COLOR_ERROR = "Color images are not supported; provide a grayscale TIFF, PNG, or BMP."
_UNSUPPORTED_ERROR = "Unsupported image format; provide a grayscale TIFF, PNG, or BMP."


def register_wafer_image(project_path: str | Path, source_path: str | Path) -> ImageAsset:
    """Validate and register one external grayscale TIFF, PNG, or BMP source."""

    project_info = open_project(project_path)
    image_path = Path(source_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    reader = QImageReader(str(image_path))
    image_format = bytes(reader.format()).decode("ascii", errors="ignore").upper()
    image = reader.read()
    if image.isNull() or image_format not in _SUPPORTED_FORMATS:
        raise ImageAssetError(_UNSUPPORTED_ERROR)

    dtype_by_format = {
        "TIFF": {
            QImage.Format_Grayscale8: "uint8",
            QImage.Format_Grayscale16: "uint16",
        },
        "PNG": {
            QImage.Format_Grayscale8: "uint8",
            QImage.Format_Grayscale16: "uint16",
        },
        "BMP": {
            QImage.Format_Indexed8: "uint8",
        },
    }
    dtype = dtype_by_format[image_format].get(image.format())
    if dtype is None:
        if image.format() in _COLOR_FORMATS or image.depth() >= 16 or image.hasAlphaChannel():
            raise ImageAssetError(_COLOR_ERROR)
        raise ImageAssetError(_UNSUPPORTED_ERROR)

    fingerprint = _sha256(image_path)
    asset = ImageAsset(
        image_asset_id=str(uuid.uuid4()),
        path=image_path,
        width=image.width(),
        height=image.height(),
        dtype=dtype,
        format=image_format,
        fingerprint=fingerprint,
    )

    database_path = project_info.path / "project.sqlite"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if schema_version == 1:
            connection.execute(_IMAGE_ASSETS_TABLE_SQL)
            updated = connection.execute(
                "UPDATE project_metadata SET schema_version = ? "
                "WHERE project_id = ? AND schema_version = 1",
                (_SCHEMA_VERSION, project_info.project_id),
            ).rowcount
            if updated != 1:
                raise ImageAssetError(f"Invalid project metadata: {database_path}")
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        elif schema_version != _SCHEMA_VERSION:
            raise ImageAssetError(f"Unsupported project schema: {database_path}")

        connection.execute(
            "INSERT INTO image_assets "
            "(image_asset_id, path, width, height, dtype, format, fingerprint) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                asset.image_asset_id,
                str(asset.path),
                asset.width,
                asset.height,
                asset.dtype,
                asset.format,
                asset.fingerprint,
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    return asset


def load_image_assets(project_path: str | Path) -> tuple[ReopenedWaferImage, ...]:
    """Reopen persisted image metadata and report source health without writing."""

    project_info = open_project(project_path)
    if project_info.schema_version == 1:
        return ()

    database_path = project_info.path / "project.sqlite"
    try:
        connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise ImageAssetError(f"Cannot open project database: {database_path}") from error
    try:
        rows = connection.execute(
            "SELECT image_asset_id, path, width, height, dtype, format, fingerprint "
            "FROM image_assets ORDER BY image_asset_id"
        ).fetchall()
    except sqlite3.Error as error:
        raise ImageAssetError(f"Invalid image asset metadata: {database_path}") from error
    finally:
        connection.close()

    reopened = []
    for row in rows:
        asset = ImageAsset(
            image_asset_id=row[0],
            path=Path(row[1]).expanduser().resolve(),
            width=row[2],
            height=row[3],
            dtype=row[4],
            format=row[5],
            fingerprint=row[6],
        )
        reopened.append(ReopenedWaferImage(asset, _source_health(asset)))
    return tuple(reopened)


def _source_health(asset: ImageAsset) -> SourceHealth:
    path = asset.path.expanduser().resolve()
    if not path.is_file():
        return SourceHealth.MISSING
    if _sha256(path) == asset.fingerprint:
        return SourceHealth.AVAILABLE
    return SourceHealth.CHANGED


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
