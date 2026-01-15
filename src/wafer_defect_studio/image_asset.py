"""Register external grayscale TIFF images as project image assets."""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from dataclasses import dataclass
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


def register_wafer_image(project_path: str | Path, source_path: str | Path) -> ImageAsset:
    """Validate and register one external 16-bit grayscale TIFF source."""

    project_info = open_project(project_path)
    image_path = Path(source_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    reader = QImageReader(str(image_path))
    image = reader.read()
    if image.isNull():
        raise ImageAssetError(f"Cannot read image source: {image_path}")
    image_format = bytes(reader.format()).decode("ascii", errors="ignore").upper()
    if image_format != "TIFF":
        raise ImageAssetError(f"Only grayscale TIFF image sources are supported: {image_path}")
    if image.format() != QImage.Format_Grayscale16:
        raise ImageAssetError(f"Image source must be grayscale16: {image_path}")

    fingerprint = _sha256(image_path)
    asset = ImageAsset(
        image_asset_id=str(uuid.uuid4()),
        path=image_path,
        width=image.width(),
        height=image.height(),
        dtype="uint16",
        format="TIFF",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
