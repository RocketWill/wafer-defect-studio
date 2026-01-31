"""Persisted effective-wafer-area geometry for each imported image."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .project import (
    _EFFECTIVE_WAFER_AREAS_TABLE_SQL,
    _IMAGE_GRID_PLACEMENT_SCHEMA_VERSION,
    _SCHEMA_VERSION,
    ProjectError,
    open_project,
)
from .grid_geometry import AnnotationGrid


@dataclass(frozen=True)
class EllipseGeometry:
    center_x: int
    center_y: int
    radius_x: int
    radius_y: int


@dataclass(frozen=True)
class SourcePoint:
    x: int
    y: int


@dataclass(frozen=True)
class PolygonGeometry:
    vertices: tuple[SourcePoint, ...]


@dataclass(frozen=True)
class EffectiveWaferArea:
    image_asset_id: str
    shape: str
    geometry: EllipseGeometry | PolygonGeometry
    confirmed: bool


class EffectiveWaferAreaError(ProjectError):
    """Raised when an effective-wafer-area record is invalid or unavailable."""


def set_effective_ellipse(
    project_path: str | Path,
    image_asset_id: str,
    center_x: int,
    center_y: int,
    radius_x: int,
    radius_y: int,
) -> EffectiveWaferArea:
    """Replace the current image area with an unconfirmed source-aligned ellipse."""

    _identifier(image_asset_id)
    _integer("center_x", center_x)
    _integer("center_y", center_y)
    _integer("radius_x", radius_x)
    _integer("radius_y", radius_y)
    if radius_x <= 0 or radius_y <= 0:
        raise ValueError("ellipse radii must be positive")

    project_info = open_project(project_path)
    if project_info.schema_version < _IMAGE_GRID_PLACEMENT_SCHEMA_VERSION:
        raise EffectiveWaferAreaError("Effective wafer areas require project schema 5 or newer")

    database_path = project_info.path / "project.sqlite"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]

        image = connection.execute(
            "SELECT width, height FROM image_assets WHERE image_asset_id = ?",
            (image_asset_id,),
        ).fetchone()
        if image is None:
            raise EffectiveWaferAreaError(f"Unknown image asset: {image_asset_id}")
        image_width, image_height = int(image[0]), int(image[1])
        if (
            center_x - radius_x < 0
            or center_y - radius_y < 0
            or center_x + radius_x > image_width
            or center_y + radius_y > image_height
        ):
            raise ValueError("ellipse must lie wholly within the source image")

        if schema_version == _IMAGE_GRID_PLACEMENT_SCHEMA_VERSION:
            connection.execute(_EFFECTIVE_WAFER_AREAS_TABLE_SQL)
            updated = connection.execute(
                "UPDATE project_metadata SET schema_version = ? "
                "WHERE project_id = ? AND schema_version = 5",
                (_SCHEMA_VERSION, project_info.project_id),
            ).rowcount
            if updated != 1:
                raise EffectiveWaferAreaError(f"Invalid project metadata: {database_path}")
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        elif schema_version != _SCHEMA_VERSION:
            raise EffectiveWaferAreaError(f"Unsupported project schema: {database_path}")

        geometry = EllipseGeometry(center_x, center_y, radius_x, radius_y)
        connection.execute(
            "INSERT INTO effective_wafer_areas "
            "(image_asset_id, shape, geometry_json, confirmed) "
            "VALUES (?, 'ellipse', ?, 0) "
            "ON CONFLICT(image_asset_id) DO UPDATE SET "
            "shape = excluded.shape, geometry_json = excluded.geometry_json, "
            "confirmed = excluded.confirmed",
            (image_asset_id, _geometry_json(geometry)),
        )
        connection.commit()
    except sqlite3.Error as error:
        connection.rollback()
        raise EffectiveWaferAreaError(
            f"Invalid effective wafer area metadata: {database_path}"
        ) from error
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    return EffectiveWaferArea(image_asset_id, "ellipse", geometry, False)


def set_effective_polygon(
    project_path: str | Path,
    image_asset_id: str,
    vertices: Sequence[tuple[int, int] | SourcePoint],
) -> EffectiveWaferArea:
    """Replace the current image area with an unconfirmed simple polygon."""

    _identifier(image_asset_id)
    polygon = PolygonGeometry(_normalize_polygon(vertices))

    project_info = open_project(project_path)
    if project_info.schema_version < _IMAGE_GRID_PLACEMENT_SCHEMA_VERSION:
        raise EffectiveWaferAreaError("Effective wafer areas require project schema 5 or newer")

    database_path = project_info.path / "project.sqlite"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        image = connection.execute(
            "SELECT width, height FROM image_assets WHERE image_asset_id = ?",
            (image_asset_id,),
        ).fetchone()
        if image is None:
            raise EffectiveWaferAreaError(f"Unknown image asset: {image_asset_id}")
        _validate_polygon_bounds(polygon, int(image[0]), int(image[1]))

        if schema_version == _IMAGE_GRID_PLACEMENT_SCHEMA_VERSION:
            connection.execute(_EFFECTIVE_WAFER_AREAS_TABLE_SQL)
            updated = connection.execute(
                "UPDATE project_metadata SET schema_version = ? "
                "WHERE project_id = ? AND schema_version = 5",
                (_SCHEMA_VERSION, project_info.project_id),
            ).rowcount
            if updated != 1:
                raise EffectiveWaferAreaError(f"Invalid project metadata: {database_path}")
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        elif schema_version != _SCHEMA_VERSION:
            raise EffectiveWaferAreaError(f"Unsupported project schema: {database_path}")

        connection.execute(
            "INSERT INTO effective_wafer_areas "
            "(image_asset_id, shape, geometry_json, confirmed) "
            "VALUES (?, 'polygon', ?, 0) "
            "ON CONFLICT(image_asset_id) DO UPDATE SET "
            "shape = excluded.shape, geometry_json = excluded.geometry_json, "
            "confirmed = excluded.confirmed",
            (image_asset_id, _polygon_json(polygon)),
        )
        connection.commit()
    except sqlite3.Error as error:
        connection.rollback()
        raise EffectiveWaferAreaError(
            f"Invalid effective wafer area metadata: {database_path}"
        ) from error
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    return EffectiveWaferArea(image_asset_id, "polygon", polygon, False)


def load_effective_wafer_area(
    project_path: str | Path, image_asset_id: str
) -> EffectiveWaferArea | None:
    """Read the current image area without writing or migrating the project."""

    _identifier(image_asset_id)
    project_info = open_project(project_path)
    if project_info.schema_version < _IMAGE_GRID_PLACEMENT_SCHEMA_VERSION:
        raise EffectiveWaferAreaError("Effective wafer areas require project schema 5 or newer")
    database_path = project_info.path / "project.sqlite"
    try:
        connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise EffectiveWaferAreaError(f"Cannot open project database: {database_path}") from error
    try:
        image = connection.execute(
            "SELECT width, height FROM image_assets WHERE image_asset_id = ?",
            (image_asset_id,),
        ).fetchone()
        if image is None:
            raise EffectiveWaferAreaError(f"Unknown image asset: {image_asset_id}")
        if project_info.schema_version == _IMAGE_GRID_PLACEMENT_SCHEMA_VERSION:
            return None
        row = connection.execute(
            "SELECT image_asset_id, shape, geometry_json, confirmed "
            "FROM effective_wafer_areas WHERE image_asset_id = ?",
            (image_asset_id,),
        ).fetchone()
    except sqlite3.Error as error:
        raise EffectiveWaferAreaError(
            f"Invalid effective wafer area metadata: {database_path}"
        ) from error
    finally:
        connection.close()

    if row is None:
        return None
    persisted_id, shape, geometry_json, confirmed = row
    try:
        values = json.loads(geometry_json)
        if shape == "ellipse":
            if not isinstance(values, dict) or tuple(values) != (
                "center_x",
                "center_y",
                "radius_x",
                "radius_y",
            ):
                raise ValueError
            geometry = EllipseGeometry(
                _stored_integer(values["center_x"]),
                _stored_integer(values["center_y"]),
                _stored_integer(values["radius_x"]),
                _stored_integer(values["radius_y"]),
            )
        elif shape == "polygon":
            if not isinstance(values, dict) or tuple(values) != ("vertices",):
                raise ValueError
            geometry = PolygonGeometry(_normalize_polygon(values["vertices"]))
        else:
            raise ValueError
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise EffectiveWaferAreaError(
            f"Invalid effective wafer area geometry: {database_path}"
        ) from error
    if isinstance(geometry, EllipseGeometry):
        if (
            geometry.radius_x <= 0
            or geometry.radius_y <= 0
            or geometry.center_x - geometry.radius_x < 0
            or geometry.center_y - geometry.radius_y < 0
            or geometry.center_x + geometry.radius_x > int(image[0])
            or geometry.center_y + geometry.radius_y > int(image[1])
        ):
            raise EffectiveWaferAreaError(
                f"Invalid effective wafer area bounds: {database_path}"
            )
    else:
        try:
            _validate_polygon_bounds(geometry, int(image[0]), int(image[1]))
        except ValueError as error:
            raise EffectiveWaferAreaError(
                f"Invalid effective wafer area bounds: {database_path}"
            ) from error
    if confirmed not in (0, 1):
        raise EffectiveWaferAreaError(f"Invalid effective wafer area confirmation: {database_path}")
    return EffectiveWaferArea(str(persisted_id), str(shape), geometry, bool(confirmed))


def participating_annotation_grids(
    grids: Iterable[AnnotationGrid], effective_area: EffectiveWaferArea
) -> tuple[AnnotationGrid, ...]:
    """Return grid cells whose full-cell centers lie strictly inside an area."""

    if not isinstance(effective_area, EffectiveWaferArea):
        raise ValueError("effective_area must be an EffectiveWaferArea")
    if effective_area.shape == "ellipse" and isinstance(effective_area.geometry, EllipseGeometry):
        contains = _ellipse_contains
    elif effective_area.shape == "polygon" and isinstance(effective_area.geometry, PolygonGeometry):
        contains = _polygon_contains
    else:
        raise ValueError("effective area shape and geometry do not match")
    return tuple(grid for grid in grids if contains(grid, effective_area.geometry))


def confirmed_participating_grids(
    grids: Iterable[AnnotationGrid], effective_area: EffectiveWaferArea
) -> tuple[AnnotationGrid, ...]:
    """Return participating cells only after explicit area confirmation."""

    if not isinstance(effective_area, EffectiveWaferArea):
        raise ValueError("effective_area must be an EffectiveWaferArea")
    if not effective_area.confirmed:
        raise EffectiveWaferAreaError("Effective wafer area is unconfirmed")
    return participating_annotation_grids(grids, effective_area)


def confirm_effective_wafer_area(
    project_path: str | Path, image_asset_id: str
) -> EffectiveWaferArea:
    """Mark an existing image area confirmed, idempotently, in one transaction."""

    _identifier(image_asset_id)
    project_info = open_project(project_path)
    if project_info.schema_version != _SCHEMA_VERSION:
        raise EffectiveWaferAreaError("Effective wafer area confirmation requires project schema 6")

    database_path = project_info.path / "project.sqlite"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT 1 FROM effective_wafer_areas WHERE image_asset_id = ?",
            (image_asset_id,),
        ).fetchone()
        if row is None:
            raise EffectiveWaferAreaError(f"No effective wafer area for image: {image_asset_id}")
        connection.execute(
            "UPDATE effective_wafer_areas SET confirmed = 1 WHERE image_asset_id = ?",
            (image_asset_id,),
        )
        connection.commit()
    except sqlite3.Error as error:
        connection.rollback()
        raise EffectiveWaferAreaError(
            f"Invalid effective wafer area metadata: {database_path}"
        ) from error
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    area = load_effective_wafer_area(project_info.path, image_asset_id)
    if area is None:
        raise EffectiveWaferAreaError(f"No effective wafer area for image: {image_asset_id}")
    return area


def _ellipse_contains(grid: AnnotationGrid, geometry: EllipseGeometry) -> bool:
    grid_center_x2 = 2 * grid.x + grid.width
    grid_center_y2 = 2 * grid.y + grid.height
    delta_x = grid_center_x2 - 2 * geometry.center_x
    delta_y = grid_center_y2 - 2 * geometry.center_y
    radius_x = geometry.radius_x
    radius_y = geometry.radius_y
    left = delta_x * delta_x * radius_y * radius_y + delta_y * delta_y * radius_x * radius_x
    right = (2 * radius_x * radius_y) ** 2
    return left < right


def _polygon_contains(grid: AnnotationGrid, geometry: PolygonGeometry) -> bool:
    point = SourcePoint(2 * grid.x + grid.width, 2 * grid.y + grid.height)
    vertices = tuple(SourcePoint(2 * vertex.x, 2 * vertex.y) for vertex in geometry.vertices)
    for start, end in zip(vertices, vertices[1:] + vertices[:1]):
        if _on_segment(start, end, point):
            return False

    inside = False
    for start, end in zip(vertices, vertices[1:] + vertices[:1]):
        if (start.y <= point.y < end.y) or (end.y <= point.y < start.y):
            delta_y = end.y - start.y
            numerator = (start.x - point.x) * delta_y + (point.y - start.y) * (
                end.x - start.x
            )
            crosses_right = numerator > 0 if delta_y > 0 else numerator < 0
            if crosses_right:
                inside = not inside
    return inside


def _geometry_json(geometry: EllipseGeometry) -> str:
    return json.dumps(
        {
            "center_x": geometry.center_x,
            "center_y": geometry.center_y,
            "radius_x": geometry.radius_x,
            "radius_y": geometry.radius_y,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _polygon_json(geometry: PolygonGeometry) -> str:
    return json.dumps(
        {
            "vertices": [[point.x, point.y] for point in geometry.vertices],
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _normalize_polygon(
    vertices: Sequence[tuple[int, int] | SourcePoint],
) -> tuple[SourcePoint, ...]:
    try:
        values = tuple(vertices)
    except TypeError as error:
        raise ValueError("vertices must be a sequence") from error
    if len(values) < 3:
        raise ValueError("polygon requires at least three vertices")
    normalized = []
    for index, value in enumerate(values):
        if isinstance(value, SourcePoint):
            x, y = value.x, value.y
        else:
            try:
                if len(value) != 2:
                    raise ValueError
                x, y = value
            except (TypeError, ValueError) as error:
                raise ValueError(f"vertex {index} must contain exactly two coordinates") from error
        _integer("vertex x", x)
        _integer("vertex y", y)
        normalized.append(SourcePoint(x, y))
    normalized_tuple = tuple(normalized)
    if len(set(normalized_tuple)) != len(normalized_tuple):
        raise ValueError("polygon vertices must be unique and not explicitly closed")
    if _shoelace_twice(normalized_tuple) == 0:
        raise ValueError("polygon area must be nonzero")
    _validate_simple_polygon(normalized_tuple)
    return normalized_tuple


def _validate_polygon_bounds(geometry: PolygonGeometry, width: int, height: int) -> None:
    if any(
        point.x < 0 or point.x > width or point.y < 0 or point.y > height
        for point in geometry.vertices
    ):
        raise ValueError("polygon must lie within the source image")


def _shoelace_twice(vertices: tuple[SourcePoint, ...]) -> int:
    return sum(
        current.x * following.y - following.x * current.y
        for current, following in zip(vertices, vertices[1:] + vertices[:1])
    )


def _cross(a: SourcePoint, b: SourcePoint, c: SourcePoint) -> int:
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def _on_segment(a: SourcePoint, b: SourcePoint, point: SourcePoint) -> bool:
    return (
        _cross(a, b, point) == 0
        and min(a.x, b.x) <= point.x <= max(a.x, b.x)
        and min(a.y, b.y) <= point.y <= max(a.y, b.y)
    )


def _segments_intersect(
    first_start: SourcePoint,
    first_end: SourcePoint,
    second_start: SourcePoint,
    second_end: SourcePoint,
) -> bool:
    orientations = (
        _cross(first_start, first_end, second_start),
        _cross(first_start, first_end, second_end),
        _cross(second_start, second_end, first_start),
        _cross(second_start, second_end, first_end),
    )
    if (
        (orientations[0] > 0 and orientations[1] < 0)
        or (orientations[0] < 0 and orientations[1] > 0)
    ) and (
        (orientations[2] > 0 and orientations[3] < 0)
        or (orientations[2] < 0 and orientations[3] > 0)
    ):
        return True
    return (
        (orientations[0] == 0 and _on_segment(first_start, first_end, second_start))
        or (orientations[1] == 0 and _on_segment(first_start, first_end, second_end))
        or (orientations[2] == 0 and _on_segment(second_start, second_end, first_start))
        or (orientations[3] == 0 and _on_segment(second_start, second_end, first_end))
    )


def _validate_simple_polygon(vertices: tuple[SourcePoint, ...]) -> None:
    count = len(vertices)
    segments = tuple(
        (vertices[index], vertices[(index + 1) % count]) for index in range(count)
    )
    for first_index, (first_start, first_end) in enumerate(segments):
        for second_index in range(first_index + 1, count):
            second_start, second_end = segments[second_index]
            if not _segments_intersect(first_start, first_end, second_start, second_end):
                continue
            adjacent = (
                second_index == first_index + 1
                or (first_index == 0 and second_index == count - 1)
            )
            if not adjacent:
                raise ValueError("polygon edges must not intersect")
            shared = set((first_start, first_end)).intersection((second_start, second_end))
            if len(shared) != 1:
                raise ValueError("adjacent polygon edges may share only one endpoint")
            common = next(iter(shared))
            if _cross(first_start, first_end, second_start) == 0 and _cross(
                first_start, first_end, second_end
            ) == 0:
                first_vector = (first_end.x - first_start.x, first_end.y - first_start.y)
                second_vector = (second_end.x - second_start.x, second_end.y - second_start.y)
                if first_vector[0] * second_vector[0] + first_vector[1] * second_vector[1] <= 0:
                    raise ValueError("adjacent polygon edges may not backtrack or overlap")
            if common not in (first_start, first_end) or common not in (second_start, second_end):
                raise ValueError("adjacent polygon edges must share their endpoint")


def _identifier(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("image_asset_id must be a non-empty string")


def _integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _stored_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("geometry values must be integers")
    return value
