"""Deterministic generated-defect truth in source-image coordinates.

Support regions and Annotation Grids are closed. Boundary contact therefore
counts as intersection, including contact shared by adjacent grids.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import hypot

from wafer_defect_studio.grid_geometry import AnnotationGrid

Point = tuple[int, int]


@dataclass(frozen=True)
class Scratch:
    class_code: str
    points: tuple[Point, ...]
    radius: int


@dataclass(frozen=True)
class Particle:
    class_code: str
    center: Point
    radius: int


@dataclass(frozen=True)
class DefectOracle:
    image_width: int
    image_height: int
    defects: tuple[Scratch | Particle, ...]

    def to_json(self) -> str:
        payload = {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "defects": [
                {"kind": "scratch" if isinstance(defect, Scratch) else "particle", **asdict(defect)}
                for defect in self.defects
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> "DefectOracle":
        payload = json.loads(value)
        defects = []
        for item in payload["defects"]:
            item = dict(item)
            kind = item.pop("kind")
            if kind == "scratch":
                item["points"] = tuple(tuple(point) for point in item["points"])
                defects.append(Scratch(**item))
            elif kind == "particle":
                item["center"] = tuple(item["center"])
                defects.append(Particle(**item))
            else:
                raise ValueError(f"unknown generated defect kind: {kind}")
        return cls(payload["image_width"], payload["image_height"], tuple(defects))

    def grid_truth(self, grids: tuple[AnnotationGrid, ...]) -> dict[tuple[int, int], tuple[str, ...]]:
        return {
            (grid.row, grid.column): tuple(
                sorted({defect.class_code for defect in self.defects if _intersects(defect, grid)})
            )
            for grid in grids
        }


def _intersects(defect: Scratch | Particle, grid: AnnotationGrid) -> bool:
    left, top = grid.x, grid.y
    right, bottom = left + grid.width, top + grid.height
    if isinstance(defect, Particle):
        x, y = defect.center
        return hypot(x - min(max(x, left), right), y - min(max(y, top), bottom)) <= defect.radius
    return any(
        _segment_rectangle_distance(start, end, left, top, right, bottom) <= defect.radius
        for start, end in zip(defect.points, defect.points[1:])
    )


def _segment_rectangle_distance(
    start: Point, end: Point, left: int, top: int, right: int, bottom: int
) -> float:
    if _inside(start, left, top, right, bottom) or _inside(end, left, top, right, bottom):
        return 0.0
    edges = (
        ((left, top), (right, top)),
        ((right, top), (right, bottom)),
        ((right, bottom), (left, bottom)),
        ((left, bottom), (left, top)),
    )
    if any(_segments_intersect(start, end, *edge) for edge in edges):
        return 0.0
    return min(
        min(
            _point_segment_distance(edge[0], start, end),
            _point_segment_distance(edge[1], start, end),
            _point_segment_distance(start, *edge),
            _point_segment_distance(end, *edge),
        )
        for edge in edges
    )


def _inside(point: Point, left: int, top: int, right: int, bottom: int) -> bool:
    return left <= point[0] <= right and top <= point[1] <= bottom


def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    if dx == dy == 0:
        return hypot(point[0] - start[0], point[1] - start[1])
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy)))
    return hypot(point[0] - (start[0] + t * dx), point[1] - (start[1] + t * dy))


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    def orientation(p: Point, q: Point, r: Point) -> int:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p: Point, q: Point, r: Point) -> bool:
        return (
            orientation(p, q, r) == 0
            and min(p[0], q[0]) <= r[0] <= max(p[0], q[0])
            and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])
        )

    ab_c, ab_d = orientation(a, b, c), orientation(a, b, d)
    cd_a, cd_b = orientation(c, d, a), orientation(c, d, b)
    return (
        (ab_c > 0) != (ab_d > 0) and (cd_a > 0) != (cd_b > 0)
    ) or any((on_segment(a, b, c), on_segment(a, b, d), on_segment(c, d, a), on_segment(c, d, b)))
