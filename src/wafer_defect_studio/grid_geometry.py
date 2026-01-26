"""Source-image geometry for non-overlapping annotation grids."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnnotationGrid:
    """One full-size cell in the source-image grid lattice."""

    row: int
    column: int
    x: int
    y: int
    width: int
    height: int

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


def annotation_grids(
    image_width: int,
    image_height: int,
    cell_width: int,
    cell_height: int,
    origin_x: int = 0,
    origin_y: int = 0,
) -> tuple[AnnotationGrid, ...]:
    """Generate every full cell intersecting the source-image rectangle."""

    _positive_integer("image_width", image_width)
    _positive_integer("image_height", image_height)
    _positive_integer("cell_width", cell_width)
    _positive_integer("cell_height", cell_height)
    _canonical_origin("origin_x", origin_x, cell_width)
    _canonical_origin("origin_y", origin_y, cell_height)

    first_column = -1 if origin_x else 0
    last_column = (image_width - 1 - origin_x) // cell_width
    first_row = -1 if origin_y else 0
    last_row = (image_height - 1 - origin_y) // cell_height

    grids = []
    for row in range(first_row, last_row + 1):
        y = origin_y + row * cell_height
        for column in range(first_column, last_column + 1):
            x = origin_x + column * cell_width
            grids.append(AnnotationGrid(row, column, x, y, cell_width, cell_height))
    return tuple(grids)


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _canonical_origin(name: str, value: object, cell_size: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < cell_size:
        raise ValueError(f"{name} must be a canonical integer origin")
