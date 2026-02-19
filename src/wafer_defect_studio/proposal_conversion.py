"""Pure preview planning for converting reviewed proposals to Grid Annotations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .detection_windows import Rect
from .grid_geometry import AnnotationGrid
from .proposal_generation import DefectProposal
from .proposal_queue import ReviewQueueItem


@dataclass(frozen=True, slots=True)
class ConversionCell:
    """One participating Annotation Grid affected by selected proposals."""

    grid: AnnotationGrid
    class_codes: tuple[str, ...]
    proposal_ids: tuple[str, ...]

    @property
    def annotation_grid(self) -> AnnotationGrid:
        """Explicit Annotation Grid spelling for conversion consumers."""

        return self.grid

    @property
    def row(self) -> int:
        return self.grid.row

    @property
    def column(self) -> int:
        return self.grid.column

    @property
    def classes(self) -> tuple[str, ...]:
        """Short alias for the aggregated Defect Class codes."""

        return self.class_codes

    @property
    def source_proposal_ids(self) -> tuple[str, ...]:
        """Source Defect Proposal identities contributing to this cell."""

        return self.proposal_ids


@dataclass(frozen=True, slots=True)
class ConversionPreview:
    """Immutable source-coordinate conversion plan with proposal provenance."""

    cells: tuple[ConversionCell, ...]
    source_proposal_ids: tuple[str, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.cells, tuple) or not all(
            isinstance(cell, ConversionCell) for cell in self.cells
        ):
            raise ValueError("cells must be a tuple of ConversionCell values")
        if not isinstance(self.source_proposal_ids, tuple) or any(
            not isinstance(value, str) or not value for value in self.source_proposal_ids
        ):
            raise ValueError("source_proposal_ids must be a tuple of non-empty strings")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def affected_cells(self) -> tuple[ConversionCell, ...]:
        """Readable alias for the cells that receive one or more classes."""

        return self.cells

    @property
    def grid_cells(self) -> tuple[ConversionCell, ...]:
        """Alias used by callers that refer to preview rows as grid cells."""

        return self.cells

    @property
    def proposal_ids(self) -> tuple[str, ...]:
        """Short alias for selected source proposal identities."""

        return self.source_proposal_ids

    @property
    def source_ids(self) -> tuple[str, ...]:
        """Alias for selected source proposal identities."""

        return self.source_proposal_ids

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-friendly preview representation."""

        return {
            "cells": tuple(
                {
                    "row": cell.row,
                    "column": cell.column,
                    "x": cell.grid.x,
                    "y": cell.grid.y,
                    "width": cell.grid.width,
                    "height": cell.grid.height,
                    "class_codes": cell.class_codes,
                    "proposal_ids": cell.proposal_ids,
                }
                for cell in self.cells
            ),
            "source_proposal_ids": self.source_proposal_ids,
            "provenance": dict(self.provenance),
        }


def preview_proposal_conversion(
    proposals: tuple[DefectProposal | ReviewQueueItem, ...],
    participating_grids: tuple[AnnotationGrid, ...],
) -> ConversionPreview:
    """Plan reviewed proposal geometry as multi-label Grid Annotation changes.

    ``participating_grids`` is explicit: callers must derive Effective Wafer
    Area participation before invoking this pure function.  Raw
    :class:`DefectProposal` values are treated as explicitly selected;
    :class:`ReviewQueueItem` values must be ``accepted`` or ``corrected`` and
    contribute their current (possibly corrected) source rectangle.
    """

    grids = _validate_grids(participating_grids)
    entries = _validate_proposals(proposals)
    source_ids = tuple(sorted(entry[0] for entry in entries))

    cells: list[ConversionCell] = []
    for grid in sorted(grids, key=lambda value: (value.row, value.column)):
        matches = tuple(entry for entry in entries if _rects_overlap(entry[1], _grid_rect(grid)))
        if not matches:
            continue
        class_codes = tuple(sorted({entry[2] for entry in matches}))
        proposal_ids = tuple(sorted({entry[0] for entry in matches}))
        cells.append(ConversionCell(grid, class_codes, proposal_ids))

    provenance: dict[str, Any] = {
        "source_coordinate_system": "source-image-pixels",
    }
    for key in ("run_id", "profile_id"):
        values = tuple(
            sorted(
                {
                    value
                    for _, _, _, value in entries
                    if isinstance(value, Mapping)
                    for value in (value.get(key),)
                    if isinstance(value, str) and value
                }
            )
        )
        if len(values) == 1:
            provenance[key] = values[0]
        elif values:
            provenance[f"{key}s"] = values

    return ConversionPreview(tuple(cells), source_ids, provenance)


def plan_proposal_conversion(
    proposals: tuple[DefectProposal | ReviewQueueItem, ...],
    participating_grids: tuple[AnnotationGrid, ...],
) -> ConversionPreview:
    """Explicit planning alias for :func:`preview_proposal_conversion`."""

    return preview_proposal_conversion(proposals, participating_grids)


def _validate_grids(
    grids: tuple[AnnotationGrid, ...],
) -> tuple[AnnotationGrid, ...]:
    if not isinstance(grids, tuple) or not grids:
        raise ValueError("participating_grids must be a non-empty tuple")
    seen: set[tuple[int, int]] = set()
    for grid in grids:
        if not isinstance(grid, AnnotationGrid):
            raise TypeError("participating_grids must contain AnnotationGrid values")
        _integer(grid.row, "grid.row")
        _integer(grid.column, "grid.column")
        _integer(grid.x, "grid.x")
        _integer(grid.y, "grid.y")
        _positive_integer(grid.width, "grid.width")
        _positive_integer(grid.height, "grid.height")
        identity = (grid.row, grid.column)
        if identity in seen:
            raise ValueError("participating_grids must not contain duplicate row/column values")
        seen.add(identity)
    return grids


def _validate_proposals(
    proposals: tuple[DefectProposal | ReviewQueueItem, ...],
) -> tuple[tuple[str, Rect, str, Mapping[str, Any]], ...]:
    if not isinstance(proposals, tuple) or not proposals:
        raise ValueError("proposals must be a non-empty tuple")
    entries: list[tuple[str, Rect, str, Mapping[str, Any]]] = []
    seen: set[str] = set()
    for item in proposals:
        if isinstance(item, DefectProposal):
            proposal = item
            rect = item.source_rect
        elif isinstance(item, ReviewQueueItem):
            status = getattr(item.status, "value", item.status)
            if not isinstance(status, str) or status.lower() not in {"accepted", "corrected"}:
                raise ValueError("only accepted or corrected ReviewQueueItems may be converted")
            if not isinstance(item.proposal, DefectProposal):
                raise TypeError("ReviewQueueItem must contain a DefectProposal")
            proposal = item.proposal
            rect = item.source_rect
        else:
            raise TypeError("proposals must contain DefectProposal or ReviewQueueItem values")

        if not isinstance(proposal.proposal_id, str) or not proposal.proposal_id:
            raise ValueError("proposal_id must be a non-empty string")
        if proposal.proposal_id in seen:
            raise ValueError(f"duplicate proposal_id: {proposal.proposal_id}")
        seen.add(proposal.proposal_id)
        if not isinstance(proposal.class_name, str) or not proposal.class_name.strip():
            raise ValueError("class_name must be a non-empty string")
        _validate_rect(rect, "proposal source_rect")
        provenance = proposal.provenance
        if not isinstance(provenance, Mapping):
            raise ValueError("proposal provenance must be a mapping")
        entries.append((proposal.proposal_id, rect, proposal.class_name, provenance))
    return tuple(entries)


def _grid_rect(grid: AnnotationGrid) -> Rect:
    return Rect(grid.x, grid.y, grid.width, grid.height)


def _rects_overlap(left: Rect, right: Rect) -> bool:
    return (
        left.x < right.x + right.width
        and right.x < left.x + left.width
        and left.y < right.y + right.height
        and right.y < left.y + left.height
    )


def _validate_rect(rect: object, name: str) -> None:
    if not isinstance(rect, Rect):
        raise TypeError(f"{name} must be a Rect")
    _integer(rect.x, f"{name}.x")
    _integer(rect.y, f"{name}.y")
    _positive_integer(rect.width, f"{name}.width")
    _positive_integer(rect.height, f"{name}.height")


def _integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "ConversionCell",
    "ConversionPreview",
    "preview_proposal_conversion",
    "plan_proposal_conversion",
]
