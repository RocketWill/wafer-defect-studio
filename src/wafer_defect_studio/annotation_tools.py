"""Pure state and geometry helpers for Annotation Grid tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import ceil, floor
from typing import Hashable, Iterable, Mapping


class AnnotationMode(str, Enum):
    """The four visible tools offered by the annotation canvas."""

    PAN = "Pan"
    ANNOTATE = "Annotate"
    PAINT = "Paint"
    ERASE = "Erase"


# Keep the short keys intentionally stable: these are part of the canvas workflow.
FIXED_MODE_KEYS: Mapping[str, AnnotationMode] = {
    "P": AnnotationMode.PAN,
    "A": AnnotationMode.ANNOTATE,
    "B": AnnotationMode.PAINT,
    "E": AnnotationMode.ERASE,
}


def crossed_annotation_cells(
    start: tuple[float, float],
    end: tuple[float, float],
    cell_width: int,
    cell_height: int,
    origin_x: int = 0,
    origin_y: int = 0,
) -> tuple[tuple[int, int], ...]:
    """Return source-aligned cells crossed by a pointer segment once each."""

    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("cell dimensions must be positive")
    x0, y0 = start
    x1, y1 = end
    steps = max(1, ceil(max(abs(x1 - x0), abs(y1 - y0))))
    cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for index in range(steps + 1):
        fraction = index / steps
        x = x0 + (x1 - x0) * fraction
        y = y0 + (y1 - y0) * fraction
        cell = (
            floor((y - origin_y) / cell_height),
            floor((x - origin_x) / cell_width),
        )
        if cell not in seen:
            seen.add(cell)
            cells.append(cell)
    return tuple(cells)


def apply_class_set(
    existing_codes: Iterable[str],
    selected_codes: Iterable[str],
    *,
    erase: bool = False,
) -> tuple[str, ...]:
    """Replace a cell's complete set, or remove only the selected codes."""

    existing = tuple(dict.fromkeys(existing_codes))
    selected = tuple(dict.fromkeys(selected_codes))
    if erase:
        selected_set = set(selected)
        return tuple(code for code in existing if code not in selected_set)
    return selected


def deduplicate_cells(cells: Iterable[Hashable]) -> tuple[Hashable, ...]:
    """Keep crossed cells in first-seen order, including one paint per cell."""

    unique: list[Hashable] = []
    seen: set[Hashable] = set()
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            unique.append(cell)
    return tuple(unique)


@dataclass
class AnnotationToolState:
    """Mutable UI-independent state for tool modes and selected Defect Classes."""

    mode: AnnotationMode = AnnotationMode.PAN
    selected_class_codes: tuple[str, ...] = ()
    _class_keys: dict[str, str] = field(default_factory=dict, repr=False)

    def set_mode(self, mode: AnnotationMode | str) -> AnnotationMode:
        try:
            if isinstance(mode, AnnotationMode):
                selected = mode
            elif isinstance(mode, str):
                normalized = mode.strip().lower()
                selected = next(
                    candidate
                    for candidate in AnnotationMode
                    if candidate.value.lower() == normalized or candidate.name.lower() == normalized
                )
            else:
                selected = AnnotationMode(mode)
            self.mode = selected
        except (StopIteration, TypeError, ValueError) as error:
            raise ValueError(f"unknown annotation mode: {mode!r}") from error
        return self.mode

    def set_selected_classes(self, codes: Iterable[str]) -> tuple[str, ...]:
        self.selected_class_codes = tuple(dict.fromkeys(codes))
        return self.selected_class_codes

    def toggle_class(self, code: str) -> tuple[str, ...]:
        selected = list(self.selected_class_codes)
        if code in selected:
            selected.remove(code)
        else:
            selected.append(code)
        self.selected_class_codes = tuple(selected)
        return self.selected_class_codes

    def set_class_key(self, code: str, key: str) -> None:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code must be a non-empty string")
        if not isinstance(key, str) or len(key.strip()) != 1:
            raise ValueError("class key must be one character")
        normalized = key.strip().upper()
        if normalized in FIXED_MODE_KEYS:
            raise ValueError(f"class key conflicts with fixed shortcut: {normalized}")
        for existing_code, existing_key in self._class_keys.items():
            if existing_code != code and existing_key == normalized:
                raise ValueError(f"class key already assigned: {normalized}")
        self._class_keys[code] = normalized

    def handle_shortcut(self, key: str) -> AnnotationMode | tuple[str, ...] | None:
        if not isinstance(key, str) or not key:
            return None
        normalized = key.upper()[:1]
        mode = FIXED_MODE_KEYS.get(normalized)
        if mode is not None:
            return self.set_mode(mode)
        for code, class_key in self._class_keys.items():
            if class_key == normalized:
                return self.toggle_class(code)
        return None

    def apply_cells(
        self,
        cells: Iterable[Hashable],
        annotations: Mapping[Hashable, Iterable[str]],
    ) -> dict[Hashable, tuple[str, ...]]:
        """Apply selected labels to each crossed cell exactly once."""

        updated = {cell: tuple(codes) for cell, codes in annotations.items()}
        if self.mode is AnnotationMode.PAN:
            return updated
        for cell in deduplicate_cells(cells):
            updated[cell] = apply_class_set(
                updated.get(cell, ()),
                self.selected_class_codes,
                erase=self.mode is AnnotationMode.ERASE,
            )
        return updated


# A concise alias is useful to callers that refer to the state as a tool model.
ToolMode = AnnotationMode
