"""Deterministic connected-component proposals from native confidence maps."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from .cam_detection import CamDetectionArtifact
from .detection_windows import Rect


@dataclass(frozen=True, slots=True)
class DefectProposal:
    """One approximate, source-coordinate region suggested by a confidence map."""

    proposal_id: str
    class_name: str
    source_rect: Rect
    area: int
    peak_confidence: float
    mean_confidence: float
    provenance: dict[str, Any]

    @property
    def geometry(self) -> Rect:
        """Source-image ``x, y, width, height`` geometry."""

        return self.source_rect

    @property
    def source_image_rect(self) -> Rect:
        """Explicit alias for consumers displaying source-image geometry."""

        return self.source_rect

    @property
    def rect(self) -> Rect:
        """Short geometry alias used by map/review callers."""

        return self.source_rect

    @property
    def peak(self) -> float:
        """Short alias for peak confidence."""

        return self.peak_confidence

    @property
    def mean(self) -> float:
        """Short alias for mean confidence."""

        return self.mean_confidence

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible proposal representation."""

        return {
            "proposal_id": self.proposal_id,
            "class_name": self.class_name,
            "source_rect": {
                "x": self.source_rect.x,
                "y": self.source_rect.y,
                "width": self.source_rect.width,
                "height": self.source_rect.height,
            },
            "area": self.area,
            "peak_confidence": self.peak_confidence,
            "mean_confidence": self.mean_confidence,
            "provenance": _json_safe(self.provenance),
        }


def generate_proposals(
    artifact: CamDetectionArtifact | np.ndarray,
    settings: Mapping[str, Any] | Any | None = None,
    *,
    thresholds: Mapping[str, Real] | Sequence[Real] | Real | None = None,
    map_generation: Mapping[str, Any] | None = None,
    class_names: Sequence[str] | None = None,
    provenance: Mapping[str, Any] | None = None,
    closing_radius: int | None = None,
    minimum_area: int | None = None,
) -> tuple[DefectProposal, ...]:
    """Generate deterministic per-class proposals from a native HxWxC map.

    ``settings`` may be a Detection Profile or a mapping containing
    ``thresholds`` and ``map_generation``.  A bare mapping of class names to
    thresholds is also accepted.  Components are processed independently for
    each class, so regions from different classes may overlap.
    """

    maps, names, source_width, source_height, artifact_provenance = _map_inputs(
        artifact, class_names
    )
    resolved_thresholds, resolved_generation = _settings(
        settings,
        thresholds=thresholds,
        map_generation=map_generation,
        class_count=len(names),
    )
    threshold_values = _threshold_values(resolved_thresholds, names)
    radius = _closing_radius(
        resolved_generation.get("closing_radius", 0)
        if closing_radius is None
        else closing_radius
    )
    area_limit = _minimum_area(
        resolved_generation.get("minimum_area", 1)
        if minimum_area is None
        else minimum_area
    )

    base_provenance = dict(artifact_provenance)
    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        base_provenance.update(provenance)
    base_provenance.setdefault("source_coordinate_system", "source-image-pixels")

    result: list[DefectProposal] = []
    for class_index, (class_name, threshold) in enumerate(zip(names, threshold_values)):
        values = maps[:, :, class_index]
        active = np.isfinite(values) & (values >= threshold)
        closed = _close(active, radius)
        components = _components(closed)
        for component in components:
            if len(component) < area_limit:
                continue
            ys = np.fromiter((point[0] for point in component), dtype=np.intp)
            xs = np.fromiter((point[1] for point in component), dtype=np.intp)
            confidence = values[ys, xs]
            finite_confidence = confidence[np.isfinite(confidence)]
            if finite_confidence.size == 0:
                continue
            rect = Rect(
                int(xs.min()),
                int(ys.min()),
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1),
            )
            proposal_provenance = {
                **base_provenance,
                "class_name": class_name,
                "class_index": class_index,
                "threshold": threshold,
                "closing_radius": radius,
                "minimum_area": area_limit,
                "source_width": source_width,
                "source_height": source_height,
            }
            peak = float(np.max(finite_confidence))
            mean = float(np.mean(finite_confidence))
            proposal_id = _proposal_id(class_name, rect, len(component), peak, mean, proposal_provenance)
            result.append(
                DefectProposal(
                    proposal_id=proposal_id,
                    class_name=class_name,
                    source_rect=rect,
                    area=len(component),
                    peak_confidence=peak,
                    mean_confidence=mean,
                    provenance=proposal_provenance,
                )
            )
    return tuple(result)


generate_defect_proposals = generate_proposals


def _map_inputs(
    artifact: CamDetectionArtifact | np.ndarray,
    class_names: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...], int, int, dict[str, Any]]:
    if isinstance(artifact, CamDetectionArtifact):
        maps = np.asarray(artifact.maps, dtype=np.float64)
        names = tuple(artifact.class_names)
        width = artifact.source_width
        height = artifact.source_height
        artifact_provenance = dict(artifact.provenance)
    else:
        try:
            maps = np.asarray(artifact, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("confidence map must be numeric") from error
        if class_names is None:
            raise ValueError("class_names are required for a raw confidence map")
        names = tuple(class_names)
        width = int(maps.shape[1]) if maps.ndim >= 2 else 0
        height = int(maps.shape[0]) if maps.ndim >= 1 else 0
        artifact_provenance = {}
    if maps.ndim != 3:
        raise ValueError("confidence map must have shape (height, width, class_count)")
    if maps.shape[2] != len(names) or not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("class_names must match the confidence map class axis")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique")
    if width != maps.shape[1] or height != maps.shape[0] or width <= 0 or height <= 0:
        raise ValueError("confidence map dimensions must match the source image")
    if not np.all(np.isnan(maps) | ((maps >= 0.0) & (maps <= 1.0))):
        raise ValueError("confidence map values must be NaN or in the range 0 to 1")
    return maps, names, width, height, artifact_provenance


def _settings(
    settings: Any,
    *,
    thresholds: Mapping[str, Real] | Sequence[Real] | Real | None,
    map_generation: Mapping[str, Any] | None,
    class_count: int,
) -> tuple[Mapping[str, Real] | Sequence[Real] | Real, dict[str, Any]]:
    resolved_thresholds = thresholds
    resolved_generation = dict(map_generation or {})
    if settings is not None:
        if isinstance(settings, Mapping):
            if "thresholds" in settings or "class_thresholds" in settings:
                if resolved_thresholds is None:
                    resolved_thresholds = settings.get("thresholds", settings.get("class_thresholds"))
            if "map_generation" in settings or "map_settings" in settings:
                if map_generation is None:
                    value = settings.get("map_generation", settings.get("map_settings"))
                    if value is not None:
                        if not isinstance(value, Mapping):
                            raise ValueError("map_generation must be a mapping")
                        resolved_generation = dict(value)
            elif resolved_thresholds is None:
                resolved_thresholds = settings  # bare class-to-threshold mapping
        else:
            if resolved_thresholds is None:
                resolved_thresholds = getattr(settings, "thresholds", getattr(settings, "class_thresholds", None))
            if map_generation is None:
                value = getattr(settings, "map_generation", getattr(settings, "map_settings", None))
                if value is not None:
                    if not isinstance(value, Mapping):
                        raise ValueError("map_generation must be a mapping")
                    resolved_generation = dict(value)
    if resolved_thresholds is None:
        raise ValueError("per-class thresholds are required")
    if len(resolved_generation) > 0 and not isinstance(resolved_generation, dict):
        resolved_generation = dict(resolved_generation)
    return resolved_thresholds, resolved_generation


def _threshold_values(
    thresholds: Mapping[str, Real] | Sequence[Real] | Real,
    names: tuple[str, ...],
) -> tuple[float, ...]:
    if isinstance(thresholds, Mapping):
        if any(name not in thresholds for name in names):
            missing = next(name for name in names if name not in thresholds)
            raise ValueError(f"threshold missing for class: {missing}")
        values = tuple(thresholds[name] for name in names)
    elif isinstance(thresholds, Real) and not isinstance(thresholds, bool):
        values = (thresholds,) * len(names)
    else:
        if isinstance(thresholds, (str, bytes)) or not isinstance(thresholds, Sequence):
            raise ValueError("thresholds must be a mapping, scalar, or one value per class")
        if len(thresholds) != len(names):
            raise ValueError("thresholds must contain one value per class")
        values = tuple(thresholds)
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("thresholds must contain finite values from 0 to 1")
        normalized.append(float(value))
    return tuple(normalized)


def _closing_radius(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("closing_radius must be a non-negative integer")
    return int(value)


def _minimum_area(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("minimum_area must be a positive integer")
    return int(value)


def _close(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return mask.copy()
    return _erode(_dilate(mask, radius), radius)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    height, width = mask.shape
    result = np.zeros_like(mask, dtype=bool)
    for y in range(height):
        for x in range(width):
            top = max(0, y - radius)
            bottom = min(height, y + radius + 1)
            left = max(0, x - radius)
            right = min(width, x + radius + 1)
            result[y, x] = bool(np.any(mask[top:bottom, left:right]))
    return result


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    height, width = mask.shape
    result = np.zeros_like(mask, dtype=bool)
    for y in range(height):
        for x in range(width):
            top = y - radius
            bottom = y + radius + 1
            left = x - radius
            right = x + radius + 1
            if top >= 0 and left >= 0 and bottom <= height and right <= width:
                result[y, x] = bool(np.all(mask[top:bottom, left:right]))
    return result


def _components(mask: np.ndarray) -> tuple[tuple[tuple[int, int], ...], ...]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[tuple[tuple[int, int], ...]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            points: list[tuple[int, int]] = []
            while stack:
                current_y, current_x = stack.pop()
                points.append((current_y, current_x))
                for offset_y in (-1, 0, 1):
                    for offset_x in (-1, 0, 1):
                        if offset_y == 0 and offset_x == 0:
                            continue
                        next_y = current_y + offset_y
                        next_x = current_x + offset_x
                        if (
                            0 <= next_y < height
                            and 0 <= next_x < width
                            and mask[next_y, next_x]
                            and not visited[next_y, next_x]
                        ):
                            visited[next_y, next_x] = True
                            stack.append((next_y, next_x))
            components.append(tuple(sorted(points)))
    return tuple(components)


def _proposal_id(
    class_name: str,
    rect: Rect,
    area: int,
    peak: float,
    mean: float,
    provenance: Mapping[str, Any],
) -> str:
    payload = {
        "class_name": class_name,
        "source_rect": [rect.x, rect.y, rect.width, rect.height],
        "area": area,
        "peak_confidence": peak,
        "mean_confidence": mean,
        "provenance": _json_safe(provenance),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return f"{class_name}:{digest}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("provenance values must be finite")
        return value
    return value


__all__ = ["DefectProposal", "generate_defect_proposals", "generate_proposals"]
