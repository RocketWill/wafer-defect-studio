"""Approximate class activation maps in native source-image coordinates."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from .confidence_stitching import stitch_window_scores
from .detection_windows import Window


_DISCLAIMERS = ("Approximate localization", "not a segmentation mask")


@dataclass(frozen=True, slots=True)
class CamDetectionArtifact:
    """One immutable-in-use CAM result and its replay metadata.

    ``maps`` is a native source-space array with shape ``(H, W, C)``.  An
    uncovered source pixel is represented by ``NaN`` in ``maps`` and zero in
    ``coverage``; JSON uses ``null`` for that value.
    """

    maps: np.ndarray
    coverage: np.ndarray
    class_names: tuple[str, ...]
    source_width: int
    source_height: int
    source_transform: dict[str, Any]
    window_settings: dict[str, Any]
    provenance: dict[str, Any]
    disclaimers: tuple[str, ...] = _DISCLAIMERS

    @property
    def cam(self) -> np.ndarray:
        """Alias for callers that use the CAM name for the stitched maps."""

        return self.maps

    @property
    def confidence(self) -> np.ndarray:
        """Alias matching :class:`StitchedConfidenceMap`."""

        return self.maps

    def class_map(self, class_name: str | int) -> np.ndarray:
        """Return one class channel without changing class identity."""

        if isinstance(class_name, Integral) and not isinstance(class_name, bool):
            index = int(class_name)
            if index < 0 or index >= len(self.class_names):
                raise KeyError(class_name)
        elif isinstance(class_name, str):
            try:
                index = self.class_names.index(class_name)
            except ValueError as error:
                raise KeyError(class_name) from error
        else:
            raise KeyError(class_name)
        return self.maps[:, :, index]

    map_for_class = class_map

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible map, transform, and provenance payload."""

        return {
            "artifact_type": "approximate_cam",
            "map_shape": list(self.maps.shape),
            "class_names": list(self.class_names),
            "maps": _json_safe(self.maps),
            "coverage": _json_safe(self.coverage),
            "source": {
                "width": self.source_width,
                "height": self.source_height,
                "coordinate_system": "source-image pixels",
            },
            "source_transform": _json_safe(self.source_transform),
            "window_settings": _json_safe(self.window_settings),
            "provenance": _json_safe(self.provenance),
            "disclaimers": list(self.disclaimers),
        }

    def to_json(self) -> str:
        """Serialize deterministically for artifact storage or transport."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def write_json(self, path: str | Path) -> Path:
        """Write the deterministic JSON artifact and return its path."""

        destination = Path(path)
        destination.write_text(self.to_json() + "\n", encoding="utf-8")
        return destination


def generate_cam_artifact(
    windows: Sequence[Window],
    activations: Any,
    classifier_weights: Any,
    *,
    class_names: Sequence[str] | None = None,
    source_width: int | None = None,
    source_height: int | None = None,
    window_settings: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    model_id: str | None = None,
    profile_id: str | None = None,
    evaluation_id: str | None = None,
) -> CamDetectionArtifact:
    """Generate normalized per-window CAMs and stitch them in source space.

    Activations use ``(window, feature, height, width)`` and classifier
    weights use ``(class, feature)``.  Positive CAMs are normalized per window
    and class before the existing deterministic center-weighted stitcher is
    applied.  This is intentionally an approximate localization signal, not a
    pixel mask or a Grad-CAM implementation.
    """

    resolved_windows = tuple(windows)
    if not resolved_windows or any(not isinstance(window, Window) for window in resolved_windows):
        raise ValueError("windows must contain at least one Window value")
    width = resolved_windows[0].source_width if source_width is None else source_width
    height = resolved_windows[0].source_height if source_height is None else source_height
    _positive_integer("source_width", width)
    _positive_integer("source_height", height)
    for window in resolved_windows:
        if window.source_width != width or window.source_height != height:
            raise ValueError("window source dimensions must match the requested map")

    feature_maps = _activation_array(activations)
    if feature_maps.ndim == 3:
        if len(resolved_windows) != 1:
            raise ValueError("3-D activations are only valid for one window")
        feature_maps = feature_maps[None, ...]
    if feature_maps.ndim != 4 or feature_maps.shape[0] != len(resolved_windows):
        raise ValueError("activations must have shape (windows, features, height, width)")
    window_height, window_width = resolved_windows[0].height, resolved_windows[0].width
    if feature_maps.shape[2:] != (window_height, window_width):
        raise ValueError("activation spatial dimensions must match the model window")
    if any((window.height, window.width) != (window_height, window_width) for window in resolved_windows):
        raise ValueError("all windows must have equal model dimensions")

    weights = _activation_array(classifier_weights)
    if weights.ndim != 2:
        raise ValueError("classifier_weights must have shape (classes, features)")
    if weights.shape[1] == feature_maps.shape[1]:
        class_count = int(weights.shape[0])
    elif weights.shape[0] == feature_maps.shape[1]:
        weights = weights.T
        class_count = int(weights.shape[0])
    else:
        raise ValueError("classifier_weights feature axis must match activations")
    names = _class_names(class_names, class_count)
    if not np.all(np.isfinite(feature_maps)) or not np.all(np.isfinite(weights)):
        raise ValueError("activations and classifier_weights must be finite")

    # CAM is the positive weighted activation sum. Normalizing each window and
    # class keeps the artifact in the stitcher's confidence range [0, 1].
    local_maps = np.einsum("nfhw,cf->nhwc", feature_maps, weights, optimize=True)
    local_maps = np.maximum(local_maps, 0.0)
    maxima = np.max(local_maps, axis=(1, 2), keepdims=True)
    local_maps = np.divide(local_maps, maxima, out=np.zeros_like(local_maps), where=maxima > 0.0)

    stitched = stitch_window_scores(
        resolved_windows,
        local_maps,
        source_width=width,
        source_height=height,
        class_count=len(names),
    )
    metadata = _provenance(provenance, model_id, profile_id, evaluation_id)
    settings = _window_settings(window_settings, resolved_windows)
    transform = _source_transform(resolved_windows)
    return CamDetectionArtifact(
        maps=stitched.confidence.copy(),
        coverage=stitched.coverage.copy(),
        class_names=names,
        source_width=int(width),
        source_height=int(height),
        source_transform=transform,
        window_settings=settings,
        provenance=metadata,
    )


generate_approximate_cam_artifact = generate_cam_artifact


def generate_all_convolutional_artifact(
    windows: Sequence[Window],
    local_probabilities: Any,
    *,
    class_names: Sequence[str] | None = None,
    source_width: int | None = None,
    source_height: int | None = None,
    window_settings: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    model_id: str | None = None,
    profile_id: str | None = None,
    evaluation_id: str | None = None,
) -> CamDetectionArtifact:
    """Stitch trained sigmoid maps or scalar patch scores without renormalizing."""

    resolved_windows = tuple(windows)
    if not resolved_windows or any(not isinstance(window, Window) for window in resolved_windows):
        raise ValueError("windows must contain at least one Window value")
    width = resolved_windows[0].source_width if source_width is None else source_width
    height = resolved_windows[0].source_height if source_height is None else source_height
    _positive_integer("source_width", width)
    _positive_integer("source_height", height)
    values = _activation_array(local_probabilities)
    scalar_patch_scores = values.ndim == 2
    if values.ndim == 3:
        values = values[None, ...]
    if values.ndim not in (2, 4) or values.shape[0] != len(resolved_windows):
        raise ValueError(
            "local_probabilities must have shape (windows, classes) or "
            "(windows, height, width, classes)"
        )
    if not scalar_patch_scores:
        window_height, window_width = resolved_windows[0].height, resolved_windows[0].width
        if values.shape[1:3] != (window_height, window_width):
            raise ValueError("local_probabilities spatial dimensions must match the model window")
        if any((window.height, window.width) != (window_height, window_width) for window in resolved_windows):
            raise ValueError("all windows must have equal model dimensions")
    if not np.all(np.isfinite(values)) or not np.all((values >= 0.0) & (values <= 1.0)):
        raise ValueError("local_probabilities must contain finite sigmoid values from 0 to 1")
    names = _class_names(class_names, int(values.shape[-1]))
    stitched = stitch_window_scores(
        resolved_windows,
        values,
        source_width=width,
        source_height=height,
        class_count=len(names),
        center_weight=str((window_settings or {}).get("center_weighting", "linear")),
    )
    metadata = _provenance(provenance, model_id, profile_id, evaluation_id)
    map_method = (
        "patch_classification_sigmoid"
        if scalar_patch_scores
        else "all_convolutional_sigmoid"
    )
    metadata["map_method"] = map_method
    settings = _window_settings(window_settings, resolved_windows)
    settings["map_method"] = map_method
    return CamDetectionArtifact(
        maps=stitched.confidence.copy(),
        coverage=stitched.coverage.copy(),
        class_names=names,
        source_width=int(width),
        source_height=int(height),
        source_transform=_source_transform(resolved_windows),
        window_settings=settings,
        provenance=metadata,
    )


def _activation_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("activation values must be numeric") from error


def _class_names(value: Sequence[str] | None, class_count: int) -> tuple[str, ...]:
    if value is None:
        return tuple(f"class_{index}" for index in range(class_count))
    if isinstance(value, (str, bytes)):
        raise ValueError("class_names must contain one non-empty string per class")
    names = tuple(value)
    if len(names) != class_count or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("class_names must contain one non-empty string per class")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique")
    return names


def _provenance(
    value: Mapping[str, Any] | None,
    model_id: str | None,
    profile_id: str | None,
    evaluation_id: str | None,
) -> dict[str, Any]:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("provenance must be a mapping")
    result = dict(value or {})
    for key, explicit in (
        ("model_id", model_id),
        ("profile_id", profile_id),
        ("evaluation_id", evaluation_id),
    ):
        if explicit is not None:
            result[key] = explicit
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f"provenance must include non-empty {key}")
    return result


def _window_settings(value: Mapping[str, Any] | None, windows: tuple[Window, ...]) -> dict[str, Any]:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("window_settings must be a mapping")
    result = dict(value or {})
    result.setdefault("window_size", [windows[0].width, windows[0].height])
    result.setdefault("padding", "reflect")
    result.setdefault("window_count", len(windows))
    return result


def _source_transform(windows: tuple[Window, ...]) -> dict[str, Any]:
    return {
        "coordinate_system": "source-image pixels",
        "padding": "reflect",
        "windows": [
            {
                "index": window.index,
                "source_rect": _rect(window.source_rect),
                "read_rect": _rect(window.read_rect),
                "padding": {
                    "left": window.padding.left,
                    "top": window.padding.top,
                    "right": window.padding.right,
                    "bottom": window.padding.bottom,
                },
                "mapping": "source_to_window canonical; window_to_source reflect",
            }
            for window in windows
        ],
    }


def _rect(rect: Any) -> dict[str, int]:
    return {"x": int(rect.x), "y": int(rect.y), "width": int(rect.width), "height": int(rect.height)}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _positive_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "CamDetectionArtifact",
    "generate_approximate_cam_artifact",
    "generate_all_convolutional_artifact",
    "generate_cam_artifact",
]
