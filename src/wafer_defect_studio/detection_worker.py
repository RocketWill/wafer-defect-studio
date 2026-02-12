"""Spawn-safe full-resolution CAM detection worker.

The worker receives immutable values, never opens the project database, and
publishes a filesystem stage only after its map/provenance checksums validate.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .cam_detection import CamDetectionArtifact, generate_cam_artifact
from .detection_windows import Window, enumerate_inference_windows


PROTOCOL_VERSION = 1
_STAGE_FORMAT = "cam_detection_stage_v1"
_TERMINAL_STATUSES = {"completed", "cancelled", "failed", "interrupted"}
_STAGE_FILES = ("maps.json", "provenance.json")


class DetectionWorkerError(RuntimeError):
    """Raised when a Detection worker request or stage is invalid."""


class DetectionProtocolError(ValueError):
    """Raised when a worker message is not a supported value message."""


@dataclass(frozen=True, slots=True)
class DetectionRequest:
    """Serializable values needed for one native-coordinate detection run."""

    request_id: str
    source: Any
    staging_path: str | Path
    run_id: str | None = None
    profile_id: str | None = None
    approved_run_id: str | None = None
    approved_profile_id: str | None = None
    approved_evaluation_id: str | None = None
    class_names: Sequence[str] | None = None
    window_size: int | Sequence[int] = 32
    stride: int | Sequence[int] = 16
    reflect_padding: bool = True
    center_weighting: str = "linear"
    device: str = "cpu"
    model_id: str = "synthetic-edge-v1"
    classifier_weights: Sequence[Sequence[float]] | Sequence[float] | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.request_id, "request_id")
        source = _source_array(self.source)
        object.__setattr__(self, "source", source)
        if isinstance(self.staging_path, Path):
            object.__setattr__(self, "staging_path", str(self.staging_path))
        _require_text(self.staging_path, "staging_path")

        chosen_run = _coalesce_id(self.run_id, self.approved_run_id, "run_id")
        chosen_profile = _coalesce_id(self.profile_id, self.approved_profile_id, "profile_id")
        _require_text(chosen_run, "run_id")
        _require_text(chosen_profile, "profile_id")
        object.__setattr__(self, "run_id", chosen_run)
        object.__setattr__(self, "profile_id", chosen_profile)
        object.__setattr__(self, "approved_run_id", chosen_run)
        object.__setattr__(self, "approved_profile_id", chosen_profile)
        if self.approved_evaluation_id is not None:
            _require_text(self.approved_evaluation_id, "approved_evaluation_id")

        names = ("defect",) if self.class_names is None else tuple(self.class_names)
        if not names or any(not isinstance(name, str) or not name.strip() for name in names):
            raise DetectionWorkerError("class_names must contain non-empty names")
        if len(set(names)) != len(names):
            raise DetectionWorkerError("class_names must be unique")
        object.__setattr__(self, "class_names", names)
        object.__setattr__(self, "window_size", _pair(self.window_size, "window_size"))
        object.__setattr__(self, "stride", _pair(self.stride, "stride"))
        if any(self.stride[index] > self.window_size[index] for index in range(2)):
            raise DetectionWorkerError("stride must not exceed window_size")
        if not isinstance(self.reflect_padding, bool):
            raise DetectionWorkerError("reflect_padding must be a boolean")
        if self.center_weighting not in {"linear", "uniform"}:
            raise DetectionWorkerError("center_weighting must be 'linear' or 'uniform'")
        object.__setattr__(self, "device", _device_name(self.device))
        _require_text(self.model_id, "model_id")
        object.__setattr__(self, "provenance", _json_mapping(self.provenance, "provenance"))
        if self.classifier_weights is not None:
            try:
                values = np.asarray(self.classifier_weights, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise DetectionWorkerError("classifier_weights must be numeric") from error
            if values.ndim not in (1, 2) or not np.all(np.isfinite(values)):
                raise DetectionWorkerError("classifier_weights must be finite")
            object.__setattr__(self, "classifier_weights", values.tolist())

    def to_payload(self) -> dict[str, Any]:
        """Return JSON-compatible values without a project or SQLite handle."""

        return {
            "request_id": self.request_id,
            "source": self.source.tolist(),
            "source_dtype": str(self.source.dtype),
            "staging_path": str(self.staging_path),
            "run_id": self.run_id,
            "profile_id": self.profile_id,
            "approved_run_id": self.approved_run_id,
            "approved_profile_id": self.approved_profile_id,
            "approved_evaluation_id": self.approved_evaluation_id,
            "class_names": list(self.class_names),
            "window_size": list(self.window_size),
            "stride": list(self.stride),
            "reflect_padding": self.reflect_padding,
            "center_weighting": self.center_weighting,
            "device": self.device,
            "model_id": self.model_id,
            "classifier_weights": self.classifier_weights,
            "provenance": dict(self.provenance),
        }

    def to_json(self) -> str:
        return json.dumps(
            {"version": PROTOCOL_VERSION, "type": "detection_request", "payload": self.to_payload()},
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "DetectionRequest":
        expected = {
            "request_id",
            "source",
            "source_dtype",
            "staging_path",
            "run_id",
            "profile_id",
            "approved_run_id",
            "approved_profile_id",
            "approved_evaluation_id",
            "class_names",
            "window_size",
            "stride",
            "reflect_padding",
            "center_weighting",
            "device",
            "model_id",
            "classifier_weights",
            "provenance",
        }
        if set(value) != expected:
            raise DetectionProtocolError("detection request fields are invalid")
        try:
            source = np.asarray(value["source"], dtype=value["source_dtype"])
            values = dict(value)
            values["source"] = source
            values.pop("source_dtype")
            return cls(**values)
        except (TypeError, ValueError) as error:
            if isinstance(error, DetectionWorkerError):
                raise
            raise DetectionProtocolError("invalid detection request") from error


@dataclass(frozen=True, slots=True)
class DetectionProgress:
    """Truthful progress emitted by the Detection worker process."""

    message_type: ClassVar[str] = "detection_progress"
    request_id: str
    phase: str
    completed: int
    total: int
    message: str = ""

    def __post_init__(self) -> None:
        _require_text(self.request_id, "request_id")
        _require_text(self.phase, "phase")
        _non_negative_int(self.completed, "completed")
        _positive_int(self.total, "total")
        if self.completed > self.total:
            raise DetectionProtocolError("completed cannot exceed total")
        if not isinstance(self.message, str):
            raise DetectionProtocolError("message must be a string")


@dataclass(frozen=True, slots=True)
class DetectionTerminal:
    """Truthful completed, cancelled, or failed worker state."""

    message_type: ClassVar[str] = "detection_terminal"
    request_id: str
    status: str
    message: str = ""
    error_code: str | None = None
    artifact_staging_path: str | Path | None = None

    def __post_init__(self) -> None:
        _require_text(self.request_id, "request_id")
        if self.status not in _TERMINAL_STATUSES:
            raise DetectionProtocolError("unsupported Detection terminal status")
        if not isinstance(self.message, str):
            raise DetectionProtocolError("message must be a string")
        if self.error_code is not None:
            _require_text(self.error_code, "error_code")
        if isinstance(self.artifact_staging_path, Path):
            object.__setattr__(self, "artifact_staging_path", str(self.artifact_staging_path))


@dataclass(slots=True)
class DetectionWorkerHandle:
    """Parent-side controls and output queue for one Detection worker."""

    process: mp.Process
    queue: Any
    cancel_event: Any
    request_id: str

    def cancel(self, reason: str = "user") -> bool:
        _require_text(reason, "reason")
        if self.cancel_event.is_set():
            return False
        self.cancel_event.set()
        return True

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self.process.join(timeout)

    @property
    def exitcode(self) -> int | None:
        return self.process.exitcode


def encode_message(message: DetectionRequest | DetectionProgress | DetectionTerminal) -> str:
    """Encode one deterministic versioned worker message."""

    if isinstance(message, DetectionRequest):
        message_type = "detection_request"
        payload = message.to_payload()
    elif isinstance(message, DetectionProgress):
        message_type = message.message_type
        payload = {
            "request_id": message.request_id,
            "phase": message.phase,
            "completed": message.completed,
            "total": message.total,
            "message": message.message,
        }
    elif isinstance(message, DetectionTerminal):
        message_type = message.message_type
        payload = {
            "request_id": message.request_id,
            "status": message.status,
            "message": message.message,
            "error_code": message.error_code,
            "artifact_staging_path": message.artifact_staging_path,
        }
    else:
        raise DetectionProtocolError("unsupported Detection message")
    return json.dumps(
        {"version": PROTOCOL_VERSION, "type": message_type, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
    )


def decode_message(serialized: str) -> DetectionRequest | DetectionProgress | DetectionTerminal:
    """Decode and validate one worker message."""

    if not isinstance(serialized, str):
        raise DetectionProtocolError("serialized message must be a string")
    try:
        envelope = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise DetectionProtocolError("invalid Detection message JSON") from error
    if not isinstance(envelope, dict) or set(envelope) != {"version", "type", "payload"}:
        raise DetectionProtocolError("invalid Detection message envelope")
    if envelope["version"] != PROTOCOL_VERSION or not isinstance(envelope["payload"], dict):
        raise DetectionProtocolError("unsupported Detection message version or payload")
    payload = envelope["payload"]
    try:
        if envelope["type"] == "detection_request":
            return DetectionRequest.from_payload(payload)
        if envelope["type"] == DetectionProgress.message_type:
            expected = {"request_id", "phase", "completed", "total", "message"}
            if set(payload) != expected:
                raise DetectionProtocolError("invalid Detection progress fields")
            return DetectionProgress(**payload)
        if envelope["type"] == DetectionTerminal.message_type:
            expected = {"request_id", "status", "message", "error_code", "artifact_staging_path"}
            if set(payload) != expected:
                raise DetectionProtocolError("invalid Detection terminal fields")
            return DetectionTerminal(**payload)
    except (TypeError, ValueError) as error:
        if isinstance(error, DetectionProtocolError):
            raise
        raise DetectionProtocolError("invalid Detection message payload") from error
    raise DetectionProtocolError(f"unknown Detection message type: {envelope['type']!r}")


def start_detection_worker(
    request: DetectionRequest,
    *,
    step_delay: float = 0.0,
    context: mp.context.BaseContext | None = None,
) -> DetectionWorkerHandle:
    """Start one isolated Detection process using the platform spawn method."""

    if not isinstance(request, DetectionRequest):
        raise DetectionWorkerError("request must be a DetectionRequest value")
    _delay(step_delay)
    worker_context = context or mp.get_context("spawn")
    output_queue = worker_context.Queue()
    cancel_event = worker_context.Event()
    process = worker_context.Process(
        target=run_detection_worker,
        args=(request.to_json(), output_queue, cancel_event),
        kwargs={"step_delay": step_delay},
        name=f"wafer-detection-{request.request_id}",
        daemon=True,
    )
    process.start()
    return DetectionWorkerHandle(process, output_queue, cancel_event, request.request_id)


def run_detection_worker(
    request: DetectionRequest | Mapping[str, Any] | str,
    output_queue: Any,
    cancel_event: Any,
    *,
    step_delay: float = 0.0,
) -> None:
    """Evaluate native windows and stage a checksummed CAM map set."""

    _delay(step_delay)
    request_id = _request_id(request)
    request_value: DetectionRequest | None = None
    staging: Path | None = None
    try:
        request_value = _coerce_request(request)
        staging = Path(request_value.staging_path).expanduser().resolve()
        _cleanup_stage(staging)
        windows = enumerate_inference_windows(
            request_value.source.shape[1],
            request_value.source.shape[0],
            request_value.window_size,
            request_value.stride,
        )
        total = len(windows) + 2
        _emit(output_queue, DetectionProgress(request_value.request_id, "detect", 0, total, "Starting CAM Detection"))
        if _cancelled(cancel_event):
            _emit_cancelled(output_queue, request_value)
            return
        device = _resolve_device(request_value.device)
        patches = np.empty(
            (len(windows), request_value.window_size[1], request_value.window_size[0]),
            dtype=np.float32,
        )
        for index, window in enumerate(windows):
            if _cancelled(cancel_event):
                _cleanup_stage(staging)
                _emit_cancelled(output_queue, request_value)
                return
            patches[index] = _window_patch(request_value.source, window)
            _emit(
                output_queue,
                DetectionProgress(
                    request_value.request_id,
                    "detect",
                    index + 1,
                    total,
                    f"Evaluated source window {index + 1}/{len(windows)} on {device}",
                ),
            )
            if step_delay:
                time.sleep(step_delay)

        activations = _edge_activations(patches, device)
        weights = _classifier_weights(request_value.classifier_weights, len(request_value.class_names))
        # The profile is the approved evaluation binding known to this
        # value-only worker.  A caller may provide the explicit evaluation
        # identity; otherwise retain a transparent profile-bound value since
        # SQLite validation belongs to the GUI service.
        evaluation_id = request_value.approved_evaluation_id or request_value.profile_id
        provenance = {
            **dict(request_value.provenance),
            "request_id": request_value.request_id,
            "run_id": request_value.run_id,
            "profile_id": request_value.profile_id,
            "evaluation_id": evaluation_id,
            "evaluation_id_inferred": request_value.approved_evaluation_id is None,
            "model_id": request_value.model_id,
            "device": device,
            "source_shape": [int(request_value.source.shape[1]), int(request_value.source.shape[0])],
            "source_dtype": str(request_value.source.dtype),
            "source_coordinate_system": "source-image pixels",
            "window_count": len(windows),
        }
        artifact = generate_cam_artifact(
            windows,
            activations,
            weights,
            class_names=request_value.class_names,
            window_settings={
                "window_size": list(request_value.window_size),
                "stride": list(request_value.stride),
                "reflect_padding": request_value.reflect_padding,
                "center_weighting": request_value.center_weighting,
            },
            provenance=provenance,
            model_id=request_value.model_id,
            profile_id=request_value.profile_id,
            evaluation_id=evaluation_id,
        )
        _emit(
            output_queue,
            DetectionProgress(request_value.request_id, "map", len(windows) + 1, total, "Stitched native-coordinate CAM maps"),
        )
        if _cancelled(cancel_event):
            _cleanup_stage(staging)
            _emit_cancelled(output_queue, request_value)
            return
        _write_staged_artifacts(staging, artifact, provenance)
        if _cancelled(cancel_event):
            _cleanup_stage(staging)
            _emit_cancelled(output_queue, request_value)
            return
        _emit(output_queue, DetectionProgress(request_value.request_id, "stage", total, total, "Validated checksummed map set"))
        _emit(
            output_queue,
            DetectionTerminal(
                request_value.request_id,
                "completed",
                "Detection completed; source coordinates and provenance retained.",
                artifact_staging_path=str(staging),
            ),
        )
    except Exception as error:  # pragma: no cover - defensive process boundary
        if staging is not None:
            _cleanup_stage(staging)
        _emit(
            output_queue,
            DetectionTerminal(request_id, "failed", f"Detection worker failed: {error}", "worker_error"),
        )


def validate_staged_detection_artifacts(staging_path: str | Path) -> tuple[Path, ...]:
    """Validate a completed map stage without opening SQLite."""

    staging = Path(staging_path).expanduser().resolve()
    manifest_path = staging / "manifest.json"
    if not staging.is_dir() or not manifest_path.is_file():
        raise DetectionWorkerError("completed Detection stage requires manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DetectionWorkerError("invalid Detection artifact manifest") from error
    if not isinstance(manifest, dict) or manifest.get("format") != _STAGE_FORMAT:
        raise DetectionWorkerError("invalid Detection artifact manifest format")
    entries = manifest.get("files")
    required = manifest.get("required_files")
    if not isinstance(entries, list) or required != list(_STAGE_FILES):
        raise DetectionWorkerError("Detection manifest must declare maps and provenance")
    by_name: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
            raise DetectionWorkerError("invalid Detection manifest entry")
        by_name[entry["path"]] = entry["sha256"]
    if set(by_name) != set(_STAGE_FILES) or any(len(value) != 64 for value in by_name.values()):
        raise DetectionWorkerError("Detection manifest files are incomplete")
    validated: list[Path] = []
    for name in _STAGE_FILES:
        candidate = (staging / name).resolve()
        try:
            candidate.relative_to(staging)
        except ValueError as error:
            raise DetectionWorkerError("Detection artifact path escapes staging") from error
        if not candidate.is_file() or _sha256(candidate) != by_name[name].lower():
            raise DetectionWorkerError(f"Detection artifact checksum mismatch: {name}")
        validated.append(candidate)
    try:
        maps = json.loads((staging / "maps.json").read_text(encoding="utf-8"))
        provenance = json.loads((staging / "provenance.json").read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DetectionWorkerError("invalid Detection map or provenance JSON") from error
    if not isinstance(maps, dict) or maps.get("artifact_type") != "approximate_cam":
        raise DetectionWorkerError("Detection maps must be approximate CAM data")
    if not isinstance(provenance, dict) or not provenance.get("run_id") or not provenance.get("profile_id"):
        raise DetectionWorkerError("Detection provenance must identify run and profile")
    return tuple(validated)


# Naming aliases keep the stage seam easy to discover from GUI services/tests.
validate_detection_stage = validate_staged_detection_artifacts
validate_staged_artifacts = validate_staged_detection_artifacts


def synthetic_edge_image(
    width: int,
    height: int,
    *,
    edge_x: int | None = None,
    edge_y: int | None = None,
    dtype: Any = np.uint8,
) -> np.ndarray:
    """Create a deterministic two-tone edge image for worker smoke tests."""

    _positive_int(width, "width")
    _positive_int(height, "height")
    result = np.zeros((height, width), dtype=dtype)
    if edge_x is None:
        edge_x = width // 2
    if edge_y is None:
        result[:, max(0, min(width, edge_x)) :] = 1
    else:
        _non_negative_int(edge_y, "edge_y")
        result[max(0, min(height, edge_y)) :, :] = 1
    if np.issubdtype(result.dtype, np.integer):
        result *= np.iinfo(result.dtype).max
    else:
        result *= 1.0
    return result


def _source_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise DetectionWorkerError("source must be a grayscale ndarray or tensor") from error
    if result.ndim != 2 or result.shape[0] <= 0 or result.shape[1] <= 0:
        raise DetectionWorkerError("source must be a non-empty 2-D grayscale image")
    if not np.issubdtype(result.dtype, np.number) or not np.all(np.isfinite(result)):
        raise DetectionWorkerError("source must contain finite numeric pixels")
    return result.copy()


def _window_patch(source: np.ndarray, window: Window) -> np.ndarray:
    patch = np.empty((window.height, window.width), dtype=np.float32)
    for local_y in range(window.height):
        for local_x in range(window.width):
            point = window.window_to_source(local_x, local_y)
            patch[local_y, local_x] = float(source[int(point.y), int(point.x)])
    return patch


def _edge_activations(patches: np.ndarray, device: str) -> np.ndarray:
    if device.startswith("cuda"):
        try:
            import torch
        except ImportError as error:  # pragma: no cover - depends on environment
            raise DetectionWorkerError("CUDA device requested but PyTorch is unavailable") from error
        values = torch.as_tensor(patches, dtype=torch.float32, device=device)
        horizontal = torch.zeros_like(values)
        vertical = torch.zeros_like(values)
        horizontal[:, :, 1:] = torch.abs(values[:, :, 1:] - values[:, :, :-1])
        vertical[:, 1:, :] = torch.abs(values[:, 1:, :] - values[:, :-1, :])
        edge = torch.maximum(horizontal, vertical)
        return _normalize_edge(edge.detach().cpu().numpy())[:, None, :, :]
    horizontal = np.zeros_like(patches, dtype=np.float32)
    vertical = np.zeros_like(patches, dtype=np.float32)
    horizontal[:, :, 1:] = np.abs(patches[:, :, 1:] - patches[:, :, :-1])
    vertical[:, 1:, :] = np.abs(patches[:, 1:, :] - patches[:, :-1, :])
    return _normalize_edge(np.maximum(horizontal, vertical))[:, None, :, :]


def _normalize_edge(edge: np.ndarray) -> np.ndarray:
    result = np.zeros_like(edge, dtype=np.float32)
    for index, values in enumerate(edge):
        maximum = float(np.max(values))
        if maximum > 0:
            result[index] = np.clip(values / maximum, 0.0, 1.0)
    return result


def _classifier_weights(value: Any, class_count: int) -> np.ndarray:
    if value is None:
        return np.ones((class_count, 1), dtype=np.float32)
    result = np.asarray(value, dtype=np.float32)
    if result.ndim == 1:
        result = result[:, None]
    if result.shape != (class_count, 1) or not np.all(np.isfinite(result)):
        raise DetectionWorkerError("classifier_weights must match class_names")
    return result


def _write_staged_artifacts(staging: Path, artifact: CamDetectionArtifact, provenance: Mapping[str, Any]) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    maps_path = staging / "maps.json"
    provenance_path = staging / "provenance.json"
    manifest_path = staging / "manifest.json"
    maps_path.write_text(artifact.to_json() + "\n", encoding="utf-8")
    provenance_path.write_text(
        json.dumps(dict(provenance), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    entries = [{"path": name, "sha256": _sha256(staging / name)} for name in _STAGE_FILES]
    manifest = {
        "format": _STAGE_FORMAT,
        "required_files": list(_STAGE_FILES),
        "files": entries,
        "provenance": {"run_id": provenance.get("run_id"), "profile_id": provenance.get("profile_id")},
    }
    temporary_manifest = staging / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    _validate_manifest_payload(staging, manifest)
    os.replace(temporary_manifest, manifest_path)
    validate_staged_detection_artifacts(staging)


def _validate_manifest_payload(staging: Path, manifest: Mapping[str, Any]) -> None:
    if manifest.get("required_files") != list(_STAGE_FILES):
        raise DetectionWorkerError("Detection stage required files are incomplete")
    for entry in manifest.get("files", []):
        if not isinstance(entry, Mapping) or _sha256(staging / entry["path"]) != entry["sha256"]:
            raise DetectionWorkerError("Detection stage checksum validation failed")


def _cleanup_stage(staging: Path) -> None:
    for name in (*_STAGE_FILES, "manifest.json.tmp"):
        try:
            (staging / name).unlink(missing_ok=True)
        except OSError:
            pass


def _emit(output_queue: Any, message: DetectionProgress | DetectionTerminal) -> None:
    output_queue.put(encode_message(message))


def _emit_cancelled(output_queue: Any, request: DetectionRequest) -> None:
    _emit(output_queue, DetectionTerminal(request.request_id, "cancelled", "Detection cancelled; no map set was published."))


def _coerce_request(value: DetectionRequest | Mapping[str, Any] | str) -> DetectionRequest:
    if isinstance(value, DetectionRequest):
        return value
    if isinstance(value, str):
        decoded = decode_message(value)
        if not isinstance(decoded, DetectionRequest):
            raise DetectionWorkerError("serialized worker value must be a DetectionRequest")
        return decoded
    if isinstance(value, Mapping):
        try:
            return DetectionRequest.from_payload(value)
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, DetectionWorkerError):
                raise
            raise DetectionWorkerError("invalid worker request payload") from error
    raise DetectionWorkerError("request must be a DetectionRequest, payload, or JSON value")


def _request_id(value: Any) -> str:
    if isinstance(value, DetectionRequest):
        return value.request_id
    if isinstance(value, Mapping) and isinstance(value.get("request_id"), str):
        return value["request_id"]
    if isinstance(value, str):
        try:
            envelope = json.loads(value)
            payload = envelope.get("payload", {})
            if isinstance(payload, Mapping) and isinstance(payload.get("request_id"), str):
                return payload["request_id"]
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return "unknown-detection"


def _resolve_device(value: str) -> str:
    name = _device_name(value)
    if name.startswith("cuda"):
        try:
            import torch
        except ImportError as error:  # pragma: no cover - depends on environment
            raise DetectionWorkerError("CUDA device requested but PyTorch is unavailable") from error
        if not torch.cuda.is_available():
            raise DetectionWorkerError("CUDA device requested but CUDA is unavailable")
        try:
            torch.device(name)
        except (RuntimeError, ValueError) as error:
            raise DetectionWorkerError(f"invalid CUDA device: {name}") from error
    return name


def _device_name(value: Any) -> str:
    if not isinstance(value, str) or value not in {"cpu", "cuda"} and not value.startswith("cuda:"):
        raise DetectionWorkerError("device must be 'cpu', 'cuda', or 'cuda:N'")
    return value


def _pair(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, bool):
        raise DetectionWorkerError(f"{name} must be a positive integer or pair")
    values = (value, value) if isinstance(value, int) else value
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence) or len(values) != 2:
        raise DetectionWorkerError(f"{name} must be a positive integer or pair")
    result = tuple(values)
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in result):
        raise DetectionWorkerError(f"{name} values must be positive integers")
    return result  # type: ignore[return-value]


def _coalesce_id(first: str | None, second: str | None, name: str) -> str | None:
    if first is not None and second is not None and first != second:
        raise DetectionWorkerError(f"{name} aliases must identify the same record")
    return first if first is not None else second


def _json_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DetectionWorkerError(f"{name} must be a mapping")
    result = dict(value)
    try:
        json.dumps(result, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise DetectionWorkerError(f"{name} must contain JSON-compatible values") from error
    return result


def _require_text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DetectionWorkerError(f"{name} must be a non-empty string")


def _positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DetectionWorkerError(f"{name} must be a positive integer")


def _non_negative_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DetectionWorkerError(f"{name} must be a non-negative integer")


def _delay(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise DetectionWorkerError("step_delay must be a non-negative number")


def _cancelled(event: Any) -> bool:
    return event is not None and bool(event.is_set())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DetectionProgress",
    "DetectionProtocolError",
    "DetectionRequest",
    "DetectionTerminal",
    "DetectionWorkerError",
    "DetectionWorkerHandle",
    "decode_message",
    "encode_message",
    "run_detection_worker",
    "start_detection_worker",
    "synthetic_edge_image",
    "validate_detection_stage",
    "validate_staged_artifacts",
    "validate_staged_detection_artifacts",
]
