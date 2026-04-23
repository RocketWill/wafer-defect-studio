"""Versioned, value-only messages exchanged with a training worker.

The worker receives JSON values and a staging directory.  It never receives a
SQLite connection or the GUI-side project service; only the GUI service may
publish staged artifacts back into the project database.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path
from typing import Any, ClassVar, TypeAlias


PROTOCOL_VERSION = 1


class TrainingProtocolError(ValueError):
    """Raised when a training message is invalid or unsupported."""


class UnsupportedProtocolVersion(TrainingProtocolError):
    """Raised when a message was produced by an unsupported protocol version."""


class _VersionedMessage:
    """Shared JSON helpers for immutable protocol messages."""

    message_type: ClassVar[str]

    @property
    def version(self) -> int:
        return PROTOCOL_VERSION

    def to_json(self) -> str:
        return encode_message(self)

    @classmethod
    def from_json(cls, serialized: str) -> Any:
        message = decode_message(serialized)
        if not isinstance(message, cls):
            raise TrainingProtocolError(
                f"Expected {cls.__name__}, received {type(message).__name__}"
            )
        return message


@dataclass(frozen=True, slots=True)
class TrainingConfig(_VersionedMessage):
    """Immutable model and data values needed by a worker."""

    message_type: ClassVar[str] = "training_config"

    snapshot_id: str
    split_id: str
    class_count: int
    epochs: int
    batch_size: int
    architecture: str = "resnet18"
    device: str = "auto"
    seed: int = 0
    learning_rate: float = 0.001
    weights_policy: str = "none"
    patch_size: int | None = None
    patch_stride: int | None = None

    def __post_init__(self) -> None:
        for name in ("snapshot_id", "split_id", "architecture", "device", "weights_policy"):
            _require_non_empty_string(getattr(self, name), name)
        if self.architecture.lower() != "resnet18":
            raise TrainingProtocolError("only the resnet18 architecture is supported")
        if self.weights_policy not in {"none", "imagenet"}:
            raise TrainingProtocolError("weights_policy must be 'none' or 'imagenet'")
        for name in ("class_count", "epochs", "batch_size"):
            _require_positive_int(getattr(self, name), name)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TrainingProtocolError("seed must be an integer")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not isfinite(float(self.learning_rate))
            or self.learning_rate <= 0
        ):
            raise TrainingProtocolError("learning_rate must be a positive finite number")
        if (self.patch_size is None) != (self.patch_stride is None):
            raise TrainingProtocolError(
                "patch_size and patch_stride must be provided together"
            )
        if self.patch_size is not None:
            _require_positive_int(self.patch_size, "patch_size")
            _require_positive_int(self.patch_stride, "patch_stride")
            if self.patch_stride > self.patch_size:
                raise TrainingProtocolError("patch_stride cannot exceed patch_size")

    def validate_patch_geometry(self, sample_width: int, sample_height: int) -> None:
        """Reject configured Model Patches that cannot fit one frozen Grid sample."""

        _require_positive_int(sample_width, "sample_width")
        _require_positive_int(sample_height, "sample_height")
        if self.patch_size is None:
            raise TrainingProtocolError("patch geometry is not configured")
        if self.patch_size > sample_width or self.patch_size > sample_height:
            raise TrainingProtocolError(
                f"patch_size {self.patch_size} exceeds sample rectangle "
                f"{sample_width}x{sample_height}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "split_id": self.split_id,
            "class_count": self.class_count,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "architecture": self.architecture,
            "device": self.device,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "weights_policy": self.weights_policy,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "TrainingConfig":
        _require_object(value, "config")
        expected = {item.name for item in fields(cls)}
        legacy = expected - {"patch_size", "patch_stride"}
        if set(value) == legacy:
            value = {**value, "patch_size": None, "patch_stride": None}
        else:
            _require_keys(value, expected, "config")
        try:
            return cls(**value)
        except (TypeError, ValueError) as error:
            if isinstance(error, TrainingProtocolError):
                raise
            raise TrainingProtocolError("invalid training config") from error


@dataclass(frozen=True, slots=True)
class TrainingRequest(_VersionedMessage):
    """GUI-to-worker request containing only serializable values."""

    message_type: ClassVar[str] = "training_request"

    request_id: str
    config: TrainingConfig
    artifact_staging_path: str | Path
    input_bundle_path: str | Path | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.request_id, "request_id")
        if not isinstance(self.config, TrainingConfig):
            raise TrainingProtocolError("config must be a TrainingConfig value")
        if isinstance(self.artifact_staging_path, Path):
            object.__setattr__(self, "artifact_staging_path", str(self.artifact_staging_path))
        _require_non_empty_string(self.artifact_staging_path, "artifact_staging_path")
        if isinstance(self.input_bundle_path, Path):
            object.__setattr__(self, "input_bundle_path", str(self.input_bundle_path))
        if self.input_bundle_path is not None:
            _require_non_empty_string(self.input_bundle_path, "input_bundle_path")

    @property
    def run_id(self) -> str:
        """Alias used by run lifecycle code while the wire name stays request_id."""

        return self.request_id

    def to_worker_payload(self) -> dict[str, Any]:
        """Return the complete worker input without any writable project handle."""

        payload = {
            "request_id": self.request_id,
            "config": self.config.to_dict(),
            "artifact_staging_path": str(self.artifact_staging_path),
        }
        if self.input_bundle_path is not None:
            payload["input_bundle_path"] = str(self.input_bundle_path)
        return payload


@dataclass(frozen=True, slots=True)
class ProgressMessage(_VersionedMessage):
    """A worker progress event; it never mutates project state."""

    message_type: ClassVar[str] = "progress"

    request_id: str
    phase: str
    epoch: int
    total_epochs: int
    step: int
    total_steps: int
    loss: float | None = None
    eta_seconds: float | None = None
    message: str = ""

    def __post_init__(self) -> None:
        _require_non_empty_string(self.request_id, "request_id")
        _require_non_empty_string(self.phase, "phase")
        for name in ("epoch", "total_epochs", "step", "total_steps"):
            _require_non_negative_int(getattr(self, name), name)
        if self.epoch > self.total_epochs:
            raise TrainingProtocolError("epoch cannot exceed total_epochs")
        if self.step > self.total_steps:
            raise TrainingProtocolError("step cannot exceed total_steps")
        _require_optional_finite_number(self.loss, "loss")
        _require_optional_non_negative_number(self.eta_seconds, "eta_seconds")
        if not isinstance(self.message, str):
            raise TrainingProtocolError("message must be a string")


_TERMINAL_STATUSES = {"completed", "cancelled", "failed", "interrupted"}


@dataclass(frozen=True, slots=True)
class TerminalMessage(_VersionedMessage):
    """The truthful terminal state reported by a worker."""

    message_type: ClassVar[str] = "terminal"

    request_id: str
    status: str
    message: str = ""
    error_code: str | None = None
    artifact_staging_path: str | Path | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.request_id, "request_id")
        if self.status not in _TERMINAL_STATUSES:
            raise TrainingProtocolError(
                "status must be completed, cancelled, failed, or interrupted"
            )
        if not isinstance(self.message, str):
            raise TrainingProtocolError("message must be a string")
        if self.error_code is not None:
            _require_non_empty_string(self.error_code, "error_code")
        if isinstance(self.artifact_staging_path, Path):
            object.__setattr__(self, "artifact_staging_path", str(self.artifact_staging_path))
        if self.artifact_staging_path is not None:
            _require_non_empty_string(self.artifact_staging_path, "artifact_staging_path")


@dataclass(frozen=True, slots=True)
class CancelMessage(_VersionedMessage):
    """Explicit cooperative cancellation request."""

    message_type: ClassVar[str] = "cancel"

    request_id: str
    reason: str = "user"

    def __post_init__(self) -> None:
        _require_non_empty_string(self.request_id, "request_id")
        _require_non_empty_string(self.reason, "reason")


ProtocolMessage: TypeAlias = (
    TrainingConfig | TrainingRequest | ProgressMessage | TerminalMessage | CancelMessage
)


def encode_message(message: ProtocolMessage) -> str:
    """Encode one typed message as a deterministic versioned JSON envelope."""

    if isinstance(message, TrainingConfig):
        payload = message.to_dict()
    elif isinstance(message, TrainingRequest):
        payload = message.to_worker_payload()
    elif isinstance(message, ProgressMessage):
        payload = {
            "request_id": message.request_id,
            "phase": message.phase,
            "epoch": message.epoch,
            "total_epochs": message.total_epochs,
            "step": message.step,
            "total_steps": message.total_steps,
            "loss": message.loss,
            "eta_seconds": message.eta_seconds,
            "message": message.message,
        }
    elif isinstance(message, TerminalMessage):
        payload = {
            "request_id": message.request_id,
            "status": message.status,
            "message": message.message,
            "error_code": message.error_code,
            "artifact_staging_path": (
                str(message.artifact_staging_path)
                if message.artifact_staging_path is not None
                else None
            ),
        }
    elif isinstance(message, CancelMessage):
        payload = {"request_id": message.request_id, "reason": message.reason}
    else:
        raise TrainingProtocolError("message must be a supported training protocol value")
    return json.dumps(
        {"version": PROTOCOL_VERSION, "type": message.message_type, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def decode_message(serialized: str) -> ProtocolMessage:
    """Decode and validate a versioned message envelope."""

    if not isinstance(serialized, str):
        raise TrainingProtocolError("serialized message must be a JSON string")
    try:
        envelope = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise TrainingProtocolError("invalid training message JSON") from error
    _require_object(envelope, "message")
    _require_keys(envelope, {"version", "type", "payload"}, "message")
    version = envelope["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != PROTOCOL_VERSION:
        raise UnsupportedProtocolVersion(
            f"unsupported training protocol version: {version!r}"
        )
    message_type = envelope["type"]
    payload = envelope["payload"]
    _require_object(payload, "payload")
    try:
        if message_type == TrainingConfig.message_type:
            return TrainingConfig.from_dict(payload)
        if message_type == TrainingRequest.message_type:
            request_keys = {"request_id", "config", "artifact_staging_path"}
            if set(payload) not in (request_keys, request_keys | {"input_bundle_path"}):
                raise TrainingProtocolError(
                    f"request fields must be exactly {sorted(request_keys)!r} "
                    "or include input_bundle_path"
                )
            return TrainingRequest(
                request_id=payload["request_id"],
                config=TrainingConfig.from_dict(payload["config"]),
                artifact_staging_path=payload["artifact_staging_path"],
                input_bundle_path=payload.get("input_bundle_path"),
            )
        if message_type == ProgressMessage.message_type:
            _require_keys(
                payload,
                {
                    "request_id",
                    "phase",
                    "epoch",
                    "total_epochs",
                    "step",
                    "total_steps",
                    "loss",
                    "eta_seconds",
                    "message",
                },
                "progress",
            )
            return ProgressMessage(**payload)
        if message_type == TerminalMessage.message_type:
            _require_keys(
                payload,
                {"request_id", "status", "message", "error_code", "artifact_staging_path"},
                "terminal",
            )
            return TerminalMessage(**payload)
        if message_type == CancelMessage.message_type:
            _require_keys(payload, {"request_id", "reason"}, "cancel")
            return CancelMessage(**payload)
    except (TypeError, ValueError) as error:
        if isinstance(error, TrainingProtocolError):
            raise
        raise TrainingProtocolError(f"invalid {message_type!r} message payload") from error
    raise TrainingProtocolError(f"unknown training message type: {message_type!r}")


def _require_object(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise TrainingProtocolError(f"{name} must be a JSON object")


def _require_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise TrainingProtocolError(f"{name} fields must be exactly {sorted(expected)!r}")


def _require_non_empty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TrainingProtocolError(f"{name} must be a non-empty string")


def _require_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrainingProtocolError(f"{name} must be a positive integer")


def _require_non_negative_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingProtocolError(f"{name} must be a non-negative integer")


def _require_optional_finite_number(value: Any, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise TrainingProtocolError(f"{name} must be a finite number or null")


def _require_optional_non_negative_number(value: Any, name: str) -> None:
    _require_optional_finite_number(value, name)
    if value is not None and value < 0:
        raise TrainingProtocolError(f"{name} must be non-negative or null")


__all__ = [
    "CancelMessage",
    "PROTOCOL_VERSION",
    "ProgressMessage",
    "ProtocolMessage",
    "TerminalMessage",
    "TrainingConfig",
    "TrainingProtocolError",
    "TrainingRequest",
    "UnsupportedProtocolVersion",
    "decode_message",
    "encode_message",
]
