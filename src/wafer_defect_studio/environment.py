"""Collect JSON-safe environment facts for Training Run provenance."""

from __future__ import annotations

import importlib
from importlib import metadata
import platform
from typing import Any


_UNAVAILABLE = "unavailable"
_PACKAGE_NAMES = ("numpy", "PySide6", "torch", "torchvision")


def collect_training_environment() -> dict[str, Any]:
    """Return canonical, serializable environment facts without project I/O."""

    torch = _optional_module("torch")
    cuda, cuda_driver, gpu = _cuda_facts(torch)
    return {
        "python": _text(platform.python_version()),
        "pytorch": _package_or_module_version("torch", torch),
        "torchvision": _package_or_module_version(
            "torchvision", _optional_module("torchvision")
        ),
        "cuda": cuda,
        "cuda_driver": cuda_driver,
        "os": _text(platform.platform()),
        "gpu": gpu,
        "packages": {
            name: _distribution_version(name)
            for name in _PACKAGE_NAMES
        },
    }


def _optional_module(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _distribution_version(name: str) -> str:
    try:
        return _text(metadata.version(name))
    except Exception:
        return _UNAVAILABLE


def _package_or_module_version(name: str, module: Any | None) -> str:
    version = _distribution_version(name)
    if version != _UNAVAILABLE:
        return version
    if module is None:
        return _UNAVAILABLE
    return _text(getattr(module, "__version__", None))


def _cuda_facts(torch: Any | None) -> tuple[str, str, str]:
    if torch is None:
        return _UNAVAILABLE, _UNAVAILABLE, _UNAVAILABLE
    try:
        cuda_api = torch.cuda
        if not bool(cuda_api.is_available()):
            return _UNAVAILABLE, _UNAVAILABLE, _UNAVAILABLE
        cuda_version = _text(getattr(getattr(torch, "version", None), "cuda", None))
        gpu = _text(cuda_api.get_device_name(0))
        driver = _cuda_driver_version(torch, cuda_api)
        return cuda_version, driver, gpu
    except Exception:
        return _UNAVAILABLE, _UNAVAILABLE, _UNAVAILABLE


def _cuda_driver_version(torch: Any, cuda_api: Any) -> str:
    for source, attribute in (
        (cuda_api, "driver_version"),
        (getattr(torch, "_C", None), "_cuda_getDriverVersion"),
    ):
        value = getattr(source, attribute, None)
        try:
            value = value() if callable(value) else value
        except Exception:
            continue
        text = _text(value)
        if text != _UNAVAILABLE:
            return text
    return _UNAVAILABLE


def _text(value: Any) -> str:
    if value is None:
        return _UNAVAILABLE
    text = str(value).strip()
    return text or _UNAVAILABLE
