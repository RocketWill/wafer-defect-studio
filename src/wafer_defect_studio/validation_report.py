"""Value-based supported-environment validation records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One deterministic pass/fail fact in a validation report."""

    name: str
    passed: bool
    details: str = ""
    category: str = "general"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ValidationCheck.name must be a non-empty string")
        if not isinstance(self.passed, bool):
            raise ValueError("ValidationCheck.passed must be a boolean")
        if not isinstance(self.details, str):
            raise ValueError("ValidationCheck.details must be a string")
        if not isinstance(self.category, str) or not self.category.strip():
            raise ValueError("ValidationCheck.category must be a non-empty string")

    @property
    def ok(self) -> bool:
        """Short alias for callers that use ``ok`` for a pass/fail fact."""

        return self.passed

    @property
    def kind(self) -> str:
        """Alias for the check category."""

        return self.category


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Immutable environment facts, checks, and explicit known limitations."""

    environment: Mapping[str, Any]
    checks: tuple[ValidationCheck, ...] = ()
    known_limits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.environment, Mapping):
            raise ValueError("environment must be a mapping")
        if any(not isinstance(key, str) or not key.strip() for key in self.environment):
            raise ValueError("environment keys must be non-empty strings")
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))
        checks = tuple(self.checks)
        if any(not isinstance(check, ValidationCheck) for check in checks):
            raise ValueError("checks must contain only ValidationCheck values")
        object.__setattr__(self, "checks", checks)
        limits = tuple(self.known_limits)
        if any(not isinstance(limit, str) or not limit.strip() for limit in limits):
            raise ValueError("known_limits must contain non-empty strings")
        object.__setattr__(self, "known_limits", limits)

    def to_markdown(self) -> str:
        """Render a concise, stable Markdown report."""

        lines = ["# MVP Validation Report", "", "## Environment"]
        if self.environment:
            for key in sorted(self.environment):
                lines.append(f"- `{key}`: {_format_value(self.environment[key])}")
        else:
            lines.append("- None recorded.")

        lines.extend(("", "## Checks"))
        ordered_checks = sorted(
            self.checks,
            key=lambda check: (check.category, check.name, check.details, not check.passed),
        )
        if ordered_checks:
            for check in ordered_checks:
                state = "PASS" if check.passed else "FAIL"
                line = f"- {state} — {check.category} — {check.name}"
                if check.details:
                    line += f": {check.details}"
                lines.append(line)
        else:
            lines.append("- None recorded.")

        lines.extend(("", "## Known limitations"))
        if self.known_limits:
            lines.extend(f"- {limit}" for limit in sorted(self.known_limits))
        else:
            lines.append("- None recorded.")
        return "\n".join(lines) + "\n"

    @property
    def markdown(self) -> str:
        """Convenient read-only alias for :meth:`to_markdown`."""

        return self.to_markdown()


def render_markdown(report: ValidationReport) -> str:
    """Render a :class:`ValidationReport` through its public value seam."""

    if not isinstance(report, ValidationReport):
        raise TypeError("report must be a ValidationReport")
    return report.to_markdown()


def record_validation_report(
    environment: Mapping[str, Any],
    checks: Iterable[ValidationCheck] = (),
    known_limits: Iterable[str] = (),
) -> ValidationReport:
    """Build an immutable report without probing or changing the environment."""

    return ValidationReport(
        environment=dict(environment),
        checks=tuple(checks),
        known_limits=tuple(known_limits),
    )


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "ValidationCheck",
    "ValidationReport",
    "record_validation_report",
    "render_markdown",
]
