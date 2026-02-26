"""Small value-based keyboard and semantic-label audit for Qt Widgets."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSlider,
    QSpinBox,
    QTableWidget,
    QWidget,
)


@dataclass(frozen=True, slots=True)
class AccessibilityFinding:
    """One actionable widget that still needs a semantic/accessibility fix."""

    object_name: str
    issue: str
    accessible_name: str = ""

    @property
    def reason(self) -> str:
        return self.issue


_ACTIONABLE_TYPES = (
    QAbstractButton,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QSlider,
    QTableWidget,
)


def ensure_accessible_labels(root: QWidget) -> tuple[AccessibilityFinding, ...]:
    """Apply text/object-name labels and keyboard focus to visible actions."""

    _require_root(root)
    for widget in _actionable_widgets(root, include_hidden=True):
        if not widget.accessibleName().strip():
            label = _label_for(widget)
            if label:
                widget.setAccessibleName(label)
        if widget.focusPolicy() == Qt.FocusPolicy.NoFocus:
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    return audit_primary_actions(root)


def audit_primary_actions(root: QWidget) -> tuple[AccessibilityFinding, ...]:
    """Return immutable failures for visible actionable widgets only."""

    _require_root(root)
    findings: list[AccessibilityFinding] = []
    for widget in _actionable_widgets(root):
        name = widget.accessibleName().strip()
        object_name = widget.objectName().strip() or widget.__class__.__name__
        if not name:
            findings.append(AccessibilityFinding(object_name, "missing accessible name"))
        if widget.focusPolicy() == Qt.FocusPolicy.NoFocus:
            findings.append(
                AccessibilityFinding(object_name, "focus policy is NoFocus", name)
            )
    return tuple(findings)


def _actionable_widgets(root: QWidget, *, include_hidden: bool = False) -> tuple[QWidget, ...]:
    widgets: list[QWidget] = []
    if isinstance(root, _ACTIONABLE_TYPES):
        widgets.append(root)
    for widget_type in _ACTIONABLE_TYPES:
        widgets.extend(root.findChildren(widget_type))
    return tuple(
        widget
        for widget in widgets
        if (include_hidden or widget.isVisibleTo(root))
        and not _is_qt_internal(widget, root)
    )


def _is_qt_internal(widget: QWidget, root: QWidget) -> bool:
    current = widget
    while current is not None and current is not root:
        if current.objectName().startswith("qt_"):
            return True
        current = current.parentWidget()
    return False


def _label_for(widget: QWidget) -> str:
    if isinstance(widget, QAbstractButton):
        text = widget.text().replace("&", " ").strip()
        if text:
            return text
    if isinstance(widget, QComboBox):
        text = widget.currentText().strip()
        if text:
            return text
    if isinstance(widget, QLineEdit):
        text = widget.placeholderText().strip()
        if text:
            return text
    return widget.objectName().strip()


def _require_root(root: object) -> None:
    if not isinstance(root, QWidget):
        raise TypeError("root must be a QWidget")


__all__ = [
    "AccessibilityFinding",
    "audit_primary_actions",
    "ensure_accessible_labels",
]
