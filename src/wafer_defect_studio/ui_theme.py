"""Minimal System/Light semantic theme tokens and focus indicator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QWidget


class ThemeMode(str, Enum):
    SYSTEM = "system"
    LIGHT = "light"


@dataclass(frozen=True, slots=True)
class SemanticColors:
    """Stable semantic colors; widgets should not infer meaning from color alone."""

    window_background: str = "#f6f8fb"
    surface: str = "#ffffff"
    text: str = "#172033"
    muted_text: str = "#475569"
    border: str = "#94a3b8"
    focus: str = "#1d4ed8"
    success: str = "#166534"
    warning: str = "#92400e"
    error: str = "#b91c1c"

    @property
    def focus_contrast(self) -> float:
        return _contrast_ratio(self.focus, self.surface)


@dataclass(frozen=True, slots=True)
class TypographyScale:
    """System-font roles expressed as Qt point sizes.

    The values use a 1.25 modular step around the 12 pt body baseline.  The
    font family remains the one supplied by the operating system; this scale
    only controls point sizes.
    """

    caption_pt: float = 9.6
    body_pt: float = 12.0
    section_pt: float = 15.0
    title_pt: float = 18.75


_TYPOGRAPHY_SCALE = TypographyScale()


def typography_scale() -> TypographyScale:
    """Return the immutable system-font typography scale."""

    return _TYPOGRAPHY_SCALE


def apply_theme(
    target: QApplication | QWidget,
    mode: ThemeMode | str = ThemeMode.SYSTEM,
) -> SemanticColors:
    """Apply System/Light palette and a visible keyboard focus indicator."""

    chosen = _coerce_mode(mode)
    if not isinstance(target, (QApplication, QWidget)):
        raise TypeError("target must be a QApplication or QWidget")
    colors = SemanticColors()
    font: QFont = target.font()
    font.setPointSizeF(typography_scale().body_pt)
    target.setFont(font)
    target.setStyleSheet(_focus_stylesheet(colors))
    if chosen is ThemeMode.LIGHT:
        _apply_light_palette(target, colors)
    return colors


def semantic_colors(mode: ThemeMode | str = ThemeMode.SYSTEM) -> SemanticColors:
    """Return immutable semantic tokens without changing application state."""

    _coerce_mode(mode)
    return SemanticColors()


def _apply_light_palette(target: QApplication | QWidget, colors: SemanticColors) -> None:
    palette = target.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(colors.window_background))
    palette.setColor(QPalette.ColorRole.Base, QColor(colors.surface))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors.window_background))
    palette.setColor(QPalette.ColorRole.Text, QColor(colors.text))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(colors.text))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors.text))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(colors.muted_text))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(colors.focus))
    target.setPalette(palette)


def _focus_stylesheet(colors: SemanticColors) -> str:
    return (
        "QPushButton:focus, QToolButton:focus, QCheckBox:focus, "
        "QComboBox:focus, QLineEdit:focus, QSpinBox:focus, "
        "QDoubleSpinBox:focus, QSlider:focus, QTableWidget:focus, "
        "QListWidget:focus, QGraphicsView:focus, QPlainTextEdit:focus, "
        "QTabWidget:focus, QTabBar:focus, QToolBar:focus { "
        f"border: 2px solid {colors.focus}; "
        f"outline: 1px solid {colors.border}; "
        "border-radius: 2px; "
        "}"
    )


def _coerce_mode(value: ThemeMode | str) -> ThemeMode:
    try:
        return value if isinstance(value, ThemeMode) else ThemeMode(value)
    except (TypeError, ValueError) as error:
        raise ValueError("mode must be System or Light") from error


def _contrast_ratio(foreground: str, background: str) -> float:
    foreground_rgb = QColor(foreground)
    background_rgb = QColor(background)
    return (_relative_luminance(foreground_rgb) + 0.05) / (
        _relative_luminance(background_rgb) + 0.05
    ) if _relative_luminance(foreground_rgb) >= _relative_luminance(background_rgb) else (
        _relative_luminance(background_rgb) + 0.05
    ) / (_relative_luminance(foreground_rgb) + 0.05)


def _relative_luminance(color: QColor) -> float:
    channels = []
    for channel in (color.redF(), color.greenF(), color.blueF()):
        channels.append(channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


__all__ = [
    "SemanticColors",
    "ThemeMode",
    "TypographyScale",
    "apply_theme",
    "semantic_colors",
    "typography_scale",
]
