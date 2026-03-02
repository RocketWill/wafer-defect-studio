"""Application entry point for the Wafer Defect Studio desktop shell."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main(argv: Sequence[str] | None = None) -> int:
    """Create the application shell, show its main window, and run Qt."""

    application_args = list(sys.argv if argv is None else argv)
    app = QApplication.instance() or QApplication(application_args)
    window = MainWindow()
    window.show()
    return app.exec()
