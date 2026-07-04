"""Smoke tests for Phase 0 scaffolding."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """Shared QApplication for the test session."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_constructs(qapp: QApplication) -> None:
    """MainWindow can be instantiated under the offscreen platform."""
    window = MainWindow()
    assert window.windowTitle() == "Drum Stem to MIDI"


def test_main_module_importable() -> None:
    """Entry point module imports without error."""
    import app.main  # noqa: F401
