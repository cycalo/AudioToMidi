"""Smoke tests for HitMap GUI shell construction."""

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


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_constructs_hitmap_shell(qapp) -> None:
    win = MainWindow()
    assert win.windowTitle() == "HitMap"
    assert win.convert_btn.objectName() == "primaryButton"
    assert win.save_btn.objectName() == "secondaryButton"
    assert win.play_btn.objectName() == "transportButton"
    assert win.stop_btn.objectName() == "transportButton"
    assert win.device_combo.currentData() == "auto"
    assert win.controller._device == "auto"
    assert win.source_combo.objectName() == "previewCombo"
    assert win.clear_btn is not None
    win.show()
    assert win.width() >= 900
    win.close()
