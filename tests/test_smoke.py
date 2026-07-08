"""Smoke tests for Phase 0 scaffolding."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ui.main_window import MainWindow, _SLIDER_DEFAULT, _SLIDER_MIN  # noqa: E402


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


def test_main_window_has_phase5_widgets(qapp: QApplication) -> None:
    """The Phase 5 controls exist and start in the expected disabled state."""
    window = MainWindow()
    assert window.browse_btn is not None
    assert window.convert_btn is not None
    assert window.save_btn is not None
    assert window.device_combo is not None
    assert window.device_combo.count() >= 2
    # Save + sensitivity are disabled until an analysis has run.
    assert not window.save_btn.isEnabled()
    assert not window.sensitivity_slider.isEnabled()
    # Convert is disabled until a WAV is chosen.
    assert not window.convert_btn.isEnabled()


def test_plugin_combo_lists_seven_profiles(qapp: QApplication) -> None:
    """The dropdown surfaces the 7 plugin profiles (general_midi excluded)."""
    window = MainWindow()
    assert window.plugin_combo.count() == 7
    stems = {window.plugin_combo.itemData(i) for i in range(window.plugin_combo.count())}
    assert "general_midi" not in stems
    assert "superior_drummer_3" in stems
    assert "drumforge" in stems


def test_waveform_view_constructs(qapp: QApplication) -> None:
    """WaveformView instantiates and its clear() is safe on an empty plot."""
    from app.ui.waveform_view import WaveformView

    view = WaveformView()
    view.clear()
    view.set_events([])


def test_elapsed_timer_resets_on_plugin_change(qapp: QApplication) -> None:
    window = MainWindow()
    window.elapsed_label.setText("Elapsed: 42.0s")
    window._elapsed_timer.start()
    window._on_plugin_changed(0)
    assert window.elapsed_label.text() == ""
    assert not window._elapsed_timer.isActive()


def test_convert_resets_elapsed_and_sensitivity(qapp: QApplication) -> None:
    window = MainWindow()
    window._wav_path = "dummy.wav"
    window.sensitivity_slider.setValue(_SLIDER_MIN)
    window.elapsed_label.setText("Elapsed: 10.0s")
    with patch.object(window.controller, "run_analysis"):
        window._on_convert()
    assert window.elapsed_label.text() == "Elapsed: 0.0s"
    assert window._elapsed_timer.isActive()
    assert window.sensitivity_slider.value() == _SLIDER_DEFAULT


def test_main_module_importable() -> None:
    """Entry point module imports without error."""
    import app.main  # noqa: F401
