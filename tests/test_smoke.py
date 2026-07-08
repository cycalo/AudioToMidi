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
    assert window.play_btn is not None
    assert window.stop_btn is not None
    assert window.source_combo is not None
    assert window.device_combo is not None
    assert window.device_combo.count() >= 2
    # Save + sensitivity + preview are disabled until an analysis has run.
    assert not window.save_btn.isEnabled()
    assert not window.sensitivity_slider.isEnabled()
    assert not window.play_btn.isEnabled()
    assert not window.stop_btn.isEnabled()
    assert window.reset_position_btn is not None
    assert window.clear_btn is not None
    assert not window.reset_position_btn.isEnabled()
    # Convert is disabled until a WAV is chosen.
    assert not window.convert_btn.isEnabled()


def test_plugin_combo_defaults_to_ggd(qapp: QApplication) -> None:
    """GGD is listed first and selected by default for preview-focused workflow."""
    window = MainWindow()
    assert window.plugin_combo.itemData(0) == "ggd"
    assert window.controller.plugin_id == "ggd"


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


def test_waveform_view_hides_auto_range_button(qapp: QApplication) -> None:
    from app.ui.waveform_view import WaveformView

    view = WaveformView()
    assert view._plot.getPlotItem().buttonsHidden


def test_clear_all_resets_ui(qapp: QApplication) -> None:
    window = MainWindow()
    window._wav_path = "dummy.wav"
    window.path_edit.setText("dummy.wav")
    window.convert_btn.setEnabled(True)
    window._on_clear_all()
    assert window._wav_path is None
    assert window.path_edit.text() == ""
    assert not window.convert_btn.isEnabled()


def test_waveform_view_voice_filter_toggle(qapp: QApplication) -> None:
    from app.ui.waveform_view import WaveformView
    from pipeline.drum_voices import ALL_VOICES

    view = WaveformView()
    view.set_events([(0.0, 36, 100), (0.5, 38, 90)])
    view._on_voice_clicked("kick")
    assert view.voice_filter() == frozenset({"kick"})
    view._on_voice_clicked("snare")
    assert view.voice_filter() == frozenset({"kick", "snare"})
    view._on_all_voices_clicked()
    assert view.voice_filter() == ALL_VOICES


def test_transcription_combo_defaults_to_v2(qapp: QApplication) -> None:
    window = MainWindow()
    assert window.transcription_combo.itemData(0) == "v2"
    assert window.controller.transcription_version == "v2"


def test_main_module_importable() -> None:
    """Entry point module imports without error."""
    import app.main  # noqa: F401
