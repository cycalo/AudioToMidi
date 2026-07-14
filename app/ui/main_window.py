"""Main application window.

Phase 5: the full GUI. A file picker loads a drum WAV, a dropdown selects the
target plugin (annotated by mapping confidence), Convert runs the pipeline in the
background with a staged progress bar, the waveform view shows detected onsets for
review, a sensitivity slider re-runs detection live, and Save MIDI exports a
plugin-remapped ``.mid``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.controller import PipelineController  # noqa: E402
from app.ui.waveform_view import WaveformView  # noqa: E402
from pipeline.remap import available_profiles, load_profile  # noqa: E402
from pipeline.drum_voices import ALL_VOICES  # noqa: E402
from pipeline.separation import device_options  # noqa: E402

# Slider endpoints map to onset-sensitivity delta scales (see onset_detection).
_SLIDER_MIN = 0
_SLIDER_MAX = 100
_SLIDER_DEFAULT = 50
_SCALE_FEWER = 2.0  # slider far left: only the strongest hits
_SCALE_DEFAULT = 1.0
_SCALE_MORE = 0.25  # slider far right: many hits


def _slider_to_scale(value: int) -> float:
    """Map a 0-100 slider position to a delta-scale (2.0 .. 1.0 .. 0.25)."""
    if value <= _SLIDER_DEFAULT:
        frac = value / _SLIDER_DEFAULT
        return _SCALE_FEWER + frac * (_SCALE_DEFAULT - _SCALE_FEWER)
    frac = (value - _SLIDER_DEFAULT) / (_SLIDER_MAX - _SLIDER_DEFAULT)
    return _SCALE_DEFAULT + frac * (_SCALE_MORE - _SCALE_DEFAULT)


class MainWindow(QMainWindow):
    """Primary window for the Drum Stem to MIDI application."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Drum Stem to MIDI")
        self.resize(900, 640)

        self.controller = PipelineController(self)
        self._wav_path: Optional[str] = None
        self._elapsed = QElapsedTimer()
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(500)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)

        self._build_ui()
        self._connect_controller()
        self._populate_plugins()

    # -- UI construction ---------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)

        # Input row: read-only path + browse.
        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("WAV:", central))
        self.path_edit = QLineEdit(central)
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Select a drum stem WAV...")
        input_row.addWidget(self.path_edit, stretch=1)
        self.browse_btn = QPushButton("Browse...", central)
        self.browse_btn.clicked.connect(self._on_browse)
        input_row.addWidget(self.browse_btn)
        root.addLayout(input_row)

        # Plugin row.
        plugin_row = QHBoxLayout()
        plugin_row.addWidget(QLabel("Plugin:", central))
        self.plugin_combo = QComboBox(central)
        self.plugin_combo.currentIndexChanged.connect(self._on_plugin_changed)
        plugin_row.addWidget(self.plugin_combo, stretch=1)
        root.addLayout(plugin_row)

        # Compute device row (Demucs separation).
        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Device:", central))
        self.device_combo = QComboBox(central)
        self.device_combo.setToolTip(
            "Compute device for stem separation. GPU (CUDA) is only listed when "
            "PyTorch detects a compatible NVIDIA GPU."
        )
        for value, label in device_options():
            self.device_combo.addItem(label, userData=value)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        device_row.addWidget(self.device_combo, stretch=1)
        root.addLayout(device_row)

        self.warning_label = QLabel("", central)
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #b8860b;")
        self.warning_label.setVisible(False)
        root.addWidget(self.warning_label)

        # Action row.
        action_row = QHBoxLayout()
        self.convert_btn = QPushButton("Convert", central)
        self.convert_btn.setEnabled(False)
        self.convert_btn.clicked.connect(self._on_convert)
        action_row.addWidget(self.convert_btn)
        self.save_btn = QPushButton("Save MIDI...", central)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save)
        action_row.addWidget(self.save_btn)
        self.play_btn = QPushButton("Play", central)
        self.play_btn.setEnabled(False)
        self.play_btn.setToolTip(
            "Preview remapped MIDI through the GGD Preview Kit samples."
        )
        self.play_btn.clicked.connect(self._on_play)
        action_row.addWidget(self.play_btn)
        self.stop_btn = QPushButton("Stop", central)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_preview)
        action_row.addWidget(self.stop_btn)
        action_row.addWidget(QLabel("Source:", central))
        self.source_combo = QComboBox(central)
        self.source_combo.addItem("MIDI", userData="midi")
        self.source_combo.addItem("Original", userData="original")
        self.source_combo.addItem("Both", userData="both")
        self.source_combo.setEnabled(False)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        action_row.addWidget(self.source_combo)
        self.reset_position_btn = QPushButton("Reset Position", central)
        self.reset_position_btn.setEnabled(False)
        self.reset_position_btn.setToolTip("Move the preview playhead back to the start.")
        self.reset_position_btn.clicked.connect(self._on_reset_position)
        action_row.addWidget(self.reset_position_btn)
        self.clear_btn = QPushButton("Clear All", central)
        self.clear_btn.setToolTip("Stop playback and clear the loaded WAV and analysis.")
        self.clear_btn.clicked.connect(self._on_clear_all)
        action_row.addWidget(self.clear_btn)
        action_row.addStretch(1)
        root.addLayout(action_row)

        # Progress row.
        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar(central)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        progress_row.addWidget(self.progress_bar, stretch=1)
        self.elapsed_label = QLabel("", central)
        progress_row.addWidget(self.elapsed_label)
        root.addLayout(progress_row)

        self.status_label = QLabel("Load a WAV and pick a plugin to begin.", central)
        root.addWidget(self.status_label)

        # Waveform review (stretches).
        self.waveform = WaveformView(central)
        root.addWidget(self.waveform, stretch=1)

        # Sensitivity tuning row.
        sens_row = QHBoxLayout()
        sens_row.addWidget(QLabel("Fewer", central))
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal, central)
        self.sensitivity_slider.setRange(_SLIDER_MIN, _SLIDER_MAX)
        self.sensitivity_slider.setValue(_SLIDER_DEFAULT)
        self.sensitivity_slider.setEnabled(False)
        self.sensitivity_slider.setToolTip(
            "Onset sensitivity: left detects only the strongest hits, right detects "
            "more (quieter) hits and eases the bleed gate. Re-runs detection on release."
        )
        self.sensitivity_slider.sliderReleased.connect(self._on_sensitivity_changed)
        sens_row.addWidget(self.sensitivity_slider, stretch=1)
        sens_row.addWidget(QLabel("More", central))
        root.addLayout(sens_row)

        self.setCentralWidget(central)

    def _connect_controller(self) -> None:
        self.controller.progress.connect(self._on_progress)
        self.controller.analysisFinished.connect(self._on_analysis_finished)
        self.controller.analysisFailed.connect(self._on_analysis_failed)
        self.controller.eventsUpdated.connect(self._on_events_updated)
        self.controller.exportFinished.connect(self._on_export_finished)
        self.controller.exportFailed.connect(self._on_export_failed)
        self.controller.previewStarted.connect(self._on_preview_started)
        self.controller.previewFinished.connect(self._on_preview_finished)
        self.controller.previewFailed.connect(self._on_preview_failed)
        self.controller.previewPosition.connect(self.waveform.set_playhead)
        self.controller.sessionReset.connect(self._on_session_reset)
        self.waveform.seekRequested.connect(self._on_waveform_seek)
        self.waveform.voiceFilterChanged.connect(self._on_voice_filter_changed)

    def _populate_plugins(self) -> None:
        stems = [s for s in available_profiles() if s != "general_midi"]
        if "ggd" in stems:
            stems.remove("ggd")
            stems.insert(0, "ggd")
        for stem in stems:
            try:
                profile = load_profile(stem)
            except (ValueError, FileNotFoundError):
                continue
            label = self._plugin_label(profile)
            self.plugin_combo.addItem(label, userData=stem)
        if self.plugin_combo.count():
            self._on_plugin_changed(self.plugin_combo.currentIndex())
        if self.device_combo.count():
            self._on_device_changed(self.device_combo.currentIndex())

    @staticmethod
    def _plugin_label(profile: dict) -> str:
        name = profile.get("plugin", "")
        suffix = profile.get("ui_label_suffix")
        if suffix:
            return f"{name} ({suffix})"
        confidence = profile.get("confidence", "high")
        if confidence in ("medium", "low"):
            return f"{name} (mapping may need verification)"
        return name

    # -- slots: user input -------------------------------------------------
    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a drum stem WAV", "", "WAV audio (*.wav)"
        )
        if not path:
            return
        self._wav_path = path
        self.path_edit.setText(path)
        self.convert_btn.setEnabled(True)
        self.status_label.setText("Ready to convert.")

    def _on_plugin_changed(self, index: int) -> None:
        stem = self.plugin_combo.itemData(index)
        if not stem:
            return
        self.controller.set_plugin(stem)
        try:
            profile = load_profile(stem)
        except (ValueError, FileNotFoundError):
            self.warning_label.setVisible(False)
            return
        confidence = profile.get("confidence", "high")
        if confidence in ("medium", "low"):
            hint = profile.get("ui_hint") or "Mapping may need verification."
            prefix = "Note" if confidence == "medium" else "Heads up"
            self.warning_label.setText(f"{prefix}: {hint}")
            self.warning_label.setVisible(True)
        else:
            self.warning_label.setVisible(False)
        self._reset_elapsed_display()
        self._update_preview_controls()

    def _on_device_changed(self, index: int) -> None:
        device = self.device_combo.itemData(index)
        if device:
            self.controller.set_device(device)

    def _on_convert(self) -> None:
        if not self._wav_path:
            return
        self._set_busy(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._reset_sensitivity_slider()
        self._reset_elapsed_display(running=True)
        self.controller.run_analysis(self._wav_path)

    def _on_save(self) -> None:
        default_name = ""
        if self._wav_path:
            default_name = str(Path(self._wav_path).with_suffix(".mid").name)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save MIDI", default_name, "MIDI file (*.mid)"
        )
        if not path:
            return
        self.controller.export_midi(path)

    def _on_sensitivity_changed(self) -> None:
        if self.controller.state is None:
            return
        scale = _slider_to_scale(self.sensitivity_slider.value())
        self.status_label.setText("Updating onsets...")
        self._set_busy(True, keep_save=True)
        self.controller.set_delta_scale(scale)

    def _on_play(self) -> None:
        mode = self.source_combo.currentData()
        if not mode:
            return
        self.status_label.setText(f"Rendering preview ({mode})...")
        self._set_busy(True, keep_save=True, keep_preview=True)
        self.controller.preview_play(mode)

    def _on_stop_preview(self) -> None:
        self.controller.preview_stop()

    def _on_reset_position(self) -> None:
        self.controller.preview_reset_position()

    def _on_waveform_seek(self, time_s: float) -> None:
        self.controller.preview_seek(time_s)

    def _on_voice_filter_changed(self, voices: object) -> None:
        self.controller.set_voice_filter(voices)  # type: ignore[arg-type]

    def _on_source_changed(self, index: int) -> None:
        mode = self.source_combo.itemData(index)
        if not mode or self.controller.is_busy():
            return
        if self.controller.is_preview_playing() or self.controller.has_preview_cache():
            self.controller.preview_change_source(mode)

    def _on_clear_all(self) -> None:
        if self.controller.is_busy():
            return
        self.controller.reset_session()

    # -- slots: controller signals -----------------------------------------
    def _on_progress(self, message: str, percent: int) -> None:
        self.status_label.setText(message)
        self.progress_bar.setValue(percent)

    def _on_analysis_finished(self, state: object) -> None:
        self._elapsed_timer.stop()
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        self._set_busy(False)
        self.save_btn.setEnabled(True)
        self.sensitivity_slider.setEnabled(True)
        self.waveform.set_waveform(state.wav_path)
        self.waveform.set_events(state.events)
        self._update_preview_controls()
        self.status_label.setText(
            f"Detected {len(state.events)} events. Review, tune sensitivity, then Play or Save MIDI."
        )

    def _on_analysis_failed(self, error: str) -> None:
        self._elapsed_timer.stop()
        self.progress_bar.setVisible(False)
        self._set_busy(False)
        self.status_label.setText(f"Analysis failed: {error}")

    def _on_events_updated(self, events: object) -> None:
        self._set_busy(False, keep_save=True)
        self.sensitivity_slider.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.waveform.set_events(events)
        self._update_preview_controls()
        self.status_label.setText(f"Detected {len(events)} events after tuning.")

    def _on_export_finished(self, path: str) -> None:
        self.status_label.setText(f"Saved MIDI to {path}")

    def _on_export_failed(self, error: str) -> None:
        self.status_label.setText(f"Export failed: {error}")

    def _on_preview_started(self, mode: str) -> None:
        self._set_busy(False, keep_save=True)
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.reset_position_btn.setEnabled(True)
        self.waveform.set_seek_enabled(True)
        self._update_preview_controls()
        self.status_label.setText(f"Playing preview ({mode})...")

    def _on_preview_finished(self) -> None:
        self.waveform.set_playhead(None)
        self._update_preview_controls()
        if self.controller.state is not None:
            self.status_label.setText("Preview finished.")

    def _on_preview_failed(self, error: str) -> None:
        self.waveform.set_playhead(None)
        self._set_busy(False, keep_save=True)
        self._update_preview_controls()
        self.status_label.setText(f"Preview failed: {error}")

    def _on_session_reset(self) -> None:
        self._wav_path = None
        self.path_edit.clear()
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self._elapsed_timer.stop()
        self._reset_elapsed_display()
        self._reset_sensitivity_slider()
        self.warning_label.setVisible(False)
        self.waveform.clear()
        self.waveform.set_seek_enabled(False)
        self.waveform.set_voice_filter(ALL_VOICES)
        self.convert_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.sensitivity_slider.setEnabled(False)
        self._update_preview_controls()
        self.status_label.setText("Load a WAV and pick a plugin to begin.")

    # -- helpers -----------------------------------------------------------
    def _reset_elapsed_display(self, *, running: bool = False) -> None:
        """Clear the elapsed-time readout; optionally start a fresh timer."""
        self._elapsed_timer.stop()
        self._elapsed.restart()
        self.elapsed_label.setText("Elapsed: 0.0s" if running else "")
        if running:
            self._elapsed_timer.start()

    def _reset_sensitivity_slider(self) -> None:
        """Return sensitivity to tuned defaults (matches a fresh Convert run)."""
        self.sensitivity_slider.blockSignals(True)
        self.sensitivity_slider.setValue(_SLIDER_DEFAULT)
        self.sensitivity_slider.blockSignals(False)

    def _tick_elapsed(self) -> None:
        seconds = self._elapsed.elapsed() / 1000.0
        self.elapsed_label.setText(f"Elapsed: {seconds:0.1f}s")

    def _update_preview_controls(self) -> None:
        has_state = self.controller.state is not None
        preview_ok = has_state and self.controller.preview_supported()
        playing = self.controller.is_preview_playing()
        busy = self.controller.is_busy()

        if not preview_ok:
            self.play_btn.setEnabled(False)
            if self.controller.plugin_id != "ggd":
                self.play_btn.setToolTip(
                    "Preview is available for GetGood Drums only (v1)."
                )
            else:
                self.play_btn.setToolTip(
                    "Preview remapped MIDI through the GGD Preview Kit samples."
                )
        else:
            self.play_btn.setEnabled(has_state and not busy and not playing)
            self.play_btn.setToolTip(
                "Preview remapped MIDI through the GGD Preview Kit samples."
            )

        self.stop_btn.setEnabled(playing)
        self.reset_position_btn.setEnabled(
            has_state and preview_ok and (playing or self.controller.has_preview_cache())
        )
        self.waveform.set_seek_enabled(
            has_state and preview_ok and (playing or self.controller.has_preview_cache())
        )
        self.source_combo.setEnabled(
            (has_state and preview_ok and not busy) or playing
        )

    def _set_busy(self, busy: bool, *, keep_save: bool = False, keep_preview: bool = False) -> None:
        self.convert_btn.setEnabled(not busy and self._wav_path is not None)
        self.browse_btn.setEnabled(not busy)
        self.plugin_combo.setEnabled(not busy)
        self.device_combo.setEnabled(not busy)
        self.sensitivity_slider.setEnabled(not busy and self.controller.state is not None)
        if not keep_save:
            self.save_btn.setEnabled(not busy and self.controller.state is not None)
        if not keep_preview:
            self._update_preview_controls()
        elif busy:
            self.play_btn.setEnabled(False)
            if not self.controller.is_preview_playing():
                self.source_combo.setEnabled(False)
