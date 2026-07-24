"""HitMap main window — left control rail + waveform stage."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtGui import QAction, QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.controller import PipelineController  # noqa: E402
from app.log_buffer import get_log_buffer  # noqa: E402
from app.ui.debug_log_dialog import DebugLogDialog  # noqa: E402
from app.ui.theme import app_stylesheet  # noqa: E402
from app.ui.waveform_view import WaveformView  # noqa: E402
from pipeline.remap import available_profiles, load_profile  # noqa: E402
from pipeline.drum_voices import ALL_VOICES  # noqa: E402
from pipeline.separation import device_options  # noqa: E402

_SLIDER_MIN = 0
_SLIDER_MAX = 100
_SLIDER_DEFAULT = 50
_SCALE_FEWER = 2.0
_SCALE_DEFAULT = 1.0
_SCALE_MORE = 0.25
_RAIL_WIDTH = 280
_STAGE_MIN_WIDTH = 964  # ~1280px total with rail + margins
_WINDOW_CHROME_PAD_H = 48
_DEFAULT_WINDOW_WIDTH = 1280


def _slider_to_scale(value: int) -> float:
    """Map a 0-100 slider position to a delta-scale (2.0 .. 1.0 .. 0.25)."""
    if value <= _SLIDER_DEFAULT:
        frac = value / _SLIDER_DEFAULT
        return _SCALE_FEWER + frac * (_SCALE_DEFAULT - _SCALE_FEWER)
    frac = (value - _SLIDER_DEFAULT) / (_SLIDER_MAX - _SLIDER_DEFAULT)
    return _SCALE_DEFAULT + frac * (_SCALE_MORE - _SCALE_DEFAULT)


def _section_label(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setObjectName("sectionLabel")
    return label


class MainWindow(QMainWindow):
    """HitMap primary window: rail controls + waveform stage."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HitMap")
        self.setStyleSheet(app_stylesheet())

        self.controller = PipelineController(self)
        self._wav_path: Optional[str] = None
        self._rail_body: Optional[QWidget] = None
        self._rail_scroll: Optional[QScrollArea] = None
        self._initial_fit_done = False
        self._elapsed = QElapsedTimer()
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(500)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)

        self._build_ui()
        self._build_menu()
        self._connect_controller()
        self._populate_plugins()
        self._set_session_meta("Load a drum stem to begin")

    def _build_menu(self) -> None:
        help_menu = self.menuBar().addMenu("&Help")
        self._view_debug_log_action = QAction("View debug log…", self)
        self._view_debug_log_action.triggered.connect(self._on_view_debug_log)
        help_menu.addAction(self._view_debug_log_action)

    def _on_view_debug_log(self) -> None:
        dialog = DebugLogDialog(self)
        dialog.exec()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._initial_fit_done:
            self._initial_fit_done = True
            QTimer.singleShot(0, self._fit_window_to_content)

    def _fit_window_to_content(self) -> None:
        """Resize on first show so the rail fits without scrolling."""
        if self._rail_body is None:
            return
        self._rail_body.adjustSize()
        layout = self.centralWidget().layout()
        margins = layout.contentsMargins() if layout is not None else None
        margin_h = (margins.top() + margins.bottom()) if margins is not None else 24
        margin_w = (margins.left() + margins.right()) if margins is not None else 24
        spacing = layout.spacing() if layout is not None else 12

        rail_h = self._rail_body.sizeHint().height()
        win_h = rail_h + margin_h + _WINDOW_CHROME_PAD_H
        win_w = max(
            _DEFAULT_WINDOW_WIDTH,
            _RAIL_WIDTH + _STAGE_MIN_WIDTH + margin_w + spacing,
        )

        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            win_w = min(win_w, avail.width())
            win_h = min(win_h, avail.height())

        self.setMinimumSize(min(win_w, 1000), min(win_h, 560))
        self.resize(win_w, win_h)

        if self._rail_scroll is not None:
            self._rail_scroll.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )

    # -- UI construction ---------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 12, 12)
        root.setSpacing(12)

        rail = self._build_rail(central)
        root.addWidget(rail)

        stage = self._build_stage(central)
        root.addWidget(stage, stretch=1)

        self.setCentralWidget(central)

    def _build_rail(self, parent: QWidget) -> QWidget:
        rail_frame = QFrame(parent)
        rail_frame.setObjectName("railPanel")
        rail_frame.setFixedWidth(_RAIL_WIDTH)

        outer = QVBoxLayout(rail_frame)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea(rail_frame)
        self._rail_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        body = QWidget(scroll)
        self._rail_body = body
        layout = QVBoxLayout(body)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(8)

        brand = QLabel("HitMap", body)
        brand.setObjectName("brandTitle")
        layout.addWidget(brand)

        tagline = QLabel("Drum stem → MIDI", body)
        tagline.setObjectName("brandTagline")
        layout.addWidget(tagline)

        self.session_meta = QLabel("", body)
        self.session_meta.setObjectName("sessionMeta")
        self.session_meta.setWordWrap(True)
        layout.addWidget(self.session_meta)

        layout.addWidget(_section_label("Source", body))

        self.path_edit = QLineEdit(body)
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Drop WAV or Browse…")
        layout.addWidget(self.path_edit)

        self.browse_btn = QPushButton("Browse…", body)
        self.browse_btn.setObjectName("ghostButton")
        self.browse_btn.clicked.connect(self._on_browse)
        layout.addWidget(self.browse_btn)

        self.plugin_combo = QComboBox(body)
        self.plugin_combo.currentIndexChanged.connect(self._on_plugin_changed)
        layout.addWidget(self.plugin_combo)

        layout.addWidget(_section_label("Device", body))
        self.device_combo = QComboBox(body)
        self.device_combo.setToolTip(
            "Compute device for stem separation. Auto uses the GPU when CUDA is "
            "available, otherwise CPU."
        )
        for value, label in device_options():
            self.device_combo.addItem(label, userData=value)
        # device_options() lists auto first — keep that as the default.
        self.device_combo.setCurrentIndex(0)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        layout.addWidget(self.device_combo)

        self.warning_label = QLabel("", body)
        self.warning_label.setObjectName("warningLabel")
        self.warning_label.setWordWrap(True)
        self.warning_label.setVisible(False)
        layout.addWidget(self.warning_label)

        self.convert_btn = QPushButton("Convert", body)
        self.convert_btn.setObjectName("primaryButton")
        self.convert_btn.setEnabled(False)
        self.convert_btn.clicked.connect(self._on_convert)
        layout.addWidget(self.convert_btn)

        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar(body)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        progress_row.addWidget(self.progress_bar, stretch=1)
        self.elapsed_label = QLabel("", body)
        self.elapsed_label.setObjectName("statusLabel")
        progress_row.addWidget(self.elapsed_label)
        layout.addLayout(progress_row)

        self.status_label = QLabel("Load a WAV and pick a plugin to begin.", body)
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addWidget(_section_label("Review", body))

        sens_header = QHBoxLayout()
        fewer = QLabel("Fewer", body)
        fewer.setObjectName("statusLabel")
        more = QLabel("More", body)
        more.setObjectName("statusLabel")
        sens_header.addWidget(fewer)
        sens_header.addStretch(1)
        sens_header.addWidget(more)
        layout.addLayout(sens_header)

        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal, body)
        self.sensitivity_slider.setRange(_SLIDER_MIN, _SLIDER_MAX)
        self.sensitivity_slider.setValue(_SLIDER_DEFAULT)
        self.sensitivity_slider.setEnabled(False)
        self.sensitivity_slider.setToolTip(
            "Onset sensitivity: left detects only the strongest hits, right detects "
            "more (quieter) hits and eases the bleed gate. Re-runs detection on release."
        )
        self.sensitivity_slider.sliderReleased.connect(self._on_sensitivity_changed)
        layout.addWidget(self.sensitivity_slider)

        bpm_row = QHBoxLayout()
        bpm_label = QLabel("BPM", body)
        bpm_label.setObjectName("statusLabel")
        bpm_row.addWidget(bpm_label)
        self.bpm_spin = QSpinBox(body)
        self.bpm_spin.setRange(40, 240)
        self.bpm_spin.setSingleStep(1)
        self.bpm_spin.setValue(120)
        self.bpm_spin.setEnabled(False)
        self.bpm_spin.setToolTip(
            "Tempo stamped into the MIDI file. Must match your DAW project BPM so "
            "playback speed matches the drum track. Auto-detected from the audio; "
            "edit if the estimate is wrong."
        )
        bpm_row.addWidget(self.bpm_spin, stretch=1)
        layout.addLayout(bpm_row)

        layout.addWidget(_section_label("Preview", body))
        preview_hint = QLabel("Source ▾", body)
        preview_hint.setObjectName("comboHintLabel")
        preview_hint.setToolTip("Choose what to hear during preview")
        layout.addWidget(preview_hint)

        self.source_combo = QComboBox(body)
        self.source_combo.setObjectName("previewCombo")
        self.source_combo.addItem("MIDI", userData="midi")
        self.source_combo.addItem("Original", userData="original")
        self.source_combo.addItem("Both", userData="both")
        self.source_combo.setEnabled(False)
        self.source_combo.setToolTip(
            "Choose what to hear during preview: remapped MIDI, original audio, or both."
        )
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        layout.addWidget(self.source_combo)

        preview_row = QHBoxLayout()
        self.play_btn = QPushButton("Play", body)
        self.play_btn.setObjectName("transportButton")
        self.play_btn.setEnabled(False)
        self.play_btn.setToolTip(
            "Preview remapped MIDI through the GGD Preview Kit samples."
        )
        self.play_btn.clicked.connect(self._on_play)
        preview_row.addWidget(self.play_btn)
        self.stop_btn = QPushButton("Stop", body)
        self.stop_btn.setObjectName("transportButton")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_preview)
        preview_row.addWidget(self.stop_btn)
        layout.addLayout(preview_row)

        self.reset_position_btn = QPushButton("Reset Position", body)
        self.reset_position_btn.setEnabled(False)
        self.reset_position_btn.setToolTip("Move the preview playhead back to the start.")
        self.reset_position_btn.clicked.connect(self._on_reset_position)
        layout.addWidget(self.reset_position_btn)

        self.save_btn = QPushButton("Save MIDI", body)
        self.save_btn.setObjectName("secondaryButton")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save)
        layout.addWidget(self.save_btn)

        self.clear_btn = QPushButton("Clear All", body)
        self.clear_btn.setToolTip("Stop playback and clear the loaded WAV and analysis.")
        self.clear_btn.clicked.connect(self._on_clear_all)
        layout.addWidget(self.clear_btn)

        scroll.setWidget(body)
        outer.addWidget(scroll)
        return rail_frame

    def _build_stage(self, parent: QWidget) -> QWidget:
        stage = QFrame(parent)
        stage.setObjectName("stagePanel")
        layout = QVBoxLayout(stage)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(0)
        self.waveform = WaveformView(stage)
        layout.addWidget(self.waveform, stretch=1)
        return stage

    def _set_session_meta(self, text: str) -> None:
        self.session_meta.setText(text)

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
        else:
            self.status_label.setText(
                "No plugin mappings found. Rebuild the app so mappings/ is bundled."
            )
            self.warning_label.setText(
                "Plugin profiles missing — remapping and GGD preview unavailable."
            )
            self.warning_label.setVisible(True)
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
        self.path_edit.setText(Path(path).name)
        self.path_edit.setToolTip(path)
        self.convert_btn.setEnabled(True)
        self.status_label.setText("Ready to convert.")
        self._set_session_meta(Path(path).name)

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
        self.controller.export_midi(path, tempo=float(self.bpm_spin.value()))

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
        get_log_buffer().append(f"[{percent}%] {message}")

    def _on_analysis_finished(self, state: object) -> None:
        self._elapsed_timer.stop()
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        self._set_busy(False)
        self.save_btn.setEnabled(True)
        self.sensitivity_slider.setEnabled(True)
        self.bpm_spin.blockSignals(True)
        self.bpm_spin.setValue(int(round(float(getattr(state, "detected_bpm", 120.0)))))
        self.bpm_spin.blockSignals(False)
        self.bpm_spin.setEnabled(True)
        self.waveform.set_waveform(state.wav_path)
        self.waveform.set_events(state.events)
        self._update_preview_controls()
        bpm = int(self.bpm_spin.value())
        n = len(state.events)
        self._set_session_meta(f"{n} hits · {bpm} BPM")
        self.status_label.setText(
            f"Detected {n} events · BPM {bpm} (editable). "
            "Review, then Play or Save MIDI."
        )

    def _on_analysis_failed(self, error: str) -> None:
        self._elapsed_timer.stop()
        self.progress_bar.setVisible(False)
        self._set_busy(False)
        self.status_label.setText(f"Analysis failed: {error}")
        get_log_buffer().append(f"ERROR analysis: {error}")

    def _on_events_updated(self, events: object) -> None:
        self._set_busy(False, keep_save=True)
        self.sensitivity_slider.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.waveform.set_events(events)
        self._update_preview_controls()
        bpm = int(self.bpm_spin.value())
        self._set_session_meta(f"{len(events)} hits · {bpm} BPM")
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
        self.path_edit.setToolTip("")
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
        self.bpm_spin.blockSignals(True)
        self.bpm_spin.setValue(120)
        self.bpm_spin.blockSignals(False)
        self.bpm_spin.setEnabled(False)
        self._update_preview_controls()
        self._set_session_meta("Load a drum stem to begin")
        self.status_label.setText("Load a WAV and pick a plugin to begin.")

    # -- helpers -----------------------------------------------------------
    def _reset_elapsed_display(self, *, running: bool = False) -> None:
        self._elapsed_timer.stop()
        self._elapsed.restart()
        self.elapsed_label.setText("0.0s" if running else "")
        if running:
            self._elapsed_timer.start()

    def _reset_sensitivity_slider(self) -> None:
        self.sensitivity_slider.blockSignals(True)
        self.sensitivity_slider.setValue(_SLIDER_DEFAULT)
        self.sensitivity_slider.blockSignals(False)

    def _tick_elapsed(self) -> None:
        seconds = self._elapsed.elapsed() / 1000.0
        self.elapsed_label.setText(f"{seconds:0.1f}s")

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

    def _set_busy(
        self, busy: bool, *, keep_save: bool = False, keep_preview: bool = False
    ) -> None:
        self.convert_btn.setEnabled(not busy and self._wav_path is not None)
        self.browse_btn.setEnabled(not busy)
        self.plugin_combo.setEnabled(not busy)
        self.device_combo.setEnabled(not busy)
        self.sensitivity_slider.setEnabled(not busy and self.controller.state is not None)
        self.bpm_spin.setEnabled(not busy and self.controller.state is not None)
        self.clear_btn.setEnabled(not busy)
        if not keep_save:
            self.save_btn.setEnabled(not busy and self.controller.state is not None)
        if not keep_preview:
            self._update_preview_controls()
        elif busy:
            self.play_btn.setEnabled(False)
            if not self.controller.is_preview_playing():
                self.source_combo.setEnabled(False)
