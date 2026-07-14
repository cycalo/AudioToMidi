"""Waveform and onset review widget using pyqtgraph.

Phase 5: show the input WAV as a waveform with detected onsets overlaid as
color-coded vertical markers (one color per drum voice), so the user can review
detection before exporting. ``set_events`` is called again whenever the
sensitivity slider re-runs detection.

Pan/zoom is clamped to the loaded clip (plus a small margin). Double-click the
plot or use **Reset View** to return to the full waveform after zooming in.

Click a voice label in the legend to filter preview playback (additive toggle);
click **All** to reset the filter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.drum_voices import (  # noqa: E402
    ALL_VOICES,
    GM_NOTE_TO_VOICE,
    VOICE_ORDER,
    filter_gm_events_by_voices,
    is_all_voices,
    toggle_voice_filter,
)

Event = Tuple[float, int, int]

_VOICE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "kick": (214, 39, 40),
    "snare": (44, 160, 44),
    "toms": (31, 119, 180),
    "cymbals": (255, 127, 14),
    "hihat": (227, 199, 0),
}

# Cap the number of plotted samples so long clips stay responsive.
_MAX_DISPLAY_SAMPLES = 8000

# Pan/zoom margin around the waveform data (fraction of span, with floors).
_X_MARGIN_FRAC = 0.02
_X_MARGIN_MIN_S = 0.1
_Y_MARGIN_FRAC = 0.08
_MIN_X_ZOOM_S = 0.05  # tightest zoom-in window


class _VoiceLegendButton(QPushButton):
    """Clickable legend chip for one drum voice."""

    def __init__(self, voice: str, parent: QWidget | None = None) -> None:
        super().__init__(voice, parent)
        self.voice = voice
        r, g, b = _VOICE_COLORS[voice]
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"Click to filter preview to {voice}")
        self._color = (r, g, b)
        self._active = True
        self._refresh_style()

    def set_active(self, active: bool) -> None:
        self._active = active
        self._refresh_style()

    def _refresh_style(self) -> None:
        r, g, b = self._color
        if self._active:
            self.setStyleSheet(
                f"QPushButton {{ background-color: rgb({r}, {g}, {b}); color: #111;"
                f" border: 2px solid #fff; padding: 2px 8px; font-weight: bold; }}"
            )
        else:
            self.setStyleSheet(
                f"QPushButton {{ background-color: rgb({r}, {g}, {b}); color: #333;"
                f" border: 1px solid #666; padding: 2px 8px; opacity: 0.45; }}"
            )


class WaveformView(QWidget):
    """Waveform plot with color-coded onset markers for review."""

    seekRequested = Signal(float)
    voiceFilterChanged = Signal(object)  # FrozenSet[str]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        pg.setConfigOptions(antialias=True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._voice_buttons: Dict[str, _VoiceLegendButton] = {}
        self._all_btn: Optional[QPushButton] = None
        self._all_events: List[Event] = []
        self._voice_filter: FrozenSet[str] = ALL_VOICES

        toolbar = QHBoxLayout()
        self._build_voice_legend(toolbar)
        toolbar.addStretch(1)
        self.reset_btn = QPushButton("Reset View")
        self.reset_btn.setEnabled(False)
        self.reset_btn.setToolTip("Fit the full waveform in view (double-click the plot too).")
        self.reset_btn.clicked.connect(self.reset_view)
        toolbar.addWidget(self.reset_btn)
        layout.addLayout(toolbar)

        self._plot = pg.PlotWidget()
        self._plot.setBackground(None)
        self._plot.setLabel("bottom", "Time", units="s")
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.showGrid(x=True, y=False, alpha=0.2)
        layout.addWidget(self._plot)

        self._plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

        self._wave_item: pg.PlotDataItem | None = None
        self._marker_items: List[pg.PlotDataItem] = []
        self._playhead: pg.InfiniteLine | None = None
        self._amplitude: float = 1.0
        self._duration: Optional[float] = None
        self._seek_enabled: bool = False

    def set_seek_enabled(self, enabled: bool) -> None:
        """Allow single-click on the plot to request a playback seek."""
        self._seek_enabled = bool(enabled)

    def voice_filter(self) -> FrozenSet[str]:
        return self._voice_filter

    def set_voice_filter(self, voices: FrozenSet[str]) -> None:
        """Update the active voice filter and legend highlighting."""
        self._voice_filter = voices
        self._sync_legend_styles()
        self._redraw_markers()

    def set_waveform(self, wav_path: str) -> None:
        """Load and display the (decimated) waveform of ``wav_path``."""
        self.clear()
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        if y.size == 0:
            return
        step = max(1, y.size // _MAX_DISPLAY_SAMPLES)
        y_ds = y[::step]
        times = np.arange(y_ds.size) * (step / float(sr))
        self._amplitude = float(np.max(np.abs(y))) or 1.0
        self._duration = float(times[-1]) if times.size else 1.0

        self._wave_item = self._plot.plot(
            times, y_ds, pen=pg.mkPen((150, 150, 150), width=1)
        )
        self._apply_view_limits()
        self.reset_view()
        self.reset_btn.setEnabled(True)

    def set_events(self, events: Sequence[Event]) -> None:
        """Overlay onset markers for ``events``, respecting the voice filter."""
        self._all_events = [(float(t), int(n), int(v)) for t, n, v in events]
        self._redraw_markers()

    def set_playhead(self, time_s: Optional[float]) -> None:
        """Show or hide a vertical playhead at ``time_s`` seconds."""
        if time_s is None:
            if self._playhead is not None:
                self._playhead.setVisible(False)
            return
        if self._playhead is None:
            self._playhead = pg.InfiniteLine(
                pos=0.0,
                angle=90,
                pen=pg.mkPen((255, 255, 255), width=2),
            )
            self._plot.addItem(self._playhead)
        self._playhead.setPos(float(time_s))
        self._playhead.setVisible(True)

    def reset_view(self) -> None:
        """Zoom and pan back to the full waveform."""
        if self._duration is None:
            return
        self._plot.setXRange(0.0, self._duration, padding=0.02)
        self._plot.setYRange(-self._amplitude, self._amplitude, padding=0.05)

    def clear(self) -> None:
        """Remove the waveform and all markers."""
        self.set_playhead(None)
        if self._playhead is not None:
            self._plot.removeItem(self._playhead)
            self._playhead = None
        self._clear_markers()
        if self._wave_item is not None:
            self._plot.removeItem(self._wave_item)
            self._wave_item = None
        self._duration = None
        self._amplitude = 1.0
        self._all_events = []
        self._voice_filter = ALL_VOICES
        self._sync_legend_styles()
        self._plot.getViewBox().setLimits(
            xMin=None, xMax=None, yMin=None, yMax=None,
            minXRange=None, maxXRange=None, minYRange=None, maxYRange=None,
        )
        self.reset_btn.setEnabled(False)

    def _apply_view_limits(self) -> None:
        """Clamp pan/zoom so the view cannot drift into empty space."""
        if self._duration is None:
            return
        x_margin = max(self._duration * _X_MARGIN_FRAC, _X_MARGIN_MIN_S)
        y_margin = self._amplitude * _Y_MARGIN_FRAC
        y_span = 2.0 * self._amplitude + 2.0 * y_margin
        vb = self._plot.getViewBox()
        vb.setLimits(
            xMin=-x_margin,
            xMax=self._duration + x_margin,
            yMin=-self._amplitude - y_margin,
            yMax=self._amplitude + y_margin,
            minXRange=_MIN_X_ZOOM_S,
            maxXRange=self._duration + 2.0 * x_margin,
            minYRange=self._amplitude * 0.02,
            maxYRange=y_span,
        )

    def _build_voice_legend(self, layout: QHBoxLayout) -> None:
        """Compact clickable color key above the plot."""
        self._all_btn = QPushButton("All")
        self._all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._all_btn.setToolTip("Show and play all drum voices")
        self._all_btn.clicked.connect(self._on_all_voices_clicked)
        layout.addWidget(self._all_btn)

        for voice in VOICE_ORDER:
            r, g, b = _VOICE_COLORS[voice]
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setToolTip(voice)
            swatch.setStyleSheet(
                f"background-color: rgb({r}, {g}, {b}); border: 1px solid #888;"
            )
            layout.addWidget(swatch)
            btn = _VoiceLegendButton(voice)
            btn.clicked.connect(lambda _checked=False, v=voice: self._on_voice_clicked(v))
            self._voice_buttons[voice] = btn
            layout.addWidget(btn)
            layout.addSpacing(8)
        self._sync_legend_styles()

    def _on_voice_clicked(self, voice: str) -> None:
        new_filter = toggle_voice_filter(self._voice_filter, voice)
        if new_filter == self._voice_filter:
            return
        self._voice_filter = new_filter
        self._sync_legend_styles()
        self._redraw_markers()
        self.voiceFilterChanged.emit(self._voice_filter)

    def _on_all_voices_clicked(self) -> None:
        if is_all_voices(self._voice_filter):
            return
        self._voice_filter = ALL_VOICES
        self._sync_legend_styles()
        self._redraw_markers()
        self.voiceFilterChanged.emit(self._voice_filter)

    def _sync_legend_styles(self) -> None:
        all_active = is_all_voices(self._voice_filter)
        for voice, btn in self._voice_buttons.items():
            btn.set_active(all_active or voice in self._voice_filter)
        if self._all_btn is not None:
            if all_active:
                self._all_btn.setStyleSheet(
                    "QPushButton { font-weight: bold; border: 2px solid #fff; padding: 2px 8px; }"
                )
            else:
                self._all_btn.setStyleSheet(
                    "QPushButton { padding: 2px 8px; border: 1px solid #666; }"
                )

    def _redraw_markers(self) -> None:
        self._clear_markers()
        events = filter_gm_events_by_voices(self._all_events, self._voice_filter)
        by_voice: Dict[str, List[float]] = {v: [] for v in VOICE_ORDER}
        for time_s, note, _velocity in events:
            voice = GM_NOTE_TO_VOICE.get(int(note))
            if voice is None:
                continue
            by_voice[voice].append(float(time_s))

        top = self._amplitude
        for voice in VOICE_ORDER:
            times = by_voice[voice]
            if not times:
                continue
            xs = np.empty(len(times) * 3, dtype=float)
            ys = np.empty(len(times) * 3, dtype=float)
            xs[0::3] = times
            xs[1::3] = times
            xs[2::3] = np.nan
            ys[0::3] = -top
            ys[1::3] = top
            ys[2::3] = np.nan
            item = self._plot.plot(
                xs,
                ys,
                connect="finite",
                pen=pg.mkPen(_VOICE_COLORS[voice], width=1),
            )
            self._marker_items.append(item)

    def _on_plot_clicked(self, event) -> None:
        if event.double() and event.button() == Qt.MouseButton.LeftButton:
            self.reset_view()
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._seek_enabled
            and self._duration is not None
        ):
            pos = self._plot.getViewBox().mapSceneToView(event.scenePos())
            time_s = max(0.0, min(float(pos.x()), self._duration))
            self.seekRequested.emit(time_s)

    def _clear_markers(self) -> None:
        for item in self._marker_items:
            self._plot.removeItem(item)
        self._marker_items = []
