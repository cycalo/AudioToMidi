"""Waveform and onset review widget using pyqtgraph.

Phase 5: show the input WAV as a waveform with detected onsets overlaid as
color-coded vertical markers (one color per drum voice), so the user can review
detection before exporting. ``set_events`` is called again whenever the
sensitivity slider re-runs detection.

Pan/zoom is clamped to the loaded clip (plus a small margin). Double-click the
plot or use **Reset View** to return to the full waveform after zooming in.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

Event = Tuple[float, int, int]

# Map each General MIDI note the pipeline emits to a drum voice + display color.
_NOTE_TO_VOICE: Dict[int, str] = {
    36: "kick",
    38: "snare",
    45: "toms",
    47: "toms",
    50: "toms",
    49: "cymbals",
    42: "hihat",
}
_VOICE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "kick": (214, 39, 40),
    "snare": (44, 160, 44),
    "toms": (31, 119, 180),
    "cymbals": (255, 127, 14),
    "hihat": (227, 199, 0),
}
_VOICE_ORDER = ("kick", "snare", "toms", "cymbals", "hihat")

# Cap the number of plotted samples so long clips stay responsive.
_MAX_DISPLAY_SAMPLES = 8000

# Pan/zoom margin around the waveform data (fraction of span, with floors).
_X_MARGIN_FRAC = 0.02
_X_MARGIN_MIN_S = 0.1
_Y_MARGIN_FRAC = 0.08
_MIN_X_ZOOM_S = 0.05  # tightest zoom-in window


class WaveformView(QWidget):
    """Waveform plot with color-coded onset markers for review."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        pg.setConfigOptions(antialias=True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

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
        self._plot.showGrid(x=True, y=False, alpha=0.2)
        layout.addWidget(self._plot)

        self._plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

        self._wave_item: pg.PlotDataItem | None = None
        self._marker_items: List[pg.PlotDataItem] = []
        self._amplitude: float = 1.0
        self._duration: Optional[float] = None

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
        """Overlay onset markers for ``events``, replacing any previous markers."""
        self._clear_markers()
        by_voice: Dict[str, List[float]] = {v: [] for v in _VOICE_ORDER}
        for time_s, note, _velocity in events:
            voice = _NOTE_TO_VOICE.get(int(note), "cymbals")
            by_voice[voice].append(float(time_s))

        top = self._amplitude
        for voice in _VOICE_ORDER:
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

    def reset_view(self) -> None:
        """Zoom and pan back to the full waveform."""
        if self._duration is None:
            return
        self._plot.setXRange(0.0, self._duration, padding=0.02)
        self._plot.setYRange(-self._amplitude, self._amplitude, padding=0.05)

    def clear(self) -> None:
        """Remove the waveform and all markers."""
        self._clear_markers()
        if self._wave_item is not None:
            self._plot.removeItem(self._wave_item)
            self._wave_item = None
        self._duration = None
        self._amplitude = 1.0
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
        """Compact color key above the plot (keeps the waveform area unobstructed)."""
        for voice in ("kick", "snare", "toms", "cymbals"):
            r, g, b = _VOICE_COLORS[voice]
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setToolTip(voice)
            swatch.setStyleSheet(
                f"background-color: rgb({r}, {g}, {b}); border: 1px solid #888;"
            )
            layout.addWidget(swatch)
            layout.addWidget(QLabel(voice))
            layout.addSpacing(12)

    def _on_plot_clicked(self, event) -> None:
        if event.double() and event.button() == Qt.MouseButton.LeftButton:
            self.reset_view()

    def _clear_markers(self) -> None:
        for item in self._marker_items:
            self._plot.removeItem(item)
        self._marker_items = []
