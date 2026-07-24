"""Constant-tempo BPM estimation for MIDI export.

Detected note times remain absolute seconds. The BPM stamped into the MIDI file
controls how those seconds convert to ticks. Many DAWs then schedule ticks at
the *project* tempo, so the stamped BPM must match the track tempo or playback
speed will be wrong.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np

PathLike = Union[str, Path]

DEFAULT_BPM = 120.0
MIN_BPM = 40.0
MAX_BPM = 240.0


def normalize_bpm(bpm: float, *, default: float = DEFAULT_BPM) -> float:
    """Clamp and sanitize a user- or detector-provided BPM."""
    try:
        value = float(bpm)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value) or value <= 0:
        return float(default)
    return float(max(MIN_BPM, min(MAX_BPM, value)))


def estimate_bpm(audio_path: PathLike, *, sr: Optional[int] = None) -> float:
    """Return estimated tempo in BPM (clamped). Falls back to 120 on failure."""
    path = Path(audio_path)
    if not path.is_file():
        return DEFAULT_BPM
    try:
        y, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
        if y.size == 0:
            return DEFAULT_BPM
        tempo, _beats = librosa.beat.beat_track(y=y, sr=loaded_sr)
        # librosa may return a scalar or a length-1 ndarray depending on version.
        raw = float(np.atleast_1d(tempo)[0])
        rounded = round(raw, 1)
        return normalize_bpm(rounded)
    except Exception:  # noqa: BLE001 - never fail Convert because of tempo
        return DEFAULT_BPM
