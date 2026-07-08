"""Non-blocking audio playback via sounddevice for preview buffers."""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - exercised when dependency missing
    sd = None  # type: ignore[assignment]


class AudioPlayback:
    """Play a mono float32 buffer and report elapsed position."""

    def __init__(self) -> None:
        self._sample_rate: int = 44100
        self._start_time: float = 0.0
        self._duration_s: float = 0.0
        self._playing: bool = False

    def play(self, buffer: np.ndarray, sample_rate: int) -> None:
        """Start non-blocking playback of ``buffer``."""
        if sd is None:
            raise RuntimeError(
                "sounddevice is not installed. Install it to use preview playback."
            )
        self.stop()
        data = np.asarray(buffer, dtype=np.float32)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        self._sample_rate = int(sample_rate)
        self._duration_s = float(data.size) / self._sample_rate if self._sample_rate else 0.0
        sd.play(data, self._sample_rate, blocking=False)
        self._start_time = time.monotonic()
        self._playing = True

    def stop(self) -> None:
        """Stop any active playback."""
        if sd is not None:
            sd.stop()
        self._playing = False

    def is_playing(self) -> bool:
        """Return True while audio is still playing."""
        if not self._playing:
            return False
        if sd is None:
            return False
        # sounddevice exposes active stream state via sd.get_stream()
        try:
            stream = sd.get_stream()
            if stream is None or not stream.active:
                self._playing = False
                return False
        except sd.PortAudioError:
            self._playing = False
            return False
        if self.position_seconds() >= self._duration_s:
            self._playing = False
            return False
        return True

    def position_seconds(self) -> float:
        """Elapsed playback time in seconds."""
        if not self._playing:
            return 0.0
        elapsed = time.monotonic() - self._start_time
        return min(elapsed, self._duration_s)

    def duration_seconds(self) -> float:
        return self._duration_s

    def poll_finished(self, on_finished: Optional[Callable[[], None]] = None) -> bool:
        """Check playback state; invoke ``on_finished`` when playback ends."""
        if self.is_playing():
            return False
        if on_finished is not None:
            on_finished()
        return True
