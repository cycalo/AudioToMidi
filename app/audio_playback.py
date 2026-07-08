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
    """Play a mono float32 buffer with seek support."""

    def __init__(self) -> None:
        self._sample_rate: int = 44100
        self._buffer: Optional[np.ndarray] = None
        self._offset_s: float = 0.0
        self._start_time: float = 0.0
        self._duration_s: float = 0.0
        self._playing: bool = False

    def play(
        self,
        buffer: np.ndarray,
        sample_rate: int,
        *,
        start_s: float = 0.0,
    ) -> None:
        """Start non-blocking playback of ``buffer`` from ``start_s`` seconds."""
        if sd is None:
            raise RuntimeError(
                "sounddevice is not installed. Install it to use preview playback."
            )
        # Stop PortAudio output only — do not clear ``_playing`` until the new
        # stream starts, or the preview poll timer may emit a spurious finish
        # during rapid seeks.
        if sd is not None:
            try:
                sd.stop()
            except Exception:  # noqa: BLE001
                pass
        data = np.asarray(buffer, dtype=np.float32)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        self._buffer = data
        self._sample_rate = int(sample_rate)
        self._duration_s = float(data.size) / self._sample_rate if self._sample_rate else 0.0
        self._offset_s = max(0.0, min(float(start_s), self._duration_s))
        start_idx = int(self._offset_s * self._sample_rate)
        segment = data[start_idx:]
        if segment.size == 0:
            self._playing = False
            return
        sd.play(segment, self._sample_rate, blocking=False)
        self._start_time = time.monotonic()
        self._playing = True

    def stop(self) -> None:
        """Stop any active playback and reset the playhead to the start."""
        if sd is not None and self._playing:
            try:
                sd.stop()
            except Exception:  # noqa: BLE001 - PortAudio may be unavailable in tests
                pass
        self._playing = False
        self._offset_s = 0.0

    def is_playing(self) -> bool:
        """Return True while audio is still playing."""
        if not self._playing:
            return False
        if self.position_seconds() >= self._duration_s:
            self._playing = False
            return False
        if sd is None:
            return False
        # PortAudio may not report the stream active immediately after sd.play().
        if time.monotonic() - self._start_time < 0.15:
            return True
        try:
            stream = sd.get_stream()
            if stream is None or not stream.active:
                self._playing = False
                return False
        except sd.PortAudioError:
            self._playing = False
            return False
        return True

    def position_seconds(self) -> float:
        """Current playback time in seconds (absolute within the full buffer)."""
        if not self._playing:
            return self._offset_s
        elapsed = time.monotonic() - self._start_time
        return min(self._offset_s + elapsed, self._duration_s)

    def duration_seconds(self) -> float:
        return self._duration_s

    def poll_finished(self, on_finished: Optional[Callable[[], None]] = None) -> bool:
        """Check playback state; invoke ``on_finished`` when playback ends."""
        if self.is_playing():
            return False
        if on_finished is not None:
            on_finished()
        return True
