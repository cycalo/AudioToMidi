"""Sample-based MIDI preview rendering for in-app playback.

Loads a preview kit manifest (note -> WAV sample), renders detected MIDI events
into a mixed audio buffer, and optionally blends with the original stem. No Qt
dependency — safe to unit-test in isolation.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Literal, Optional, Sequence, Tuple

import librosa
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.drum_voices import (  # noqa: E402
    ALL_VOICES,
    STEM_NAME_BY_VOICE,
    is_all_voices,
)
from pipeline.midi_writer import Event  # noqa: E402
from pipeline.paths import repo_root as default_repo_root  # noqa: E402

logger = logging.getLogger(__name__)

PreviewMode = Literal["midi", "original", "both"]

DEFAULT_SAMPLE_RATE = 44100
VELOCITY_FLOOR = 0.08  # keep ghost notes audible in preview
PEAK_LIMIT = 0.95
MIDI_GAIN = 1.0
ORIGINAL_GAIN = 0.85


@dataclass
class PreviewKit:
    """Loaded preview samples keyed by post-remap MIDI note number."""

    name: str
    sample_rate: int
    samples: Dict[int, np.ndarray] = field(default_factory=dict)
    kit_dir: Path = field(default_factory=Path)


def resolve_kit_dir(profile: dict, *, repo_root: Optional[Path] = None) -> Path:
    """Resolve the preview kit directory from a mapping profile."""
    root = repo_root if repo_root is not None else default_repo_root()
    kit_ref = profile.get("preview_kit")
    if not kit_ref:
        raise ValueError(
            f"Mapping profile '{profile.get('plugin', '?')}' has no preview_kit path."
        )
    kit_dir = Path(kit_ref)
    if not kit_dir.is_absolute():
        kit_dir = root / kit_dir
    if not kit_dir.is_dir():
        raise FileNotFoundError(f"Preview kit directory not found: {kit_dir}")
    return kit_dir


def _load_wav_mono(path: Path, target_sr: int) -> np.ndarray:
    data, sr = sf.read(str(path), always_2d=False, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if int(sr) != target_sr:
        data = librosa.resample(data, orig_sr=int(sr), target_sr=target_sr)
    return np.asarray(data, dtype=np.float32)


def load_preview_kit(kit_dir: Path) -> PreviewKit:
    """Load ``kit.json`` and all referenced WAV samples from ``kit_dir``."""
    kit_path = Path(kit_dir)
    manifest_path = kit_path / "kit.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Preview kit manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_sr = int(manifest.get("sample_rate", DEFAULT_SAMPLE_RATE))
    raw_samples = manifest.get("samples", {})
    if not isinstance(raw_samples, dict) or not raw_samples:
        raise ValueError(f"Preview kit manifest has no samples: {manifest_path}")

    warned_missing: set[str] = set()
    samples: Dict[int, np.ndarray] = {}
    for note_str, filename in raw_samples.items():
        note = int(note_str)
        wav_path = kit_path / str(filename)
        if not wav_path.is_file():
            if filename not in warned_missing:
                logger.warning("Preview kit sample missing: %s", wav_path)
                warned_missing.add(str(filename))
            continue
        if note in samples:
            continue  # alias key points at same file; first load wins
        samples[note] = _load_wav_mono(wav_path, target_sr)

    if not samples:
        raise ValueError(f"No preview samples loaded from {kit_path}")

    return PreviewKit(
        name=str(manifest.get("name", kit_path.name)),
        sample_rate=target_sr,
        samples=samples,
        kit_dir=kit_path,
    )


def load_original_mono(wav_path: str, target_sr: int) -> np.ndarray:
    """Load the source stem as mono float32 at ``target_sr``."""
    return _load_wav_mono(Path(wav_path), target_sr)


def _longest_sample_samples(kit: PreviewKit) -> int:
    if not kit.samples:
        return 0
    return max(buf.size for buf in kit.samples.values())


def _apply_peak_limiter(buffer: np.ndarray, peak: float = PEAK_LIMIT) -> np.ndarray:
    max_val = float(np.max(np.abs(buffer))) if buffer.size else 0.0
    if max_val > peak:
        buffer = buffer * (peak / max_val)
    return buffer


def render_preview(
    events: Sequence[Event],
    kit: PreviewKit,
    *,
    duration_hint_s: Optional[float] = None,
) -> np.ndarray:
    """Mix ``events`` (post-remap note numbers) into a single audio buffer."""
    sr = kit.sample_rate
    warned_notes: set[int] = set()

    if not events:
        length = int((duration_hint_s or 0.0) * sr)
        return np.zeros(max(length, 1), dtype=np.float32)

    last_end = 0
    for time_s, note, _velocity in events:
        sample = kit.samples.get(int(note))
        if sample is None:
            if int(note) not in warned_notes:
                logger.warning("No preview sample for MIDI note %d; skipping.", int(note))
                warned_notes.add(int(note))
            continue
        start = int(float(time_s) * sr)
        last_end = max(last_end, start + sample.size)

    tail = _longest_sample_samples(kit)
    min_len = last_end
    if duration_hint_s is not None:
        min_len = max(min_len, int(float(duration_hint_s) * sr))
    out = np.zeros(max(min_len, tail, 1), dtype=np.float32)

    for time_s, note, velocity in events:
        sample = kit.samples.get(int(note))
        if sample is None:
            continue
        gain = max(VELOCITY_FLOOR, int(velocity) / 127.0)
        start = int(float(time_s) * sr)
        end = start + sample.size
        if start >= out.size:
            continue
        if end > out.size:
            chunk = sample[: out.size - start]
            out[start:] += chunk * gain
        else:
            out[start:end] += sample * gain

    return _apply_peak_limiter(out)


def _pad_or_trim(buffer: np.ndarray, length: int) -> np.ndarray:
    if buffer.size == length:
        return buffer
    if buffer.size > length:
        return buffer[:length]
    padded = np.zeros(length, dtype=np.float32)
    padded[: buffer.size] = buffer
    return padded


def mix_sources(
    midi_buf: np.ndarray,
    original_buf: Optional[np.ndarray],
    mode: PreviewMode,
    *,
    midi_gain: float = MIDI_GAIN,
    original_gain: float = ORIGINAL_GAIN,
) -> np.ndarray:
    """Combine rendered MIDI and/or original stem per ``mode``."""
    if mode == "midi":
        return _apply_peak_limiter(midi_buf * midi_gain)

    if original_buf is None:
        raise ValueError("Original audio required for 'original' or 'both' preview modes.")

    if mode == "original":
        return _apply_peak_limiter(original_buf * original_gain)

    length = max(midi_buf.size, original_buf.size)
    midi_padded = _pad_or_trim(midi_buf, length)
    orig_padded = _pad_or_trim(original_buf, length)
    mixed = midi_padded * midi_gain + orig_padded * original_gain
    return _apply_peak_limiter(mixed)


def mix_stem_wavs(
    stems_dir: Path,
    voices: FrozenSet[str],
    target_sr: int,
    *,
    duration_hint_s: Optional[float] = None,
) -> np.ndarray:
    """Sum selected separated stem WAVs from ``stems_dir``."""
    if is_all_voices(voices):
        voice_list = list(STEM_NAME_BY_VOICE.keys())
    else:
        voice_list = [v for v in STEM_NAME_BY_VOICE if v in voices]

    buffers: List[np.ndarray] = []
    for voice in voice_list:
        stem_name = STEM_NAME_BY_VOICE[voice]
        path = stems_dir / f"{stem_name}.wav"
        if not path.is_file():
            continue
        buffers.append(_load_wav_mono(path, target_sr))

    if not buffers:
        length = int((duration_hint_s or 0.0) * target_sr)
        return np.zeros(max(length, 1), dtype=np.float32)

    length = max(buf.size for buf in buffers)
    if duration_hint_s is not None:
        length = max(length, int(float(duration_hint_s) * target_sr))
    out = np.zeros(length, dtype=np.float32)
    for buf in buffers:
        out[: buf.size] += buf
    return _apply_peak_limiter(out)


def build_preview_buffer(
    events: Sequence[Event],
    kit: PreviewKit,
    *,
    wav_path: Optional[str] = None,
    mode: PreviewMode = "midi",
    duration_hint_s: Optional[float] = None,
    voices: Optional[FrozenSet[str]] = None,
    stems_dir: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """Render preview audio for playback (MIDI buffer, optional original mix)."""
    midi_buf = render_preview(events, kit, duration_hint_s=duration_hint_s)

    if mode == "midi":
        return midi_buf, kit.sample_rate

    if wav_path is None and stems_dir is None:
        raise ValueError("wav_path or stems_dir is required for original/both preview modes.")

    active_voices = voices if voices is not None else ALL_VOICES
    if not is_all_voices(active_voices) and stems_dir:
        original = mix_stem_wavs(
            Path(stems_dir),
            active_voices,
            kit.sample_rate,
            duration_hint_s=duration_hint_s,
        )
    elif wav_path is not None:
        original = load_original_mono(wav_path, kit.sample_rate)
    else:
        raise ValueError("wav_path is required when no stems_dir is available.")

    if duration_hint_s is not None:
        target_len = max(midi_buf.size, int(duration_hint_s * kit.sample_rate))
        midi_buf = _pad_or_trim(midi_buf, target_len)
        original = _pad_or_trim(original, target_len)
    else:
        length = max(midi_buf.size, original.size)
        midi_buf = _pad_or_trim(midi_buf, length)
        original = _pad_or_trim(original, length)

    return mix_sources(midi_buf, original, mode), kit.sample_rate
