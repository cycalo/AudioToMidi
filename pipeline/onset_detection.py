"""Per-stem onset detection, velocity extraction, and tom pitch clustering.

Phase 1: detect onsets in a single pre-isolated drum voice (e.g. a solo kick
loop) with ``librosa`` and assign each hit a MIDI velocity by normalizing the
per-onset peak amplitude against the loudest/quietest hits in the same file.
Runnable as a CLI that writes a General MIDI ``.mid``.

Phase 3: per-stem tuned onset detection (band-limited envelopes + per-voice
peak-picking thresholds) and adaptive tom pitch clustering (choose 2 or 3
clusters per file via silhouette comparison, mapped to low/mid/high tom notes).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np

# Ensure repo root is importable when run directly (``python pipeline/onset_detection.py``).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.midi_writer import DEFAULT_TEMPO, write_midi  # noqa: E402

# General MIDI kick drum.
DEFAULT_NOTE = 36
DEFAULT_HOP_LENGTH = 512
DEFAULT_WINDOW_MS = 20.0
# Musical velocity floor: the quietest detected hit maps here rather than to 1,
# since very low velocities are near-inaudible in most drum plugins. Tunable.
MIN_VELOCITY = 20
MAX_VELOCITY = 127

Event = Tuple[float, int, int]

# General MIDI tom notes (low, mid, high), per Section 5.1.
TOM_NOTES = (45, 47, 50)

# Tom pitch estimation window: skip the broadband attack click, then measure a
# short sustain segment. Range brackets typical tom fundamentals.
TOM_PITCH_SKIP_MS = 12.0
TOM_PITCH_WIN_MS = 80.0
TOM_PITCH_FMIN = 60.0
TOM_PITCH_FMAX = 400.0

# Adaptive k selection: prefer k=2 unless k=3's silhouette beats it by the
# margin, and collapse to a single tom if the best split is too weak.
SILHOUETTE_MARGIN = 0.05
SILHOUETTE_FLOOR = 0.5


@dataclass(frozen=True)
class StemPreset:
    """Per-stem onset detection parameters (Section 1.1)."""

    name: str
    fmin: float  # onset-envelope band low edge (Hz)
    fmax: float  # onset-envelope band high edge (Hz)
    delta: float  # peak-picking threshold (higher = fewer, stronger onsets)
    wait_ms: float  # min spacing enforced during peak-picking
    min_ioi_ms: float  # min inter-onset interval enforced later in the merge step
    gm_note: Optional[int]  # fixed GM note, or None for toms (assigned by clustering)


# Kick/snare are clean fast transients that tolerate a low threshold; cymbals are
# noisy and sustained, so they need a higher threshold and longer wait to avoid
# retriggering across the decay wash. Bands reject cross-stem bleed.
STEM_PRESETS: Dict[str, StemPreset] = {
    "kick": StemPreset("kick", 20.0, 200.0, 0.07, 30.0, 30.0, 36),
    "snare": StemPreset("snare", 150.0, 5000.0, 0.06, 30.0, 30.0, 38),
    "toms": StemPreset("toms", 60.0, 500.0, 0.07, 45.0, 45.0, None),
    "cymbals": StemPreset("cymbals", 3000.0, 16000.0, 0.12, 70.0, 70.0, 49),
    # Hi-hat only exists on the DSP separation path; secondary target for now.
    "hihat": StemPreset("hihat", 6000.0, 12000.0, 0.10, 60.0, 60.0, 42),
}


def load_audio(
    path: str, *, sr: Optional[int] = None, mono: bool = True
) -> Tuple[np.ndarray, int]:
    """Load an audio file, preserving native sample rate by default."""
    y, sr = librosa.load(path, sr=sr, mono=mono)
    return y, int(sr)


def detect_onsets(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = DEFAULT_HOP_LENGTH,
    backtrack: bool = True,
    delta: float = 0.07,
) -> np.ndarray:
    """Detect onset times (seconds) in an isolated drum voice.

    ``backtrack`` shifts each detected onset back to the nearest preceding
    energy minimum, which lines the event up with the true attack. ``delta`` is
    the peak-picking threshold; higher values suppress spurious re-triggers from
    decay/ringing.
    """
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=hop_length,
        backtrack=backtrack,
        delta=delta,
        units="frames",
    )
    return librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)


def extract_velocities(
    y: np.ndarray,
    sr: int,
    onset_times: Sequence[float],
    *,
    window_ms: float = DEFAULT_WINDOW_MS,
    velocity_min: int = MIN_VELOCITY,
    velocity_max: int = MAX_VELOCITY,
) -> np.ndarray:
    """Map each onset to a MIDI velocity via per-file peak-amplitude normalization.

    For each onset the peak absolute amplitude in a short window starting at the
    onset is measured, then linearly rescaled so the quietest hit in the file
    maps to ``velocity_min`` and the loudest to ``velocity_max``.
    """
    onset_times = np.asarray(onset_times, dtype=float)
    if onset_times.size == 0:
        return np.array([], dtype=int)

    window_samples = max(1, int(sr * window_ms / 1000.0))
    peaks = np.empty(onset_times.size, dtype=float)
    for i, t in enumerate(onset_times):
        start = max(0, int(t * sr))
        end = min(len(y), start + window_samples)
        segment = y[start:end]
        peaks[i] = float(np.max(np.abs(segment))) if segment.size else 0.0

    lo = float(peaks.min())
    hi = float(peaks.max())
    if hi <= lo:
        # All hits equally loud (or a single hit): use the top of the range.
        velocities = np.full(peaks.shape, velocity_max, dtype=float)
    else:
        norm = (peaks - lo) / (hi - lo)
        velocities = velocity_min + norm * (velocity_max - velocity_min)

    return np.clip(np.rint(velocities), velocity_min, velocity_max).astype(int)


# --------------------------------------------------------------------------
# Phase 3: per-stem tuned onset detection
# --------------------------------------------------------------------------
def _band_onset_envelope(
    y: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    hop_length: int,
    *,
    n_fft: int = 2048,
) -> np.ndarray:
    """Onset-strength envelope from an STFT restricted to the [fmin, fmax] band.

    A direct STFT band slice is used rather than a mel spectrogram because narrow
    drum bands (e.g. the kick's 20-200 Hz) span too few FFT bins to populate many
    mel filters, which produces empty-filter artifacts.
    """
    fmax = float(min(fmax, sr / 2.0))
    magnitude = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        band = np.ones_like(freqs, dtype=bool)
    band_db = librosa.power_to_db(magnitude[band, :] ** 2, ref=np.max)
    return librosa.onset.onset_strength(S=band_db, sr=sr, hop_length=hop_length)


def detect_onsets_for_stem(
    y: np.ndarray,
    sr: int,
    preset: StemPreset,
    *,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    """Detect onset times (seconds) in one drum stem using its tuned preset."""
    if y.size == 0:
        return np.array([], dtype=float)
    envelope = _band_onset_envelope(y, sr, preset.fmin, preset.fmax, hop_length)
    wait = max(1, int(round(preset.wait_ms / 1000.0 * sr / hop_length)))
    frames = librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=sr,
        hop_length=hop_length,
        delta=preset.delta,
        wait=wait,
        backtrack=True,
        units="frames",
    )
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)


# --------------------------------------------------------------------------
# Phase 3: tom pitch estimation + adaptive clustering
# --------------------------------------------------------------------------
def _hz_to_semitones(hz: float) -> float:
    """Convert a frequency in Hz to a MIDI-style semitone value (log scale)."""
    return 12.0 * np.log2(hz / 440.0) + 69.0


def _band_fft_peak_hz(seg: np.ndarray, sr: int, fmin: float, fmax: float) -> Optional[float]:
    """Dominant FFT bin frequency within a band; fallback pitch estimate."""
    if seg.size == 0:
        return None
    windowed = seg * np.hanning(seg.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(seg.size, 1.0 / sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any() or float(spectrum[band].max()) <= 0.0:
        return None
    return float(freqs[band][int(np.argmax(spectrum[band]))])


def estimate_tom_pitches(
    y: np.ndarray,
    sr: int,
    onset_times: Sequence[float],
    *,
    skip_ms: float = TOM_PITCH_SKIP_MS,
    win_ms: float = TOM_PITCH_WIN_MS,
) -> np.ndarray:
    """Estimate a per-onset pitch (in semitones) for tom hits.

    For each onset, a short window starting ``skip_ms`` after the onset (skipping
    the broadband attack click) is analyzed with ``librosa.pyin``; the median
    voiced f0 is returned in semitones. If pyin finds nothing voiced, a
    band-limited FFT peak is used as a fallback. NaN means no usable estimate.
    """
    onset_times = np.asarray(onset_times, dtype=float)
    out = np.full(onset_times.size, np.nan, dtype=float)
    if onset_times.size == 0:
        return out

    frame_length = 2048
    win_samples = max(int(win_ms / 1000.0 * sr), frame_length)
    for i, t in enumerate(onset_times):
        start = max(0, int((t + skip_ms / 1000.0) * sr))
        seg = y[start : start + win_samples]
        if seg.size < frame_length:
            seg = np.pad(seg, (0, frame_length - seg.size))
        vals = np.array([], dtype=float)
        try:
            f0, _, _ = librosa.pyin(
                seg, sr=sr, fmin=TOM_PITCH_FMIN, fmax=TOM_PITCH_FMAX
            )
            vals = f0[np.isfinite(f0)]
        except Exception:
            vals = np.array([], dtype=float)
        if vals.size:
            out[i] = _hz_to_semitones(float(np.median(vals)))
        else:
            peak = _band_fft_peak_hz(seg, sr, TOM_PITCH_FMIN, TOM_PITCH_FMAX)
            if peak:
                out[i] = _hz_to_semitones(peak)
    return out


def cluster_toms(pitches: Sequence[float]) -> Tuple[List[int], int]:
    """Adaptively cluster tom pitches into low/mid/high notes.

    Runs k-means at k=2 and (when there are enough hits) k=3 over the usable
    per-onset pitches, compares silhouette scores, and picks the k that best fits
    the file's pitch spread (preferring k=2 unless k=3 wins by ``SILHOUETTE_MARGIN``,
    and collapsing to a single tom when the best split is weaker than
    ``SILHOUETTE_FLOOR`` or there is only one usable pitch). Clusters are mapped by
    ascending center pitch to GM tom notes: k=3 -> 45/47/50, k=2 -> 45/50, k=1 -> 47.

    Returns ``(notes_per_onset, k)`` where ``notes_per_onset`` aligns with the input.
    """
    pitches = np.asarray(pitches, dtype=float)
    n = pitches.size
    notes = np.full(n, 47, dtype=int)
    if n == 0:
        return notes.tolist(), 0

    finite = np.isfinite(pitches)
    usable = pitches[finite]
    if usable.size == 0:
        return notes.tolist(), 0

    spread = float(usable.max() - usable.min())
    if usable.size == 1 or spread < 1.0:
        return notes.tolist(), 1

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    features = usable.reshape(-1, 1)
    scores: Dict[int, float] = {}
    fitted: Dict[int, "KMeans"] = {}
    candidates = [2] + ([3] if usable.size >= 4 else [])
    for k in candidates:
        model = KMeans(n_clusters=k, n_init=10, random_state=0).fit(features)
        if len(set(model.labels_)) < k:
            continue
        scores[k] = float(silhouette_score(features, model.labels_))
        fitted[k] = model

    if not scores:
        return notes.tolist(), 1

    if 2 in scores and 3 in scores:
        chosen = 3 if scores[3] >= scores[2] + SILHOUETTE_MARGIN else 2
    else:
        chosen = max(scores, key=scores.get)

    if scores[chosen] < SILHOUETTE_FLOOR:
        return notes.tolist(), 1

    model = fitted[chosen]
    centers = model.cluster_centers_.ravel()
    order = list(np.argsort(centers))  # cluster labels sorted by ascending pitch
    rank_of_label = {label: rank for rank, label in enumerate(order)}
    note_by_rank = {2: [45, 50], 3: [45, 47, 50]}[chosen]
    usable_notes = [note_by_rank[rank_of_label[label]] for label in model.labels_]

    fill = max(set(usable_notes), key=usable_notes.count)
    result: List[int] = []
    ui = 0
    for is_finite in finite:
        if is_finite:
            result.append(usable_notes[ui])
            ui += 1
        else:
            result.append(fill)
    return result, chosen


def assign_tom_notes(
    y: np.ndarray, sr: int, onset_times: Sequence[float]
) -> Tuple[List[int], int]:
    """Estimate tom pitches then cluster them into GM tom notes. Returns (notes, k)."""
    pitches = estimate_tom_pitches(y, sr, onset_times)
    return cluster_toms(pitches)


def min_ioi_by_note() -> Dict[int, float]:
    """Map each GM note this pipeline emits to its per-voice min inter-onset (ms)."""
    mapping: Dict[int, float] = {}
    for preset in STEM_PRESETS.values():
        if preset.name == "toms":
            for note in TOM_NOTES:
                mapping[note] = preset.min_ioi_ms
        elif preset.gm_note is not None:
            mapping[preset.gm_note] = preset.min_ioi_ms
    return mapping


def detect_stem_events(
    path: str,
    stem_name: str,
    *,
    window_ms: float = DEFAULT_WINDOW_MS,
) -> Tuple[List[Event], dict]:
    """Detect events for one separated stem, assigning GM notes.

    For toms, notes are assigned by adaptive pitch clustering; for other stems the
    preset's fixed GM note is used. Returns ``(events, info)`` where ``info`` holds
    the stem name, onset count, and (for toms) the chosen cluster count ``tom_k``.
    """
    preset = STEM_PRESETS.get(stem_name)
    if preset is None:
        return [], {"stem": stem_name, "onsets": 0, "tom_k": None, "skipped": True}

    y, sr = load_audio(path)
    onset_times = detect_onsets_for_stem(y, sr, preset)
    velocities = extract_velocities(y, sr, onset_times, window_ms=window_ms)

    if stem_name == "toms":
        notes, tom_k = assign_tom_notes(y, sr, onset_times)
    else:
        notes = [int(preset.gm_note)] * len(onset_times)
        tom_k = None

    events: List[Event] = [
        (float(t), int(note), int(v))
        for t, note, v in zip(onset_times, notes, velocities)
    ]
    return events, {"stem": stem_name, "onsets": len(events), "tom_k": tom_k}


def transcribe(
    path: str,
    *,
    note: int = DEFAULT_NOTE,
    sr: Optional[int] = None,
    hop_length: int = DEFAULT_HOP_LENGTH,
    window_ms: float = DEFAULT_WINDOW_MS,
    delta: float = 0.07,
) -> List[Event]:
    """Transcribe a single-voice drum WAV into ``(time, note, velocity)`` events."""
    y, sr = load_audio(path, sr=sr)
    onset_times = detect_onsets(y, sr, hop_length=hop_length, delta=delta)
    velocities = extract_velocities(y, sr, onset_times, window_ms=window_ms)
    return [
        (float(t), int(note), int(v)) for t, v in zip(onset_times, velocities)
    ]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect onsets in a pre-isolated drum WAV and write a General MIDI "
            ".mid file (Phase 1: single drum voice, GM only)."
        )
    )
    parser.add_argument("input", type=Path, help="Path to the input WAV file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .mid path (default: input filename with .mid extension).",
    )
    parser.add_argument(
        "-n",
        "--note",
        type=int,
        default=DEFAULT_NOTE,
        help=f"GM note number to assign to every hit (default: {DEFAULT_NOTE}, kick).",
    )
    parser.add_argument(
        "--tempo",
        type=float,
        default=DEFAULT_TEMPO,
        help=f"Initial tempo written to the MIDI file (default: {DEFAULT_TEMPO}).",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=DEFAULT_WINDOW_MS,
        help="Velocity measurement window after each onset, in ms.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=DEFAULT_HOP_LENGTH,
        help="Onset detection hop length in samples.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.07,
        help="Onset peak-picking threshold (higher = fewer, stronger onsets).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if not args.input.exists():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    events = transcribe(
        str(args.input),
        note=args.note,
        hop_length=args.hop_length,
        window_ms=args.window_ms,
        delta=args.delta,
    )

    output = args.output or args.input.with_suffix(".mid")
    write_midi(events, output, tempo=args.tempo)

    print(f"Detected {len(events)} onsets -> {output}")
    if events:
        velocities = [v for _, _, v in events]
        print(
            f"  velocity range: {min(velocities)}-{max(velocities)} "
            f"across notes at {events[0][0]:.3f}s .. {events[-1][0]:.3f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
