"""Transcription v2: open-hat rerouting and closed/open hat classification.

v2 reroutes cymbal onsets that look hat-like (hi-hat stem energy at the same
time) and classifies hat hits as closed (GM 42) vs open (GM 46) using sustain
and spectral features on the hi-hat stem.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.merge import (  # noqa: E402
    DEFAULT_BLEED_RATIO,
    DEFAULT_BLEED_WINDOW_MS,
    DEFAULT_HIHAT_CYMBAL_WINDOW_MS,
    DEFAULT_MIN_IOI_MS,
    STEM_ORDER,
    _cymbals_detect_path,
)
from pipeline.onset_detection import (  # noqa: E402
    STEM_PRESETS,
    Event,
    detect_stem_events,
    min_ioi_by_note,
)

# --- Tunable v2 heuristics -------------------------------------------------
HAT_ENERGY_WINDOW_MS = 15.0
HAT_ENERGY_FLOOR = 0.002
HAT_TO_CYMBAL_RATIO = 0.40

HAT_ATTACK_WINDOW_MS = 20.0
HAT_SUSTAIN_START_MS = 40.0
HAT_SUSTAIN_END_MS = 120.0
OPEN_SUSTAIN_RATIO = 1.35

HAT_CENTROID_FMIN = 6000.0
HAT_CENTROID_FMAX = 12000.0
OPEN_CENTROID_HZ = 7500.0

GM_NOTE_HIHAT_CLOSED = 42
GM_NOTE_HIHAT_OPEN = 46
GM_NOTE_CYMBALS = 49


def stem_energy_at(
    y: np.ndarray,
    sr: int,
    time_s: float,
    *,
    window_ms: float = HAT_ENERGY_WINDOW_MS,
) -> float:
    """RMS energy in a short window centered on ``time_s``."""
    if y.size == 0 or sr <= 0:
        return 0.0
    half = max(1, int(sr * window_ms / 2000.0))
    center = int(time_s * sr)
    start = max(0, center - half)
    end = min(y.size, center + half)
    segment = y[start:end]
    if segment.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))


def is_hatlike_cymbal_onset(
    time_s: float,
    hihat_y: np.ndarray,
    cymbals_y: np.ndarray,
    sr: int,
) -> bool:
    """Return True when cymbal onset energy looks like hi-hat bleed."""
    hat_rms = stem_energy_at(hihat_y, sr, time_s)
    if hat_rms < HAT_ENERGY_FLOOR:
        return False
    cym_rms = stem_energy_at(cymbals_y, sr, time_s)
    ratio = hat_rms / max(cym_rms, 1e-9)
    return ratio >= HAT_TO_CYMBAL_RATIO


def _segment_rms(y: np.ndarray, sr: int, start_s: float, end_s: float) -> float:
    start = max(0, int(start_s * sr))
    end = min(y.size, int(end_s * sr))
    segment = y[start:end]
    if segment.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))


def _hat_centroid_hz(y: np.ndarray, sr: int, time_s: float) -> float:
    """Spectral centroid in the hi-hat band after onset."""
    start = max(0, int(time_s * sr))
    end = min(y.size, start + max(1, int(sr * HAT_ATTACK_WINDOW_MS / 1000.0)))
    segment = y[start:end]
    if segment.size < 64:
        return OPEN_CENTROID_HZ
    spec = np.abs(librosa.stft(segment, n_fft=512, hop_length=128))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    band = (freqs >= HAT_CENTROID_FMIN) & (freqs <= HAT_CENTROID_FMAX)
    if not band.any():
        return OPEN_CENTROID_HZ
    band_spec = spec[band, :]
    band_freqs = freqs[band]
    weights = np.sum(band_spec, axis=1)
    total = float(np.sum(weights))
    if total <= 0.0:
        return OPEN_CENTROID_HZ
    return float(np.sum(band_freqs * weights) / total)


def classify_hat_open_closed(
    y: np.ndarray,
    sr: int,
    time_s: float,
) -> int:
    """Return GM 42 (closed) or 46 (open) for a hat onset at ``time_s``."""
    attack_end = time_s + HAT_ATTACK_WINDOW_MS / 1000.0
    sustain_start = time_s + HAT_SUSTAIN_START_MS / 1000.0
    sustain_end = time_s + HAT_SUSTAIN_END_MS / 1000.0
    attack_rms = _segment_rms(y, sr, time_s, attack_end)
    sustain_rms = _segment_rms(y, sr, sustain_start, sustain_end)
    sustain_ratio = sustain_rms / max(attack_rms, 1e-9)
    centroid = _hat_centroid_hz(y, sr, time_s)
    open_score = 0
    if sustain_ratio >= OPEN_SUSTAIN_RATIO:
        open_score += 1
    if centroid < OPEN_CENTROID_HZ:
        open_score += 1
    return GM_NOTE_HIHAT_OPEN if open_score >= 1 else GM_NOTE_HIHAT_CLOSED


def reroute_hatlike_cymbals(
    stem_events: Dict[str, List[Event]],
    hihat_y: np.ndarray,
    cymbals_y: np.ndarray,
    sr: int,
) -> Dict[str, List[Event]]:
    """Move hat-like cymbal onsets into the hihat stem bucket."""
    out = deepcopy(stem_events)
    cymbals = list(out.get("cymbals", []))
    if not cymbals:
        return out
    kept_cymbals: List[Event] = []
    rerouted: List[Event] = []
    for time_s, note, velocity in cymbals:
        if is_hatlike_cymbal_onset(time_s, hihat_y, cymbals_y, sr):
            rerouted.append((time_s, GM_NOTE_HIHAT_CLOSED, velocity))
        else:
            kept_cymbals.append((time_s, note, velocity))
    out["cymbals"] = kept_cymbals
    if rerouted:
        out.setdefault("hihat", [])
        out["hihat"] = list(out["hihat"]) + rerouted
    return out


def classify_hihat_events(
    stem_events: Dict[str, List[Event]],
    hihat_y: np.ndarray,
    sr: int,
) -> Dict[str, List[Event]]:
    """Assign GM 42 or 46 to every event in the hihat stem."""
    out = deepcopy(stem_events)
    hat_events = out.get("hihat", [])
    if not hat_events:
        return out
    out["hihat"] = [
        (time_s, classify_hat_open_closed(hihat_y, sr, time_s), velocity)
        for time_s, _note, velocity in hat_events
    ]
    return out


def _is_hihat_event_v2(note: int, stem: str) -> bool:
    return (
        stem == "hihat"
        or note in (GM_NOTE_HIHAT_CLOSED, GM_NOTE_HIHAT_OPEN)
    )


def _is_cymbal_event_v2(note: int, stem: str) -> bool:
    return stem == "cymbals" or note == GM_NOTE_CYMBALS


def resolve_cymbal_hihat_collisions_v2(
    tagged: List[Tuple[float, int, int, str]],
    *,
    window_ms: float = DEFAULT_HIHAT_CYMBAL_WINDOW_MS,
) -> List[Tuple[float, int, int, str]]:
    """Drop cymbal/crash events coincident with closed or open hi-hat hits."""
    window_s = window_ms / 1000.0
    hat_times = [
        time_s
        for time_s, note, _velocity, stem in tagged
        if _is_hihat_event_v2(note, stem)
    ]
    if not hat_times:
        return tagged

    drop: set[int] = set()
    for index, (time_s, note, _velocity, stem) in enumerate(tagged):
        if not _is_cymbal_event_v2(note, stem):
            continue
        for hat_t in hat_times:
            if abs(time_s - hat_t) <= window_s:
                drop.add(index)
                break
    return [event for index, event in enumerate(tagged) if index not in drop]


def _min_ioi_by_note_v2() -> Dict[int, float]:
    mapping = min_ioi_by_note()
    hat_ioi = mapping.get(GM_NOTE_HIHAT_CLOSED, 60.0)
    mapping[GM_NOTE_HIHAT_OPEN] = hat_ioi
    return mapping


def merge_events_v2(
    stem_events: Dict[str, List[Event]],
    *,
    apply_min_ioi: bool = True,
    min_ioi_ms: Optional[Dict[int, float]] = None,
    velocity_floor: int = 0,
    bleed_suppression: bool = False,
    bleed_window_ms: float = DEFAULT_BLEED_WINDOW_MS,
    bleed_ratio: float = DEFAULT_BLEED_RATIO,
) -> List[Event]:
    """Merge per-stem events with v2 hat/cymbal collision rules."""
    if min_ioi_ms is None:
        min_ioi_ms = _min_ioi_by_note_v2()

    tagged: List[Tuple[float, int, int, str]] = []
    for stem, events in stem_events.items():
        for time_s, note, velocity in events:
            tagged.append((float(time_s), int(note), int(velocity), stem))
    tagged.sort(key=lambda e: (e[0], e[1]))

    if apply_min_ioi:
        last_kept: Dict[int, float] = {}
        deduped: List[Tuple[float, int, int, str]] = []
        for time_s, note, velocity, stem in tagged:
            gap_s = min_ioi_ms.get(note, DEFAULT_MIN_IOI_MS) / 1000.0
            prev = last_kept.get(note)
            if prev is not None and (time_s - prev) < gap_s:
                continue
            last_kept[note] = time_s
            deduped.append((time_s, note, velocity, stem))
        tagged = deduped

    tagged = resolve_cymbal_hihat_collisions_v2(tagged)

    if velocity_floor > 0:
        tagged = [e for e in tagged if e[2] >= velocity_floor]

    if bleed_suppression:
        window_s = bleed_window_ms / 1000.0
        drop = set()
        for i, (ti, _ni, vi, si) in enumerate(tagged):
            for j, (tj, _nj, vj, sj) in enumerate(tagged):
                if i == j or si == sj:
                    continue
                if abs(ti - tj) <= window_s and vi < vj and vi < bleed_ratio * vj:
                    drop.add(i)
                    break
        tagged = [e for k, e in enumerate(tagged) if k not in drop]

    return [(t, n, v) for (t, n, v, _stem) in tagged]


def _load_stem_files(stems_path: Path) -> Dict[str, Path]:
    manifest_path = stems_path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        return {
            name: stems_path / filename
            for name, filename in manifest.get("stems", {}).items()
        }
    return {
        name: stems_path / f"{name}.wav"
        for name in STEM_PRESETS
        if (stems_path / f"{name}.wav").exists()
    }


def _detect_stem_events_all(
    stem_files: Dict[str, Path],
    *,
    delta_scale: float,
) -> Tuple[Dict[str, List[Event]], dict, List[str]]:
    """Run per-stem onset detection (shared between v1/v2 entry points)."""
    stem_events: Dict[str, List[Event]] = {}
    summary: dict = {"stems": {}, "tom_k": None}
    temp_detect_paths: List[str] = []
    cymbals_path = stem_files.get("cymbals")
    hihat_path = stem_files.get("hihat")

    for name in STEM_ORDER:
        path = stem_files.get(name)
        if path is None or not path.exists():
            continue
        detect_path = str(path)
        if (
            name == "cymbals"
            and cymbals_path is not None
            and hihat_path is not None
            and hihat_path.exists()
        ):
            detect_path, temp_path = _cymbals_detect_path(cymbals_path, hihat_path)
            if temp_path is not None:
                temp_detect_paths.append(temp_path)
        events, info = detect_stem_events(detect_path, name, delta_scale=delta_scale)
        stem_events[name] = events
        summary["stems"][name] = info
        if name == "toms":
            summary["tom_k"] = info.get("tom_k")

    return stem_events, summary, temp_detect_paths


def transcribe_stems_v2(
    stems_dir: str,
    *,
    bleed_suppression: bool = False,
    velocity_floor: int = 0,
    delta_scale: float = 1.0,
) -> Tuple[List[Event], dict]:
    """Transcribe stems with v2 open-hat rerouting and classification."""
    stems_path = Path(stems_dir)
    stem_files = _load_stem_files(stems_path)
    hihat_path = stem_files.get("hihat")
    cymbals_path = stem_files.get("cymbals")

    if hihat_path is None or not hihat_path.exists():
        from pipeline.merge import _transcribe_stems_v1  # noqa: WPS433

        events, summary = _transcribe_stems_v1(
            stems_dir,
            bleed_suppression=bleed_suppression,
            velocity_floor=velocity_floor,
            delta_scale=delta_scale,
        )
        summary["transcription_version"] = "v1"
        summary["v2_fallback"] = "hihat stem missing"
        return events, summary

    stem_events, summary, temp_detect_paths = _detect_stem_events_all(
        stem_files, delta_scale=delta_scale
    )

    hihat_y, sr = librosa.load(str(hihat_path), sr=None, mono=True)
    cymbals_y = np.zeros_like(hihat_y)
    if cymbals_path is not None and cymbals_path.exists():
        cymbals_y, _ = librosa.load(str(cymbals_path), sr=sr, mono=True)
        length = max(hihat_y.size, cymbals_y.size)
        if hihat_y.size < length:
            padded = np.zeros(length, dtype=np.float32)
            padded[: hihat_y.size] = hihat_y
            hihat_y = padded
        if cymbals_y.size < length:
            padded = np.zeros(length, dtype=np.float32)
            padded[: cymbals_y.size] = cymbals_y
            cymbals_y = padded

    stem_events = reroute_hatlike_cymbals(stem_events, hihat_y, cymbals_y, sr)
    stem_events = classify_hihat_events(stem_events, hihat_y, sr)

    for temp_path in temp_detect_paths:
        Path(temp_path).unlink(missing_ok=True)

    merged = merge_events_v2(
        stem_events,
        bleed_suppression=bleed_suppression,
        velocity_floor=velocity_floor,
    )

    note_counts: Dict[int, int] = {}
    for _t, note, _v in merged:
        note_counts[note] = note_counts.get(note, 0) + 1
    summary["total_events"] = len(merged)
    summary["note_counts"] = note_counts
    summary["transcription_version"] = "v2"
    return merged, summary
