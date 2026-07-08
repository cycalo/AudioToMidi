"""Tests for transcription v2 open-hat rerouting and classification."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.transcription_v2 import (  # noqa: E402
    GM_NOTE_HIHAT_CLOSED,
    GM_NOTE_HIHAT_OPEN,
    GM_NOTE_CYMBALS,
    classify_hat_open_closed,
    classify_hihat_events,
    is_hatlike_cymbal_onset,
    merge_events_v2,
    reroute_hatlike_cymbals,
    resolve_cymbal_hihat_collisions_v2,
)
from pipeline.merge import merge_events, transcribe_stems  # noqa: E402

SR = 44100


def _tone_burst(
    sr: int,
    *,
    start_s: float,
    duration_s: float,
    freq: float,
    amp: float = 0.5,
) -> np.ndarray:
    length = int((start_s + duration_s + 0.05) * sr)
    y = np.zeros(length, dtype=np.float32)
    start = int(start_s * sr)
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    burst = amp * np.sin(2 * np.pi * freq * t) * np.exp(-t * 8.0)
    end = min(y.size, start + n)
    y[start:end] = burst[: end - start]
    return y


def test_is_hatlike_cymbal_onset_detects_hat_energy() -> None:
    t = 1.0
    hihat_y = _tone_burst(SR, start_s=t, duration_s=0.08, freq=8000.0, amp=0.8)
    cymbals_y = _tone_burst(SR, start_s=t, duration_s=0.06, freq=4000.0, amp=0.4)
    assert is_hatlike_cymbal_onset(t, hihat_y, cymbals_y, SR)


def test_is_hatlike_cymbal_onset_rejects_isolated_crash() -> None:
    t = 1.0
    hihat_y = np.zeros(int(2 * SR), dtype=np.float32)
    cymbals_y = _tone_burst(SR, start_s=t, duration_s=0.05, freq=5000.0, amp=0.9)
    assert not is_hatlike_cymbal_onset(t, hihat_y, cymbals_y, SR)


def test_reroute_moves_hatlike_cymbal_to_hihat() -> None:
    t = 1.0
    hihat_y = _tone_burst(SR, start_s=t, duration_s=0.08, freq=8000.0, amp=0.8)
    cymbals_y = _tone_burst(SR, start_s=t, duration_s=0.06, freq=4000.0, amp=0.4)
    stem_events = {
        "cymbals": [(t, GM_NOTE_CYMBALS, 90)],
        "hihat": [],
    }
    out = reroute_hatlike_cymbals(stem_events, hihat_y, cymbals_y, SR)
    assert out["cymbals"] == []
    assert len(out["hihat"]) == 1


def test_classify_hat_open_closed_short_vs_long() -> None:
    closed_y = _tone_burst(SR, start_s=0.5, duration_s=0.02, freq=9000.0, amp=1.0)
    open_y = _tone_burst(SR, start_s=0.5, duration_s=0.25, freq=7000.0, amp=0.7)
    closed_note = classify_hat_open_closed(closed_y, SR, 0.5)
    open_note = classify_hat_open_closed(open_y, SR, 0.5)
    assert closed_note == GM_NOTE_HIHAT_CLOSED
    assert open_note == GM_NOTE_HIHAT_OPEN


def test_classify_hihat_events_assigns_notes() -> None:
    y = _tone_burst(SR, start_s=0.2, duration_s=0.25, freq=7000.0)
    stem_events = {"hihat": [(0.2, GM_NOTE_HIHAT_CLOSED, 80)]}
    out = classify_hihat_events(stem_events, y, SR)
    assert out["hihat"][0][1] in (GM_NOTE_HIHAT_CLOSED, GM_NOTE_HIHAT_OPEN)


def test_merge_v2_drops_crash_when_open_hat_coincident() -> None:
    events = {
        "hihat": [(1.0, GM_NOTE_HIHAT_OPEN, 80)],
        "cymbals": [(1.01, GM_NOTE_CYMBALS, 100)],
    }
    merged = merge_events_v2(events, apply_min_ioi=False)
    notes = [n for _, n, _ in merged]
    assert GM_NOTE_CYMBALS not in notes
    assert GM_NOTE_HIHAT_OPEN in notes


def test_merge_v1_unchanged_hat_priority() -> None:
    events = {
        "hihat": [(1.0, GM_NOTE_HIHAT_CLOSED, 80)],
        "cymbals": [(1.01, GM_NOTE_CYMBALS, 100)],
    }
    merged = merge_events(events, apply_min_ioi=False)
    notes = [n for _, n, _ in merged]
    assert GM_NOTE_CYMBALS not in notes
    assert GM_NOTE_HIHAT_CLOSED in notes


def test_transcribe_v1_explicit_version(tmp_path: Path) -> None:
    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    for name in ("kick", "snare", "toms", "hihat", "cymbals"):
        (stems_dir / f"{name}.wav").write_bytes(b"")
    # Empty wav files will fail librosa - use minimal valid wav via soundfile
    import soundfile as sf

    silence = np.zeros(SR // 10, dtype=np.float32)
    for name in ("kick", "snare", "toms", "hihat", "cymbals"):
        sf.write(str(stems_dir / f"{name}.wav"), silence, SR)
    events_v1, summary_v1 = transcribe_stems(
        str(stems_dir), transcription_version="v1"
    )
    events_v2, summary_v2 = transcribe_stems(
        str(stems_dir), transcription_version="v2"
    )
    assert summary_v1["transcription_version"] == "v1"
    assert summary_v2["transcription_version"] == "v2"
    assert isinstance(events_v1, list)
    assert isinstance(events_v2, list)


def test_resolve_v2_treats_open_hat_as_hat() -> None:
    tagged = [
        (1.0, GM_NOTE_HIHAT_OPEN, 80, "hihat"),
        (1.01, GM_NOTE_CYMBALS, 100, "cymbals"),
    ]
    out = resolve_cymbal_hihat_collisions_v2(tagged)
    assert len(out) == 1
    assert out[0][1] == GM_NOTE_HIHAT_OPEN
