"""Phase 1 tests: onset detection, velocity mapping, and MIDI writing."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.midi_writer import write_midi  # noqa: E402
from pipeline.onset_detection import MIN_VELOCITY, transcribe  # noqa: E402

SAMPLE_RATE = 44100
HIT_TIMES = [0.5, 1.0, 1.5, 2.0]
HIT_AMPLITUDES = [0.2, 0.4, 0.7, 1.0]


def _synth_kick_wav(path: Path) -> Tuple[List[float], List[float]]:
    """Write a synthetic solo-kick WAV: four decaying 60 Hz thumps of rising level."""
    duration = HIT_TIMES[-1] + 0.5
    y = np.zeros(int(SAMPLE_RATE * duration), dtype=np.float32)
    hit_len = int(0.15 * SAMPLE_RATE)
    t = np.arange(hit_len) / SAMPLE_RATE
    envelope = np.exp(-t * 30.0)
    tone = np.sin(2.0 * np.pi * 60.0 * t) * envelope
    for hit_time, amp in zip(HIT_TIMES, HIT_AMPLITUDES):
        start = int(hit_time * SAMPLE_RATE)
        y[start : start + hit_len] += (tone * amp).astype(np.float32)
    sf.write(str(path), y, SAMPLE_RATE)
    return HIT_TIMES, HIT_AMPLITUDES


def test_transcribe_detects_timed_hits(tmp_path: Path) -> None:
    wav = tmp_path / "kick.wav"
    times, _ = _synth_kick_wav(wav)

    events = transcribe(str(wav), note=36)

    assert len(events) == len(times)
    detected_times = [t for t, _, _ in events]
    for detected, expected in zip(detected_times, times):
        assert abs(detected - expected) < 0.06


def test_transcribe_notes_are_gm_kick(tmp_path: Path) -> None:
    wav = tmp_path / "kick.wav"
    _synth_kick_wav(wav)

    events = transcribe(str(wav), note=36)

    assert all(note == 36 for _, note, _ in events)


def test_velocity_tracks_loudness(tmp_path: Path) -> None:
    wav = tmp_path / "kick.wav"
    _synth_kick_wav(wav)

    events = transcribe(str(wav), note=36)
    velocities = [v for _, _, v in events]

    assert velocities == sorted(velocities)
    assert velocities[-1] > velocities[0]
    assert all(MIN_VELOCITY <= v <= 127 for v in velocities)
    # Quietest detected hit maps exactly to the musical floor, not to 1.
    assert min(velocities) == MIN_VELOCITY


def test_write_midi_roundtrip(tmp_path: Path) -> None:
    events = [(0.5, 36, 40), (1.0, 36, 80), (1.5, 36, 127)]
    out = tmp_path / "out.mid"

    write_midi(events, out)

    assert out.exists()
    pm = pretty_midi.PrettyMIDI(str(out))
    assert len(pm.instruments) == 1
    instrument = pm.instruments[0]
    assert instrument.is_drum
    assert len(instrument.notes) == len(events)
    assert all(note.pitch == 36 for note in instrument.notes)
    written_starts = sorted(note.start for note in instrument.notes)
    assert written_starts == pytest.approx([0.5, 1.0, 1.5], abs=1e-3)


def test_write_midi_stamps_tempo(tmp_path: Path) -> None:
    events = [(0.5, 36, 40), (1.0, 36, 80)]
    out = tmp_path / "tempo84.mid"

    write_midi(events, out, tempo=84.0)

    pm = pretty_midi.PrettyMIDI(str(out))
    assert pm.get_tempo_changes()[1][0] == pytest.approx(84.0, abs=0.1)
    written_starts = sorted(note.start for note in pm.instruments[0].notes)
    assert written_starts == pytest.approx([0.5, 1.0], abs=1e-3)


def test_end_to_end_kick_to_midi(tmp_path: Path) -> None:
    wav = tmp_path / "kick.wav"
    times, _ = _synth_kick_wav(wav)
    out = tmp_path / "kick.mid"

    events = transcribe(str(wav), note=36)
    write_midi(events, out)

    pm = pretty_midi.PrettyMIDI(str(out))
    notes = pm.instruments[0].notes
    assert len(notes) == len(times)
    assert all(note.pitch == 36 for note in notes)
    velocities = [note.velocity for note in sorted(notes, key=lambda n: n.start)]
    assert velocities == sorted(velocities)
