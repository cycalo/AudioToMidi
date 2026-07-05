"""Phase 3 tests: floor/rack tom clustering, merge logic, and full-pipeline runs.

Unit tests use deterministic synthetic inputs. The real-fixture tests run the
full stems-to-MIDI pipeline against the Phase 2 Demucs output in
tests/fixtures/drums_X_stems/ (skipped when those directories are absent).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.merge import merge_events, transcribe_stems  # noqa: E402
from pipeline.midi_writer import write_midi  # noqa: E402
from pipeline.onset_detection import (  # noqa: E402
    TOM_NOTE_FALLBACK,
    TOM_NOTE_FLOOR,
    TOM_NOTE_RACK,
    TOM_NOTES,
    cluster_toms,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _stem_dirs() -> list[Path]:
    return sorted(FIXTURES.glob("drums_*_stems"))


# --------------------------------------------------------------------------
# Floor/rack tom clustering (fixed k=2)
# --------------------------------------------------------------------------
def test_cluster_toms_three_pitch_groups_map_to_two_toms() -> None:
    # Three pitch groups still resolve to floor + rack (k=2), not three notes.
    pitches = [40.0, 40.2, 47.0, 47.1, 55.0, 55.3]
    notes, k = cluster_toms(pitches)
    assert k == 2
    assert set(notes) == set(TOM_NOTES)
    assert notes[0] == TOM_NOTE_FLOOR and notes[-1] == TOM_NOTE_RACK


def test_cluster_toms_two_groups() -> None:
    pitches = [41.0, 41.2, 41.1, 54.0, 54.2, 54.1]
    notes, k = cluster_toms(pitches)
    assert k == 2
    assert set(notes) == set(TOM_NOTES)


def test_cluster_toms_single_pitch() -> None:
    notes, k = cluster_toms([48.0, 48.05, 47.98])
    assert k == 1
    assert set(notes) == {TOM_NOTE_FALLBACK}


def test_cluster_toms_empty() -> None:
    notes, k = cluster_toms([])
    assert k == 0
    assert notes == []


def test_cluster_toms_unvoiced_filled() -> None:
    # NaN (unvoiced) onsets still receive a note.
    pitches = [40.0, np.nan, 55.0, 55.1, 40.1]
    notes, k = cluster_toms(pitches)
    assert len(notes) == len(pitches)
    assert all(n in set(TOM_NOTES) | {TOM_NOTE_FALLBACK} for n in notes)


# --------------------------------------------------------------------------
# merge_events
# --------------------------------------------------------------------------
def test_merge_sorts_by_time() -> None:
    merged = merge_events(
        {"kick": [(1.0, 36, 100)], "snare": [(0.5, 38, 90), (0.2, 38, 80)]},
        apply_min_ioi=False,
    )
    times = [t for t, _, _ in merged]
    assert times == sorted(times)


def test_merge_min_ioi_drops_close_doubles() -> None:
    # Two kicks 10 ms apart -> the second is suppressed (min IOI 30 ms).
    events = {"kick": [(1.000, 36, 100), (1.010, 36, 90), (1.100, 36, 95)]}
    merged = merge_events(events, min_ioi_ms={36: 30.0})
    kick_times = [t for t, n, _ in merged if n == 36]
    assert kick_times == [1.000, 1.100]


def test_merge_min_ioi_off() -> None:
    events = {"kick": [(1.000, 36, 100), (1.010, 36, 90)]}
    merged = merge_events(events, apply_min_ioi=False)
    assert len(merged) == 2


def test_merge_velocity_floor() -> None:
    events = {"snare": [(0.1, 38, 15), (0.5, 38, 60)]}
    merged = merge_events(events, velocity_floor=20)
    assert [v for _, _, v in merged] == [60]


def test_merge_bleed_suppression_off_by_default() -> None:
    # A quiet cymbal coincident with a loud kick is KEPT when suppression is off.
    events = {"kick": [(1.0, 36, 120)], "cymbals": [(1.002, 49, 20)]}
    merged = merge_events(events)
    assert (1.002, 49, 20) in merged


def test_merge_bleed_suppression_on_drops_quiet() -> None:
    events = {"kick": [(1.0, 36, 120)], "cymbals": [(1.002, 49, 20)]}
    merged = merge_events(events, bleed_suppression=True, bleed_ratio=0.35)
    assert (1.002, 49, 20) not in merged
    assert (1.0, 36, 120) in merged


def test_merge_bleed_keeps_loud_coincident() -> None:
    # Comparable-loudness coincident hits are both kept even with suppression on.
    events = {"kick": [(1.0, 36, 100)], "snare": [(1.003, 38, 95)]}
    merged = merge_events(events, bleed_suppression=True, bleed_ratio=0.35)
    assert len(merged) == 2


# --------------------------------------------------------------------------
# Full pipeline against real Demucs stems (skip if absent)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not _stem_dirs(), reason="no separated stem dirs in tests/fixtures/")
def test_pipeline_real_stems(tmp_path: Path) -> None:
    stems_dir = _stem_dirs()[0]
    events, summary = transcribe_stems(str(stems_dir))

    assert len(events) > 0
    # Time-sorted.
    times = [t for t, _, _ in events]
    assert times == sorted(times)

    notes_present = {n for _, n, _ in events}
    assert 36 in notes_present, "kick missing"
    assert 38 in notes_present, "snare missing"
    assert 49 in notes_present, "cymbals missing"
    assert notes_present & (set(TOM_NOTES) | {TOM_NOTE_FALLBACK}), "no tom notes"

    # Per-voice min-IOI respected (kick/snare 30 ms).
    for target, gap in ((36, 0.030), (38, 0.030)):
        hits = sorted(t for t, n, _ in events if n == target)
        for a, b in zip(hits, hits[1:]):
            assert (b - a) >= gap - 1e-6, f"note {target} spacing {b - a:.4f}s < {gap}s"

    # Writes a valid MIDI with all events on the drum channel.
    out = tmp_path / "out.mid"
    write_midi(events, out)
    pm = pretty_midi.PrettyMIDI(str(out))
    assert pm.instruments[0].is_drum
    assert len(pm.instruments[0].notes) == len(events)


@pytest.mark.skipif(not _stem_dirs(), reason="no separated stem dirs in tests/fixtures/")
def test_pipeline_deterministic() -> None:
    stems_dir = _stem_dirs()[0]
    events_a, _ = transcribe_stems(str(stems_dir))
    events_b, _ = transcribe_stems(str(stems_dir))
    assert events_a == events_b
