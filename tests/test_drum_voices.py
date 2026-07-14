"""Tests for drum voice filter helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.drum_voices import (  # noqa: E402
    ALL_VOICES,
    filter_gm_events_by_voices,
    toggle_voice_filter,
    voices_label,
)
from pipeline.preview import mix_stem_wavs  # noqa: E402

SR = 44100


def test_toggle_from_all_selects_single_voice() -> None:
    assert toggle_voice_filter(ALL_VOICES, "kick") == frozenset({"kick"})


def test_toggle_additive_second_voice() -> None:
    current = frozenset({"kick"})
    assert toggle_voice_filter(current, "snare") == frozenset({"kick", "snare"})


def test_toggle_remove_voice() -> None:
    current = frozenset({"kick", "snare"})
    assert toggle_voice_filter(current, "snare") == frozenset({"kick"})


def test_toggle_last_voice_returns_all() -> None:
    assert toggle_voice_filter(frozenset({"kick"}), "kick") == ALL_VOICES


def test_filter_gm_events_by_voices() -> None:
    events = [(0.0, 36, 100), (0.5, 38, 90), (1.0, 49, 80)]
    filtered = filter_gm_events_by_voices(events, frozenset({"kick"}))
    assert filtered == [(0.0, 36, 100)]
    # Unknown / retired metal notes are dropped when filtering.
    filtered_all_known = filter_gm_events_by_voices(events, frozenset({"kick", "snare", "toms"}))
    assert filtered_all_known == [(0.0, 36, 100), (0.5, 38, 90)]


def test_voices_label() -> None:
    assert voices_label(ALL_VOICES) == "all"
    assert voices_label(frozenset({"kick", "snare"})) == "kick+snare"


def test_mix_stem_wavs_selected_voices(tmp_path: Path) -> None:
    kick = (0.5 * np.ones(SR // 10, dtype=np.float32))
    snare = (0.25 * np.ones(SR // 10, dtype=np.float32))
    sf.write(str(tmp_path / "kick.wav"), kick, SR)
    sf.write(str(tmp_path / "snare.wav"), snare, SR)
    mixed = mix_stem_wavs(tmp_path, frozenset({"kick"}), SR)
    assert mixed.size == kick.size
    assert mixed[0] == pytest.approx(0.5, abs=0.05)
