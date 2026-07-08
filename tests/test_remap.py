"""Phase 4 tests: plugin profile loading and GM -> plugin note remapping.

All tests use deterministic synthetic inputs (no audio fixtures required).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pretty_midi
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import remap  # noqa: E402
from pipeline.midi_writer import write_midi  # noqa: E402
from pipeline.remap import (  # noqa: E402
    available_profiles,
    load_profile,
    remap_events,
    remap_midi_file,
)

# The GM notes the transcription pipeline actually emits (Phase 3 v1).
PRODUCED_NOTES = {36, 38, 42, 45, 47, 49, 50}

PLUGIN_PROFILES = (
    "superior_drummer_3",
    "ezdrummer_3",
    "addictive_drums_2",
    "bfd3",
    "steven_slate_5_5",
    "ggd",
    "drumforge",
)
ALL_PROFILES = PLUGIN_PROFILES + ("general_midi",)
PASS_THROUGH_PROFILES = tuple(p for p in ALL_PROFILES if p != "ggd")

VALID_CONFIDENCE = {"high", "medium", "low"}


# --------------------------------------------------------------------------
# Profile validation
# --------------------------------------------------------------------------
def test_available_profiles_lists_all_shipped_profiles():
    available = set(available_profiles())
    assert set(ALL_PROFILES).issubset(available)


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_profile_covers_produced_notes_with_int_values(name):
    profile = load_profile(name)
    assert profile["plugin"]
    assert profile["confidence"] in VALID_CONFIDENCE
    note_map = profile["map"]
    assert set(note_map.keys()) == PRODUCED_NOTES
    assert all(isinstance(k, int) and isinstance(v, int) for k, v in note_map.items())


@pytest.mark.parametrize("name", PASS_THROUGH_PROFILES)
def test_shipped_profiles_are_pass_through(name):
    note_map = load_profile(name)["map"]
    assert all(src == dst for src, dst in note_map.items())


def test_ggd_profile_maps_floor_toms_to_modern_massive_floor_tom_1():
    """GGD is verified for Modern & Massive GM: floor/fallback toms -> note 43."""
    note_map = load_profile("ggd")["map"]
    assert note_map[36] == 36
    assert note_map[38] == 38
    assert note_map[49] == 49
    assert note_map[50] == 50
    assert note_map[42] == 54
    assert note_map[45] == 43
    assert note_map[47] == 43
    assert load_profile("ggd")["confidence"] == "medium"
    assert load_profile("ggd")["preview_kit"] == "Preview Kit"


def test_load_profile_by_display_name():
    profile = load_profile("Superior Drummer 3")
    assert profile["plugin"] == "Superior Drummer 3"


def test_load_profile_unknown_raises():
    with pytest.raises(FileNotFoundError):
        load_profile("does_not_exist_plugin")


def test_load_profile_missing_key_raises(tmp_path, monkeypatch):
    bad = tmp_path / "broken.json"
    bad.write_text(json.dumps({"plugin": "Broken"}), encoding="utf-8")  # no map/confidence
    monkeypatch.setattr(remap, "MAPPINGS_DIR", tmp_path)
    with pytest.raises(ValueError):
        load_profile("broken")


def test_load_profile_non_integer_note_raises(tmp_path, monkeypatch):
    bad = tmp_path / "bad_notes.json"
    bad.write_text(
        json.dumps({"plugin": "Bad", "confidence": "low", "map": {"kick": 36}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(remap, "MAPPINGS_DIR", tmp_path)
    with pytest.raises(ValueError):
        load_profile("bad_notes")


# --------------------------------------------------------------------------
# Event remapping
# --------------------------------------------------------------------------
def test_remap_events_pass_through_identity():
    events = [(0.0, 36, 100), (0.5, 38, 90), (1.0, 50, 80), (1.5, 49, 110)]
    result = remap_events(events, load_profile("general_midi"))
    assert result == events


def test_remap_events_non_identity_profile():
    profile = {"plugin": "X", "confidence": "low", "map": {36: 60, 38: 61, 50: 48}}
    events = [(0.0, 36, 100), (0.25, 38, 90), (0.5, 50, 80)]
    result = remap_events(events, profile)
    assert [n for _t, n, _v in result] == [60, 61, 48]
    # Timing and velocity are preserved.
    assert [(t, v) for t, _n, v in result] == [(0.0, 100), (0.25, 90), (0.5, 80)]


def test_remap_events_unmapped_note_passes_through_with_warning(caplog):
    profile = {"plugin": "X", "confidence": "low", "map": {36: 60}}
    events = [(0.0, 36, 100), (0.5, 99, 80), (1.0, 99, 70)]
    with caplog.at_level(logging.WARNING, logger="pipeline.remap"):
        result = remap_events(events, profile)
    notes = [n for _t, n, _v in result]
    assert notes == [60, 99, 99]
    # Exactly one warning for the distinct unmapped note (99), not one per event.
    warnings = [r for r in caplog.records if "99" in r.getMessage()]
    assert len(warnings) == 1


# --------------------------------------------------------------------------
# MIDI file remapping + CLI
# --------------------------------------------------------------------------
def _drum_pitches(mid_path: Path) -> list[int]:
    pm = pretty_midi.PrettyMIDI(str(mid_path))
    pitches: list[int] = []
    for inst in pm.instruments:
        pitches.extend(int(n.pitch) for n in sorted(inst.notes, key=lambda n: n.start))
    return pitches


def test_remap_midi_file_pass_through_preserves_pitches(tmp_path):
    events = [(0.0, 36, 100), (0.5, 38, 90), (1.0, 45, 80), (1.5, 50, 70)]
    src = tmp_path / "gm.mid"
    write_midi(events, src)
    out = tmp_path / "sd3.mid"
    remap_midi_file(src, out, "superior_drummer_3")
    assert out.is_file()
    assert _drum_pitches(out) == [36, 38, 45, 50]


def test_remap_midi_file_non_identity(tmp_path, monkeypatch):
    profile = {"plugin": "Custom", "confidence": "low", "map": {"36": 60, "50": 40}}
    (tmp_path / "custom.json").write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.setattr(remap, "MAPPINGS_DIR", tmp_path)

    events = [(0.0, 36, 100), (0.5, 50, 70)]
    src = tmp_path / "gm.mid"
    write_midi(events, src)
    out = tmp_path / "custom.mid"
    remap_midi_file(src, out, "custom")
    assert _drum_pitches(out) == [60, 40]


def test_cli_list_returns_zero(capsys):
    assert remap.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "superior_drummer_3" in out
    assert "drumforge" in out


def test_cli_remap_writes_output(tmp_path):
    events = [(0.0, 36, 100), (0.5, 49, 90)]
    src = tmp_path / "gm.mid"
    write_midi(events, src)
    out = tmp_path / "out.mid"
    rc = remap.main([str(src), "--plugin", "general_midi", "-o", str(out)])
    assert rc == 0
    assert _drum_pitches(out) == [36, 49]


def test_cli_missing_args_returns_error():
    assert remap.main([]) == 2
