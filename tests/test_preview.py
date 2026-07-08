"""Tests for sample-based MIDI preview rendering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.preview import (  # noqa: E402
    build_preview_buffer,
    load_preview_kit,
    mix_sources,
    render_preview,
)
from pipeline.remap import load_profile, remap_events  # noqa: E402

PREVIEW_KIT_DIR = REPO_ROOT / "Preview Kit"
SR = 44100


def _write_tone(path: Path, *, freq: float = 440.0, duration_s: float = 0.05) -> None:
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False, dtype=np.float32)
    tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), tone, SR)


@pytest.fixture
def mini_kit(tmp_path: Path) -> Path:
    kit_dir = tmp_path / "kit"
    kit_dir.mkdir()
    _write_tone(kit_dir / "kick.wav", freq=100.0)
    _write_tone(kit_dir / "snare.wav", freq=200.0)
    _write_tone(kit_dir / "hat.wav", freq=8000.0)
    manifest = {
        "name": "test",
        "sample_rate": SR,
        "samples": {"36": "kick.wav", "38": "snare.wav", "54": "hat.wav"},
    }
    (kit_dir / "kit.json").write_text(json.dumps(manifest), encoding="utf-8")
    return kit_dir


def test_load_preview_kit_maps_ggd_notes():
    if not PREVIEW_KIT_DIR.is_dir():
        pytest.skip("Preview Kit not present")
    kit = load_preview_kit(PREVIEW_KIT_DIR)
    assert kit.sample_rate == 44100
    assert 36 in kit.samples
    assert 54 in kit.samples
    assert 48 in kit.samples
    assert kit.samples[43].shape == kit.samples[48].shape


def test_render_single_hit_at_correct_offset(mini_kit: Path):
    kit = load_preview_kit(mini_kit)
    events = [(0.5, 36, 127)]
    out = render_preview(events, kit)
    start = int(0.5 * SR)
    assert float(np.max(np.abs(out[:start]))) == pytest.approx(0.0, abs=1e-6)
    assert float(np.max(np.abs(out[start : start + 500]))) > 0.01


def test_render_velocity_scales_amplitude(mini_kit: Path):
    kit = load_preview_kit(mini_kit)
    loud = render_preview([(0.0, 36, 127)], kit)
    quiet = render_preview([(0.0, 36, 40)], kit)
    assert float(np.max(np.abs(loud))) > float(np.max(np.abs(quiet)))


def test_render_skips_unknown_notes(mini_kit: Path):
    kit = load_preview_kit(mini_kit)
    out = render_preview([(0.0, 57, 100)], kit)
    assert float(np.max(np.abs(out))) == pytest.approx(0.0, abs=1e-6)


def test_mix_sources_both(mini_kit: Path):
    kit = load_preview_kit(mini_kit)
    midi_buf = render_preview([(0.0, 36, 127)], kit)
    original = np.ones_like(midi_buf) * 0.1
    mixed = mix_sources(midi_buf, original, "both")
    assert float(np.max(np.abs(mixed))) > float(np.max(np.abs(midi_buf)))


def test_ggd_remap_then_render(mini_kit: Path):
    profile = load_profile("ggd")
    events = [(0.0, 42, 100)]
    remapped = remap_events(events, profile)
    assert remapped[0][1] == 54
    out = render_preview(remapped, load_preview_kit(mini_kit))
    assert float(np.max(np.abs(out))) > 0.01


def test_build_preview_buffer_original_mode(tmp_path: Path, mini_kit: Path):
    wav = tmp_path / "stem.wav"
    _write_tone(wav, freq=300.0, duration_s=0.2)
    kit = load_preview_kit(mini_kit)
    buf, sr = build_preview_buffer(
        [(0.0, 36, 127)],
        kit,
        wav_path=str(wav),
        mode="original",
        duration_hint_s=0.2,
    )
    assert sr == SR
    assert buf.size > 0
    assert float(np.max(np.abs(buf))) > 0.01
