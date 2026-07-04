"""Phase 2 tests: source separation plumbing and structural checks.

These tests verify the *plumbing* (stem files, lengths, sample rates, manifest,
and that each DSP mask routes its frequency band to the right stem). They do NOT
assert perceptual separation quality -- that is a manual by-ear check on a real
drum stem (see tests/fixtures/README.md).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.separation import (  # noqa: E402
    DEMUCS_STEMS,
    DSP_STEMS,
    MODELS_DIR,
    DRUMSEP_FILENAME,
    resolve_device,
    separate,
)

SR = 44100
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _find_fixture() -> Path | None:
    wavs = sorted(FIXTURES.glob("*.wav"))
    return wavs[0] if wavs else None


def _click_train(sr: int = SR, duration: float = 2.0) -> np.ndarray:
    """Broadband impulse train (percussive, flat spectrum) for band-routing checks."""
    y = np.zeros(int(sr * duration), dtype=np.float32)
    for t in np.arange(0.2, duration, 0.2):
        y[int(t * sr)] = 1.0
    return y


def _energy_fraction(y: np.ndarray, sr: int, bands) -> float:
    """Fraction of spectral energy of ``y`` that falls within ``bands`` (list of (lo, hi))."""
    spectrum = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    in_band = np.zeros_like(spectrum, dtype=bool)
    for lo, hi in bands:
        in_band |= (freqs >= lo) & (freqs <= hi)
    return float(spectrum[in_band].sum() / total)


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y**2)))


# --------------------------------------------------------------------------
# DSP backend: deterministic band-routing (plumbing only)
# --------------------------------------------------------------------------
def test_dsp_band_routing(tmp_path: Path) -> None:
    src = tmp_path / "clicks.wav"
    click = _click_train()
    sf.write(str(src), click, SR)

    out_dir = tmp_path / "stems"
    manifest = separate(str(src), str(out_dir), backend="dsp")

    assert manifest["backend"] == "dsp"
    assert set(manifest["stems"]) == set(DSP_STEMS)
    assert (out_dir / "manifest.json").exists()

    stems = {name: sf.read(path)[0] for name, path in manifest["stems"].items()}

    # Every stem carries some energy from the broadband clicks.
    for name, audio in stems.items():
        assert _rms(audio) > 0, f"{name} stem is silent"
        assert len(audio) == len(click), f"{name} length not preserved"

    # Each stem's energy concentrates in its designed band.
    assert _energy_fraction(stems["kick"], SR, [(0, 150)]) > 0.6
    assert _energy_fraction(stems["toms"], SR, [(60, 450)]) > 0.5
    assert _energy_fraction(stems["snare"], SR, [(120, 350), (1800, 4200)]) > 0.5
    assert _energy_fraction(stems["hihat"], SR, [(5000, SR / 2)]) > 0.6
    assert _energy_fraction(stems["cymbals"], SR, [(2500, SR / 2)]) > 0.6

    # Hi-hat owns the 6-12 kHz band; cymbals are hi-hat-subtracted there.
    hihat_hf = _energy_fraction(stems["hihat"], SR, [(6000, 12000)])
    cymbals_hf = _energy_fraction(stems["cymbals"], SR, [(6000, 12000)])
    assert hihat_hf > cymbals_hf


def test_unknown_backend_raises(tmp_path: Path) -> None:
    src = tmp_path / "clicks.wav"
    sf.write(str(src), _click_train(duration=0.5), SR)
    with pytest.raises(ValueError):
        separate(str(src), str(tmp_path / "out"), backend="nope")


def test_resolve_device_forced() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "cuda")


# --------------------------------------------------------------------------
# Real-fixture structural checks (skip when no real stem is present)
# --------------------------------------------------------------------------
@pytest.mark.skipif(_find_fixture() is None, reason="no real drum stem in tests/fixtures/")
def test_dsp_real_fixture(tmp_path: Path) -> None:
    fixture = _find_fixture()
    info = sf.info(str(fixture))

    manifest = separate(str(fixture), str(tmp_path / "stems"), backend="dsp")

    assert set(manifest["stems"]) == set(DSP_STEMS)
    assert manifest["sample_rate"] == info.samplerate
    for name, path in manifest["stems"].items():
        data, sr = sf.read(path)
        assert sr == info.samplerate
        assert len(data) == info.frames, f"{name} length not preserved"
        assert _rms(np.asarray(data)) > 0, f"{name} stem is silent"


_DEMUCS_AVAILABLE = (MODELS_DIR / DRUMSEP_FILENAME).exists() or os.environ.get(
    "AUDIOTOMIDI_RUN_DEMUCS_TESTS"
) == "1"


@pytest.mark.skipif(
    _find_fixture() is None or not _DEMUCS_AVAILABLE,
    reason="needs a real stem and the cached drumsep checkpoint (or AUDIOTOMIDI_RUN_DEMUCS_TESTS=1)",
)
def test_demucs_real_fixture(tmp_path: Path) -> None:
    fixture = _find_fixture()
    manifest = separate(str(fixture), str(tmp_path / "stems"), backend="demucs", device="cpu")

    assert manifest["backend"] == "demucs"
    assert set(manifest["stems"]) == set(DEMUCS_STEMS)
    for name, path in manifest["stems"].items():
        data, _ = sf.read(path)
        assert _rms(np.asarray(data)) > 0, f"{name} stem is silent"
