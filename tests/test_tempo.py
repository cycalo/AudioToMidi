"""Tests for BPM estimation and normalization."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.tempo import estimate_bpm, normalize_bpm  # noqa: E402

SAMPLE_RATE = 22050


def test_normalize_bpm_clamps_low() -> None:
    assert normalize_bpm(30.0) == 40.0


def test_normalize_bpm_clamps_high() -> None:
    assert normalize_bpm(300.0) == 240.0


def test_normalize_bpm_nan_falls_back() -> None:
    assert normalize_bpm(float("nan")) == 120.0


def test_normalize_bpm_passes_valid() -> None:
    assert normalize_bpm(84.0) == 84.0


def test_normalize_bpm_rounds_to_whole_number() -> None:
    assert normalize_bpm(84.4) == 84.0
    assert normalize_bpm(84.6) == 85.0
    assert normalize_bpm(119.5) == 120.0


def _write_click_track(path: Path, bpm: float, bars: int = 8) -> None:
    """Write a simple click WAV at a constant BPM (one click per quarter note)."""
    beat_period = 60.0 / bpm
    duration = bars * 4 * beat_period
    n = int(SAMPLE_RATE * duration)
    y = np.zeros(n, dtype=np.float32)
    click_len = int(0.01 * SAMPLE_RATE)
    click = np.linspace(1.0, 0.0, click_len, dtype=np.float32)
    t = 0.0
    while t < duration:
        start = int(t * SAMPLE_RATE)
        end = min(start + click_len, n)
        y[start:end] += click[: end - start]
        t += beat_period
    sf.write(str(path), y, SAMPLE_RATE)


def test_estimate_bpm_on_click_track(tmp_path: Path) -> None:
    wav = tmp_path / "clicks_120.wav"
    _write_click_track(wav, 120.0)

    bpm = estimate_bpm(wav)

    # Allow exact match, ±3 BPM, or exact half/double (no auto-correction in v1).
    assert bpm == pytest.approx(120.0, abs=3.0) or bpm == pytest.approx(
        60.0, abs=3.0
    ) or bpm == pytest.approx(240.0, abs=3.0)


def test_estimate_bpm_missing_file_falls_back(tmp_path: Path) -> None:
    assert estimate_bpm(tmp_path / "missing.wav") == 120.0
