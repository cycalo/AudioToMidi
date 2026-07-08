"""Phase 5 tests: PipelineController orchestration and delta-scale plumbing.

The heavy pipeline calls (separation, transcription) are mocked so these tests
stay fast and deterministic; a focused unit test confirms the sensitivity knob
forwards a scaled, clamped ``delta`` down to ``librosa`` onset detection.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pretty_midi
import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import controller as controller_mod  # noqa: E402
from app.controller import PipelineController  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _wait_until(predicate, timeout_ms: int = 8000) -> None:
    """Spin the Qt event loop until ``predicate()`` is true or the timeout hits."""
    if predicate():
        return
    loop = QEventLoop()
    poll = QTimer()
    poll.setInterval(20)
    poll.timeout.connect(lambda: predicate() and loop.quit())
    QTimer.singleShot(timeout_ms, loop.quit)
    poll.start()
    loop.exec()
    poll.stop()


def _run_analysis(ctrl: PipelineController, wav_path: str) -> None:
    """Start analysis and wait until it finishes and the worker is fully idle."""
    ctrl.run_analysis(wav_path)
    _wait_until(lambda: ctrl.state is not None and not ctrl.is_busy())


def test_run_analysis_populates_events_and_progress(qapp, tmp_path) -> None:
    ctrl = PipelineController()
    progress: list[tuple[str, int]] = []
    busy_on_finish: list[bool] = []
    ctrl.progress.connect(lambda m, p: progress.append((m, p)))
    ctrl.analysisFinished.connect(lambda _s: busy_on_finish.append(ctrl.is_busy()))

    events = [(0.0, 36, 100), (0.5, 38, 90)]
    with patch.object(controller_mod, "separate") as sep, patch.object(
        controller_mod, "transcribe_stems", return_value=(events, {"total_events": 2})
    ):
        _run_analysis(ctrl, str(tmp_path / "drums.wav"))

    assert ctrl.state is not None
    assert len(ctrl.state.events) == 2
    assert busy_on_finish == [False]
    assert ctrl._thread is None
    sep.assert_called_once()
    messages = [m for m, _ in progress]
    assert any("Separating" in m for m in messages)
    assert any("Detecting" in m for m in messages)
    ctrl.cleanup()


def test_set_device_forwards_to_separate(qapp, tmp_path) -> None:
    ctrl = PipelineController()
    ctrl.set_device("cuda")
    with patch.object(controller_mod, "separate") as sep, patch.object(
        controller_mod, "transcribe_stems", return_value=([(0.0, 36, 100)], {})
    ):
        _run_analysis(ctrl, str(tmp_path / "drums.wav"))
    sep.assert_called_once()
    assert sep.call_args.kwargs["device"] == "cuda"
    ctrl.cleanup()


def test_set_delta_scale_forwards_scale_and_emits_events(qapp, tmp_path) -> None:
    ctrl = PipelineController()
    with patch.object(controller_mod, "separate"), patch.object(
        controller_mod, "transcribe_stems", return_value=([(0.0, 36, 100)], {})
    ):
        _run_analysis(ctrl, str(tmp_path / "drums.wav"))

    updated: list[list] = []
    ctrl.eventsUpdated.connect(lambda e: updated.append(e))
    new_events = [(0.0, 36, 100), (0.1, 38, 90), (0.2, 49, 70)]
    with patch.object(
        controller_mod, "transcribe_stems", return_value=(new_events, {})
    ) as ts:
        ctrl.set_delta_scale(0.5)
        _wait_until(lambda: bool(updated) and not ctrl.is_busy())

    assert updated and len(updated[-1]) == 3
    ts.assert_called_once()
    assert ts.call_args.kwargs["delta_scale"] == 0.5
    ctrl.cleanup()


def test_export_midi_applies_selected_plugin(qapp, tmp_path) -> None:
    ctrl = PipelineController()
    events = [(0.0, 36, 100), (0.5, 50, 80)]
    with patch.object(controller_mod, "separate"), patch.object(
        controller_mod, "transcribe_stems", return_value=(events, {})
    ):
        _run_analysis(ctrl, str(tmp_path / "drums.wav"))

    custom = {"plugin": "Custom", "confidence": "low", "map": {36: 60, 50: 40}}
    ctrl.set_plugin("custom")
    out = tmp_path / "out.mid"
    done: list[str] = []
    ctrl.exportFinished.connect(lambda p: done.append(p))
    with patch.object(controller_mod, "load_profile", return_value=custom):
        ctrl.export_midi(str(out))

    assert done and out.is_file()
    pm = pretty_midi.PrettyMIDI(str(out))
    pitches = sorted(n.pitch for inst in pm.instruments for n in inst.notes)
    assert pitches == [40, 60]
    ctrl.cleanup()


def test_export_without_analysis_reports_error(qapp) -> None:
    ctrl = PipelineController()
    errors: list[str] = []
    ctrl.exportFailed.connect(lambda e: errors.append(e))
    ctrl.export_midi("unused.mid")
    assert errors


def test_detect_onsets_for_stem_forwards_and_clamps_delta_scale() -> None:
    """The sensitivity scale multiplies the preset delta and is clamped to range."""
    import pipeline.onset_detection as od

    y = (np.random.default_rng(0).standard_normal(44100) * 0.1).astype(np.float32)
    preset = od.STEM_PRESETS["snare"]

    with patch.object(
        od.librosa.onset, "onset_detect", return_value=np.array([], dtype=int)
    ) as detect:
        od.detect_onsets_for_stem(y, 44100, preset, delta_scale=0.5)
    assert detect.call_args.kwargs["delta"] == pytest.approx(preset.delta * 0.5)

    # Above-range scale clamps to DELTA_SCALE_MAX.
    with patch.object(
        od.librosa.onset, "onset_detect", return_value=np.array([], dtype=int)
    ) as detect:
        od.detect_onsets_for_stem(y, 44100, preset, delta_scale=99.0)
    assert detect.call_args.kwargs["delta"] == pytest.approx(
        preset.delta * od.DELTA_SCALE_MAX
    )
