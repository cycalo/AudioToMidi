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
from pipeline.drum_voices import ALL_VOICES  # noqa: E402
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
    ), patch.object(controller_mod, "estimate_bpm", return_value=84.0):
        _run_analysis(ctrl, str(tmp_path / "drums.wav"))

    assert ctrl.state is not None
    assert ctrl.state.detected_bpm == 84.0

    custom = {"plugin": "Custom", "confidence": "low", "map": {36: 60, 50: 40}}
    ctrl.set_plugin("custom")
    out = tmp_path / "out.mid"
    done: list[str] = []
    ctrl.exportFinished.connect(lambda p: done.append(p))
    with patch.object(controller_mod, "load_profile", return_value=custom):
        ctrl.export_midi(str(out), tempo=84.0)

    assert done and out.is_file()
    pm = pretty_midi.PrettyMIDI(str(out))
    pitches = sorted(n.pitch for inst in pm.instruments for n in inst.notes)
    assert pitches == [40, 60]
    assert pm.get_tempo_changes()[1][0] == pytest.approx(84.0, abs=0.1)
    ctrl.cleanup()


def test_sensitivity_preserves_detected_bpm(qapp, tmp_path) -> None:
    ctrl = PipelineController()
    events = [(0.0, 36, 100)]
    with patch.object(controller_mod, "separate"), patch.object(
        controller_mod, "transcribe_stems", return_value=(events, {})
    ), patch.object(controller_mod, "estimate_bpm", return_value=96.0):
        _run_analysis(ctrl, str(tmp_path / "drums.wav"))

    updated: list[list] = []
    ctrl.eventsUpdated.connect(lambda e: updated.append(e))
    with patch.object(
        controller_mod, "transcribe_stems", return_value=([(0.0, 36, 90)], {})
    ):
        ctrl.set_delta_scale(0.5)
        _wait_until(lambda: bool(updated) and not ctrl.is_busy())

    assert ctrl.state is not None
    assert ctrl.state.detected_bpm == 96.0
    ctrl.cleanup()


def test_export_without_analysis_reports_error(qapp) -> None:
    ctrl = PipelineController()
    errors: list[str] = []
    ctrl.exportFailed.connect(lambda e: errors.append(e))
    ctrl.export_midi("unused.mid")
    assert errors


def test_reset_session_clears_state(qapp, tmp_path) -> None:
    ctrl = PipelineController()
    with patch.object(controller_mod, "separate"), patch.object(
        controller_mod, "transcribe_stems", return_value=([(0.0, 36, 100)], {})
    ):
        _run_analysis(ctrl, str(tmp_path / "drums.wav"))

    reset_seen: list[bool] = []
    ctrl.sessionReset.connect(lambda: reset_seen.append(True))
    ctrl.reset_session()

    assert ctrl.state is None
    assert not ctrl.has_preview_cache()
    assert reset_seen == [True]
    ctrl.cleanup()


def test_preview_play_uses_cache_without_worker(qapp, tmp_path) -> None:
    ctrl = PipelineController()
    buffer = np.zeros(4410, dtype=np.float32)
    ctrl._preview_cache[("midi", ALL_VOICES)] = (buffer, 44100)
    ctrl._state = controller_mod.AnalysisState(
        wav_path=str(tmp_path / "drums.wav"),
        stems_dir=str(tmp_path / "stems"),
        events=[(0.0, 36, 100)],
    )
    started: list[str] = []
    ctrl.previewStarted.connect(lambda m: started.append(m))

    with patch.object(ctrl, "_start_worker") as worker, patch.object(
        ctrl._playback, "play"
    ):
        ctrl.preview_play("midi")

    assert started == ["midi"]
    worker.assert_not_called()
    ctrl.preview_stop()
    ctrl.cleanup()


def test_preview_seek_updates_position(qapp, tmp_path) -> None:
    ctrl = PipelineController()
    buffer = np.zeros(88200, dtype=np.float32)
    ctrl._preview_cache[("midi", ALL_VOICES)] = (buffer, 44100)
    ctrl._preview_mode = "midi"
    positions: list[float] = []
    ctrl.previewPosition.connect(lambda p: positions.append(p))

    ctrl.preview_seek(1.5)
    assert positions[-1] == pytest.approx(1.5)
    assert ctrl._pending_start_s == pytest.approx(1.5)
    ctrl.cleanup()


def test_preview_seek_restarts_while_session_active(qapp, tmp_path) -> None:
    """Seek must restart audio when the preview timer is active even if PortAudio
    briefly reports inactive (regression for multi-click waveform scrub)."""
    ctrl = PipelineController()
    buffer = np.zeros(88200, dtype=np.float32)
    ctrl._preview_cache[("midi", ALL_VOICES)] = (buffer, 44100)
    ctrl._preview_mode = "midi"
    ctrl._preview_timer.start()
    with patch.object(ctrl._playback, "is_playing", return_value=False), patch.object(
        ctrl._playback, "play"
    ) as play:
        ctrl.preview_seek(2.0)
    play.assert_called_once()
    assert ctrl._preview_timer.isActive()
    ctrl.cleanup()


def test_set_transcription_version_forwarded(qapp, tmp_path) -> None:
    ctrl = PipelineController()
    ctrl.set_transcription_version("v1")
    assert ctrl.transcription_version == "v1"
    with patch.object(controller_mod, "separate"), patch.object(
        controller_mod, "transcribe_stems", return_value=([(0.0, 36, 100)], {})
    ) as ts:
        _run_analysis(ctrl, str(tmp_path / "drums.wav"))
    ts.assert_called()
    assert ts.call_args.kwargs.get("transcription_version") == "v1"
    ctrl.cleanup()


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
