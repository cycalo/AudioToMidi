"""Wires the UI to the transcription pipeline.

Phase 5: ``PipelineController`` runs the full pipeline (separation -> onset
detection -> merge) in a background thread so the UI stays responsive, holds the
resulting General MIDI events for review, re-runs detection when the sensitivity
slider changes (fast, on the cached stems), and exports a plugin-remapped ``.mid``
on demand.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import QObject, QThread, QTimer, Signal

_DEV_ROOT = Path(__file__).resolve().parent.parent
if str(_DEV_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEV_ROOT))

from app.audio_playback import AudioPlayback  # noqa: E402
from pipeline.drum_voices import ALL_VOICES, filter_gm_events_by_voices  # noqa: E402
from pipeline.merge import transcribe_stems  # noqa: E402
from pipeline.midi_writer import Event, write_midi  # noqa: E402
from pipeline.paths import repo_root  # noqa: E402
from pipeline.preview import (  # noqa: E402
    PreviewMode,
    build_preview_buffer,
    load_preview_kit,
    resolve_kit_dir,
)
from pipeline.remap import load_profile, remap_events  # noqa: E402
from pipeline.separation import separate  # noqa: E402
from pipeline.tempo import estimate_bpm, normalize_bpm  # noqa: E402

PreviewCacheKey = Tuple[str, FrozenSet[str]]
REPO_ROOT = repo_root()


def _separation_progress_message(device: str) -> str:
    if device == "cuda":
        return "Separating stems on GPU..."
    if device == "cpu":
        return "Separating stems on CPU (this can take a while)..."
    return "Separating stems (auto device — GPU if available)..."


@dataclass
class AnalysisState:
    """Holds the result of analyzing one WAV, reused for review and export."""

    wav_path: str
    stems_dir: str
    events: List[Event] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    delta_scale: float = 1.0
    transcription_version: str = "primary"
    detected_bpm: float = 120.0


class _Worker(QObject):
    """Runs a callable off the UI thread, forwarding progress and the result.

    The callable receives a ``report(message, percent)`` callback it can use to
    emit progress updates.
    """

    progress = Signal(str, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[[Callable[[str, int], None]], object]) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn(lambda msg, pct: self.progress.emit(msg, int(pct)))
        except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the UI
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


class PipelineController(QObject):
    """Mediates between the GUI widgets and the transcription pipeline."""

    progress = Signal(str, int)
    analysisFinished = Signal(object)  # AnalysisState
    analysisFailed = Signal(str)
    eventsUpdated = Signal(object)  # List[Event]
    exportFinished = Signal(str)
    exportFailed = Signal(str)
    previewStarted = Signal(str)  # mode label
    previewFinished = Signal()
    previewFailed = Signal(str)
    previewPosition = Signal(float)
    sessionReset = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._state: Optional[AnalysisState] = None
        self._plugin_id: str = "ggd"
        self._device: str = "auto"
        self._thread: Optional[QThread] = None
        self._worker: Optional[_Worker] = None
        self._worker_kind: str = ""
        self._busy: bool = False
        self._playback = AudioPlayback()
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(50)
        self._preview_timer.timeout.connect(self._poll_preview_position)
        self._preview_cache: Dict[PreviewCacheKey, Tuple[np.ndarray, int]] = {}
        self._preview_mode: Optional[PreviewMode] = None
        self._pending_start_s: float = 0.0
        self._pending_resume_playing: bool = False
        self._transcription_version: str = "primary"
        self._voice_filter: FrozenSet[str] = ALL_VOICES

    # -- configuration -----------------------------------------------------
    def set_plugin(self, plugin_id: str) -> None:
        """Select the plugin profile applied at export time."""
        self._plugin_id = plugin_id

    @property
    def plugin_id(self) -> str:
        return self._plugin_id

    def preview_supported(self) -> bool:
        """Preview playback is available for profiles with a preview kit (v1: GGD)."""
        if self._plugin_id != "ggd":
            return False
        try:
            profile = load_profile(self._plugin_id)
            kit_dir = resolve_kit_dir(profile, repo_root=REPO_ROOT)
            return (kit_dir / "kit.json").is_file()
        except (ValueError, FileNotFoundError):
            return False

    def is_preview_playing(self) -> bool:
        return self._playback.is_playing()

    def has_preview_cache(self) -> bool:
        return bool(self._preview_cache)

    @property
    def voice_filter(self) -> FrozenSet[str]:
        return self._voice_filter

    def set_voice_filter(self, voices: FrozenSet[str]) -> None:
        """Set active drum voices for preview playback and update audio if needed."""
        if voices == self._voice_filter:
            return
        self._voice_filter = voices
        if self._state is None or not self.preview_supported() or self.is_busy():
            return
        was_playing = self._playback.is_playing()
        pos = (
            self._playback.position_seconds()
            if was_playing
            else self._pending_start_s
        )
        mode: PreviewMode = self._preview_mode or "midi"
        if was_playing:
            self._preview_timer.stop()
            self._playback.stop()
        self._pending_start_s = pos
        self._pending_resume_playing = was_playing
        cache_key = self._preview_cache_key(mode)
        if cache_key in self._preview_cache:
            buffer, sr = self._preview_cache[cache_key]
            duration = float(buffer.size) / sr if sr else 0.0
            pos = max(0.0, min(pos, duration))
            self._pending_start_s = pos
            if was_playing:
                self._start_preview_playback(buffer, sr, mode, pos)
            else:
                self.previewPosition.emit(pos)
            return
        self._start_preview_render(mode)

    def set_device(self, device: str) -> None:
        """Select the compute device for Demucs separation (``auto``, ``cpu``, ``cuda``)."""
        self._device = device

    def set_transcription_version(self, version: str) -> None:
        """Kept for API compatibility; detection is always kick/snare/toms."""
        self._transcription_version = version or "primary"

    @property
    def transcription_version(self) -> str:
        return self._transcription_version

    @property
    def state(self) -> Optional[AnalysisState]:
        return self._state

    def is_busy(self) -> bool:
        return self._busy

    # -- analysis (separation + detection) ---------------------------------
    def run_analysis(self, wav_path: str) -> None:
        """Separate ``wav_path`` and transcribe its stems in a background thread."""
        if self.is_busy():
            return
        self.preview_stop()
        self._clear_preview_cache()
        self._discard_previous_stems()
        stems_dir = tempfile.mkdtemp(prefix="audiotomidi_")

        def job(report: Callable[[str, int], None]) -> AnalysisState:
            report(_separation_progress_message(self._device), 5)
            separate(wav_path, stems_dir, device=self._device)
            report("Detecting onsets...", 80)
            events, summary = transcribe_stems(
                stems_dir,
                transcription_version=self._transcription_version,
            )
            report("Estimating tempo...", 95)
            bpm = estimate_bpm(wav_path)
            report("Ready for review", 100)
            return AnalysisState(
                wav_path=wav_path,
                stems_dir=stems_dir,
                events=events,
                summary=summary,
                delta_scale=1.0,
                transcription_version=self._transcription_version,
                detected_bpm=bpm,
            )

        self._start_worker(job, "analysis")

    # -- interactive sensitivity re-detection ------------------------------
    def set_delta_scale(self, scale: float) -> None:
        """Re-run onset detection on the cached stems at a new sensitivity."""
        if self._state is None or self.is_busy():
            return
        stems_dir = self._state.stems_dir

        wav_path = self._state.wav_path
        detected_bpm = self._state.detected_bpm

        def job(report: Callable[[str, int], None]) -> AnalysisState:
            events, summary = transcribe_stems(
                stems_dir,
                delta_scale=scale,
                transcription_version=self._transcription_version,
            )
            return AnalysisState(
                wav_path=wav_path,
                stems_dir=stems_dir,
                events=events,
                summary=summary,
                delta_scale=scale,
                transcription_version=self._transcription_version,
                detected_bpm=detected_bpm,
            )

        self._start_worker(job, "detect")

    def retranscribe(self) -> None:
        """Re-run onset detection on cached stems (version or sensitivity change)."""
        if self._state is None or self.is_busy():
            return
        self._clear_preview_cache()
        scale = self._state.delta_scale
        stems_dir = self._state.stems_dir
        wav_path = self._state.wav_path
        version = self._transcription_version
        detected_bpm = self._state.detected_bpm

        def job(_report: Callable[[str, int], None]) -> AnalysisState:
            events, summary = transcribe_stems(
                stems_dir,
                delta_scale=scale,
                transcription_version=version,
            )
            return AnalysisState(
                wav_path=wav_path,
                stems_dir=stems_dir,
                events=events,
                summary=summary,
                delta_scale=scale,
                transcription_version=version,
                detected_bpm=detected_bpm,
            )

        self._start_worker(job, "detect")

    # -- export ------------------------------------------------------------
    def export_midi(self, output_path: str, *, tempo: float = 120.0) -> None:
        """Remap the current events to the selected plugin and write a ``.mid``.

        Fast (in-memory remap + small file write), so it runs on the caller's
        thread rather than a worker. ``tempo`` must match the DAW project BPM.
        """
        if self._state is None:
            self.exportFailed.emit("Nothing to export yet; run Convert first.")
            return
        try:
            profile = load_profile(self._plugin_id)
            events = remap_events(self._state.events, profile)
            write_midi(events, output_path, tempo=normalize_bpm(tempo))
        except Exception as exc:  # noqa: BLE001 - report any load/write failure
            self.exportFailed.emit(str(exc))
            return
        self.exportFinished.emit(output_path)

    # -- preview playback --------------------------------------------------
    def preview_play(self, mode: PreviewMode) -> None:
        """Render and play a preview of the remapped events through the preview kit."""
        if self.is_busy():
            return
        if self._state is None:
            self.previewFailed.emit("Nothing to preview yet; run Convert first.")
            return
        if not self.preview_supported():
            self.previewFailed.emit(
                "Preview is only available for GetGood Drums with a bundled Preview Kit."
            )
            return
        if self._playback.is_playing():
            self._preview_timer.stop()
            self._playback.stop()

        cache_key = self._preview_cache_key(mode)
        if cache_key in self._preview_cache:
            buffer, sr = self._preview_cache[cache_key]
            self._start_preview_playback(buffer, sr, mode, self._pending_start_s)
            return

        self._pending_resume_playing = True
        self._start_preview_render(mode)

    def preview_change_source(self, mode: PreviewMode) -> None:
        """Switch preview source, keeping playback position when possible."""
        if self.is_busy() or self._state is None or not self.preview_supported():
            return
        if mode == self._preview_mode and self._preview_cache_key(mode) in self._preview_cache:
            return

        pos = (
            self._playback.position_seconds()
            if self._playback.is_playing()
            else self._pending_start_s
        )
        was_playing = self._playback.is_playing()

        cache_key = self._preview_cache_key(mode)
        if cache_key in self._preview_cache:
            buffer, sr = self._preview_cache[cache_key]
            self._preview_mode = mode
            duration = float(buffer.size) / sr if sr else 0.0
            pos = max(0.0, min(pos, duration))
            self._pending_start_s = pos
            if was_playing:
                self._start_preview_playback(buffer, sr, mode, pos)
            else:
                self.previewPosition.emit(pos)
            return

        if was_playing:
            self._preview_timer.stop()
            self._playback.stop()

        self._pending_start_s = pos
        self._pending_resume_playing = was_playing
        self._start_preview_render(mode)

    def preview_seek(self, time_s: float) -> None:
        """Jump preview playback to ``time_s`` seconds."""
        mode = self._preview_mode
        if mode is None:
            return
        cache_key = self._preview_cache_key(mode)
        if cache_key not in self._preview_cache:
            return
        buffer, sr = self._preview_cache[cache_key]
        duration = float(buffer.size) / sr if sr else 0.0
        time_s = max(0.0, min(float(time_s), duration))
        self._pending_start_s = time_s
        in_session = self._preview_timer.isActive() or self._playback.is_playing()
        if in_session:
            try:
                self._playback.play(buffer, sr, start_s=time_s)
            except Exception as exc:  # noqa: BLE001
                self.previewFailed.emit(str(exc))
                return
            if not self._preview_timer.isActive():
                self._preview_timer.start()
        self.previewPosition.emit(time_s)

    def preview_reset_position(self) -> None:
        """Move the preview playhead back to the start."""
        self.preview_seek(0.0)

    def preview_stop(self) -> None:
        """Stop preview playback."""
        self._preview_timer.stop()
        self._playback.stop()
        self._pending_start_s = 0.0
        self.previewPosition.emit(0.0)
        self.previewFinished.emit()

    def reset_session(self) -> None:
        """Stop playback and discard all analysis and preview state."""
        if self.is_busy():
            return
        self.preview_stop()
        self._clear_preview_cache()
        self._preview_mode = None
        self._pending_start_s = 0.0
        self._pending_resume_playing = False
        self._voice_filter = ALL_VOICES
        self._discard_previous_stems()
        self._state = None
        self.sessionReset.emit()

    # -- teardown ----------------------------------------------------------
    def cleanup(self) -> None:
        """Remove cached stems (call on application shutdown)."""
        self.preview_stop()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(30_000)
        self._discard_previous_stems()

    # -- internals ---------------------------------------------------------
    def _discard_previous_stems(self) -> None:
        if self._state is not None and self._state.stems_dir:
            shutil.rmtree(self._state.stems_dir, ignore_errors=True)

    def _clear_preview_cache(self) -> None:
        self._preview_cache.clear()
        self._preview_mode = None

    def _preview_cache_key(self, mode: PreviewMode) -> PreviewCacheKey:
        return (mode, self._voice_filter)

    def _start_preview_render(self, mode: PreviewMode) -> None:
        wav_path = self._state.wav_path  # type: ignore[union-attr]
        events = list(self._state.events)  # type: ignore[union-attr]
        stems_dir = self._state.stems_dir  # type: ignore[union-attr]
        plugin_id = self._plugin_id
        voices = self._voice_filter

        def job(_report: Callable[[str, int], None]) -> tuple:
            import librosa

            profile = load_profile(plugin_id)
            filtered = filter_gm_events_by_voices(events, voices)
            remapped = remap_events(filtered, profile)
            kit_dir = resolve_kit_dir(profile, repo_root=REPO_ROOT)
            kit = load_preview_kit(kit_dir)
            duration_hint = float(librosa.get_duration(path=wav_path))
            buffer, sr = build_preview_buffer(
                remapped,
                kit,
                wav_path=wav_path,
                mode=mode,
                duration_hint_s=duration_hint,
                voices=voices,
                stems_dir=stems_dir,
            )
            return buffer, sr, mode

        self._start_worker(job, "preview")

    def _start_preview_playback(
        self,
        buffer: np.ndarray,
        sr: int,
        mode: PreviewMode,
        start_s: float,
    ) -> None:
        duration = float(buffer.size) / sr if sr else 0.0
        start_s = max(0.0, min(float(start_s), duration))
        try:
            self._playback.play(buffer, sr, start_s=start_s)
        except Exception as exc:  # noqa: BLE001
            self.previewFailed.emit(str(exc))
            return
        self._preview_mode = mode
        self._pending_start_s = start_s
        self._preview_timer.start()
        self.previewStarted.emit(mode)
        self.previewPosition.emit(start_s)

    def _start_worker(
        self,
        job: Callable[[Callable[[str, int], None]], object],
        kind: str,
    ) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        thread = QThread()
        worker = _Worker(job)
        worker.moveToThread(thread)

        # Bound-method slots on this QObject (main thread) are auto-queued, so the
        # result is handled on the main thread. thread.quit / deleteLater run
        # safely without the worker waiting on its own thread.
        thread.started.connect(worker.run)
        worker.progress.connect(self.progress)
        worker.finished.connect(self._on_worker_finished)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)

        self._thread = thread
        self._worker = worker
        self._worker_kind = kind
        self._busy = True
        thread.start()

    def _on_thread_finished(self) -> None:
        thread = self._thread
        self._clear_thread_refs()
        if thread is not None:
            thread.deleteLater()

    def _on_worker_finished(self, result: object) -> None:
        kind = self._worker_kind
        self._busy = False

        if kind == "preview":
            buffer, sr, mode = result  # type: ignore[misc]
            self._preview_cache[self._preview_cache_key(mode)] = (buffer, sr)
            if self._pending_resume_playing:
                self._start_preview_playback(buffer, sr, mode, self._pending_start_s)
            else:
                self._preview_mode = mode
                self.previewPosition.emit(self._pending_start_s)
            self._pending_resume_playing = False
            return

        assert isinstance(result, AnalysisState)
        self._state = result
        if kind == "analysis":
            self.analysisFinished.emit(result)
        else:
            self._clear_preview_cache()
            self.eventsUpdated.emit(result.events)

    def _poll_preview_position(self) -> None:
        if not self._playback.is_playing():
            duration = self._playback.duration_seconds()
            near_end = duration > 0 and self._pending_start_s >= duration - 0.05
            # Ignore transient inactive states while a preview session is active
            # (e.g. rapid waveform seeks restarting PortAudio).
            if (
                self._preview_timer.isActive()
                and self._preview_mode is not None
                and not near_end
            ):
                return
            self._preview_timer.stop()
            self._pending_start_s = 0.0
            self.previewPosition.emit(0.0)
            self.previewFinished.emit()
            return
        pos = self._playback.position_seconds()
        self._pending_start_s = pos
        self.previewPosition.emit(pos)

    def _on_worker_failed(self, error: str) -> None:
        kind = self._worker_kind
        self._busy = False
        if kind == "preview":
            self.previewFailed.emit(error)
            return
        self.analysisFailed.emit(error)

    def _clear_thread_refs(self) -> None:
        self._thread = None
        self._worker = None
