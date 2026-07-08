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
from typing import Callable, List, Optional

from PySide6.QtCore import QObject, QThread, Signal

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.merge import transcribe_stems  # noqa: E402
from pipeline.midi_writer import Event, write_midi  # noqa: E402
from pipeline.remap import load_profile, remap_events  # noqa: E402
from pipeline.separation import separate  # noqa: E402


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

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._state: Optional[AnalysisState] = None
        self._plugin_id: str = "general_midi"
        self._device: str = "auto"
        self._thread: Optional[QThread] = None
        self._worker: Optional[_Worker] = None
        self._worker_kind: str = ""

    # -- configuration -----------------------------------------------------
    def set_plugin(self, plugin_id: str) -> None:
        """Select the plugin profile applied at export time."""
        self._plugin_id = plugin_id

    def set_device(self, device: str) -> None:
        """Select the compute device for Demucs separation (``auto``, ``cpu``, ``cuda``)."""
        self._device = device

    @property
    def state(self) -> Optional[AnalysisState]:
        return self._state

    def is_busy(self) -> bool:
        return self._thread is not None

    # -- analysis (separation + detection) ---------------------------------
    def run_analysis(self, wav_path: str) -> None:
        """Separate ``wav_path`` and transcribe its stems in a background thread."""
        if self.is_busy():
            return
        self._discard_previous_stems()
        stems_dir = tempfile.mkdtemp(prefix="audiotomidi_")

        def job(report: Callable[[str, int], None]) -> AnalysisState:
            report(_separation_progress_message(self._device), 5)
            separate(wav_path, stems_dir, device=self._device)
            report("Detecting onsets...", 80)
            events, summary = transcribe_stems(stems_dir)
            report("Ready for review", 100)
            return AnalysisState(
                wav_path=wav_path,
                stems_dir=stems_dir,
                events=events,
                summary=summary,
                delta_scale=1.0,
            )

        self._start_worker(job, "analysis")

    # -- interactive sensitivity re-detection ------------------------------
    def set_delta_scale(self, scale: float) -> None:
        """Re-run onset detection on the cached stems at a new sensitivity."""
        if self._state is None or self.is_busy():
            return
        stems_dir = self._state.stems_dir

        wav_path = self._state.wav_path

        def job(report: Callable[[str, int], None]) -> AnalysisState:
            events, summary = transcribe_stems(stems_dir, delta_scale=scale)
            return AnalysisState(
                wav_path=wav_path,
                stems_dir=stems_dir,
                events=events,
                summary=summary,
                delta_scale=scale,
            )

        self._start_worker(job, "detect")

    # -- export ------------------------------------------------------------
    def export_midi(self, output_path: str, *, tempo: float = 120.0) -> None:
        """Remap the current events to the selected plugin and write a ``.mid``.

        Fast (in-memory remap + small file write), so it runs on the caller's
        thread rather than a worker.
        """
        if self._state is None:
            self.exportFailed.emit("Nothing to export yet; run Convert first.")
            return
        try:
            profile = load_profile(self._plugin_id)
            events = remap_events(self._state.events, profile)
            write_midi(events, output_path, tempo=tempo)
        except Exception as exc:  # noqa: BLE001 - report any load/write failure
            self.exportFailed.emit(str(exc))
            return
        self.exportFinished.emit(output_path)

    # -- teardown ----------------------------------------------------------
    def cleanup(self) -> None:
        """Remove cached stems (call on application shutdown)."""
        self._discard_previous_stems()

    # -- internals ---------------------------------------------------------
    def _discard_previous_stems(self) -> None:
        if self._state is not None and self._state.stems_dir:
            shutil.rmtree(self._state.stems_dir, ignore_errors=True)

    def _start_worker(
        self,
        job: Callable[[Callable[[str, int], None]], object],
        kind: str,
    ) -> None:
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
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_thread_refs)

        self._thread = thread
        self._worker = worker
        self._worker_kind = kind
        thread.start()

    def _on_worker_finished(self, result: object) -> None:
        assert isinstance(result, AnalysisState)
        self._state = result
        if self._worker_kind == "analysis":
            self.analysisFinished.emit(result)
        else:
            self.eventsUpdated.emit(result.events)

    def _on_worker_failed(self, error: str) -> None:
        self.analysisFailed.emit(error)

    def _clear_thread_refs(self) -> None:
        self._thread = None
        self._worker = None
