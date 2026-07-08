"""Timeline merge and overlap/bleed handling.

Phase 3: merge per-stem onset events into one sorted General MIDI timeline, with:

- a per-voice minimum inter-onset interval to suppress double-triggers from
  decay/ringing (on by default),
- an optional ghost-note velocity floor (off by default -> keep ghost notes),
- an optional same-timestamp cross-stem bleed suppression heuristic (off by
  default until validated).

Also provides ``transcribe_stems`` to turn a Phase 2 separated-stems directory
into a single GM ``.mid``, and a CLI wrapper.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.midi_writer import DEFAULT_TEMPO, write_midi  # noqa: E402
from pipeline.onset_detection import (  # noqa: E402
    STEM_PRESETS,
    Event,
    detect_stem_events,
    min_ioi_by_note,
)
from pipeline.remap import load_profile, remap_events  # noqa: E402

# Deterministic order in which stems are processed / merged.
STEM_ORDER = ("kick", "snare", "toms", "hihat", "cymbals")

DEFAULT_MIN_IOI_MS = 30.0
DEFAULT_BLEED_WINDOW_MS = 10.0
DEFAULT_BLEED_RATIO = 0.35
DEFAULT_HIHAT_CYMBAL_WINDOW_MS = 20.0
CYMBAL_HIHAT_ATTENUATION = 0.85

GM_NOTE_HIHAT = 42
GM_NOTE_CYMBALS = 49


def _is_hihat_event(note: int, stem: str) -> bool:
    return stem == "hihat" or note == GM_NOTE_HIHAT


def _is_cymbal_event(note: int, stem: str) -> bool:
    return stem == "cymbals" or note == GM_NOTE_CYMBALS


def resolve_cymbal_hihat_collisions(
    tagged: List[Tuple[float, int, int, str]],
    *,
    window_ms: float = DEFAULT_HIHAT_CYMBAL_WINDOW_MS,
) -> List[Tuple[float, int, int, str]]:
    """Drop cymbal/crash events coincident with hi-hat hits (hat wins)."""
    window_s = window_ms / 1000.0
    hat_times = [
        time_s
        for time_s, note, _velocity, stem in tagged
        if _is_hihat_event(note, stem)
    ]
    if not hat_times:
        return tagged

    drop: set[int] = set()
    for index, (time_s, note, _velocity, stem) in enumerate(tagged):
        if not _is_cymbal_event(note, stem):
            continue
        for hat_t in hat_times:
            if abs(time_s - hat_t) <= window_s:
                drop.add(index)
                break
    return [event for index, event in enumerate(tagged) if index not in drop]


def _pad_to_length(a: np.ndarray, length: int) -> np.ndarray:
    if a.size >= length:
        return a[:length]
    out = np.zeros(length, dtype=np.float32)
    out[: a.size] = a
    return out


def _cymbals_detect_path(
    cymbals_path: Path,
    hihat_path: Path,
) -> tuple[str, Optional[str]]:
    """Build a temp cymbals WAV with hi-hat energy subtracted for onset detection."""
    cymbals, sr = librosa.load(str(cymbals_path), sr=None, mono=True)
    hihat, _ = librosa.load(str(hihat_path), sr=sr, mono=True)
    length = max(cymbals.size, hihat.size)
    cymbals = _pad_to_length(cymbals, length)
    hihat = _pad_to_length(hihat, length)
    cleaned = cymbals - hihat * CYMBAL_HIHAT_ATTENUATION
    peak = float(np.max(np.abs(cleaned)))
    if peak > 1.0:
        cleaned = cleaned / peak

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, cleaned.astype(np.float32), sr)
    tmp.close()
    return tmp.name, tmp.name


def merge_events(
    stem_events: Dict[str, List[Event]],
    *,
    apply_min_ioi: bool = True,
    min_ioi_ms: Optional[Dict[int, float]] = None,
    velocity_floor: int = 0,
    bleed_suppression: bool = False,
    bleed_window_ms: float = DEFAULT_BLEED_WINDOW_MS,
    bleed_ratio: float = DEFAULT_BLEED_RATIO,
) -> List[Event]:
    """Merge per-stem events into one sorted, cleaned ``(time, note, velocity)`` list.

    Args:
        stem_events: Mapping of stem name -> list of events for that stem.
        apply_min_ioi: Enforce a per-voice minimum inter-onset interval (default on).
        min_ioi_ms: Per-note min inter-onset in ms; defaults to the per-stem presets.
        velocity_floor: Drop events with velocity below this (default 0 = keep all).
        bleed_suppression: Enable cross-stem bleed suppression (default OFF).
        bleed_window_ms: Coincidence window for bleed suppression.
        bleed_ratio: Drop the quieter coincident event if its velocity is below
            this fraction of the louder cross-stem event's velocity.
    """
    if min_ioi_ms is None:
        min_ioi_ms = min_ioi_by_note()

    # Flatten with source-stem tags, sorted by time (note as tiebreaker for determinism).
    tagged: List[Tuple[float, int, int, str]] = []
    for stem, events in stem_events.items():
        for time_s, note, velocity in events:
            tagged.append((float(time_s), int(note), int(velocity), stem))
    tagged.sort(key=lambda e: (e[0], e[1]))

    # 1) Per-voice minimum inter-onset interval (keep the earlier of a close pair).
    if apply_min_ioi:
        last_kept: Dict[int, float] = {}
        deduped: List[Tuple[float, int, int, str]] = []
        for time_s, note, velocity, stem in tagged:
            gap_s = min_ioi_ms.get(note, DEFAULT_MIN_IOI_MS) / 1000.0
            prev = last_kept.get(note)
            if prev is not None and (time_s - prev) < gap_s:
                continue
            last_kept[note] = time_s
            deduped.append((time_s, note, velocity, stem))
        tagged = deduped

    # 2) Hi-hat vs cymbal collision: prefer hat (42) over crash (49).
    tagged = resolve_cymbal_hihat_collisions(tagged)

    # 3) Ghost-note velocity floor (off by default).
    if velocity_floor > 0:
        tagged = [e for e in tagged if e[2] >= velocity_floor]

    # 4) Same-timestamp cross-stem bleed suppression (off by default).
    if bleed_suppression:
        window_s = bleed_window_ms / 1000.0
        drop = set()
        for i, (ti, _ni, vi, si) in enumerate(tagged):
            for j, (tj, _nj, vj, sj) in enumerate(tagged):
                if i == j or si == sj:
                    continue
                if abs(ti - tj) <= window_s and vi < vj and vi < bleed_ratio * vj:
                    drop.add(i)
                    break
        tagged = [e for k, e in enumerate(tagged) if k not in drop]

    return [(t, n, v) for (t, n, v, _stem) in tagged]


def _transcribe_stems_v1(
    stems_dir: str,
    *,
    bleed_suppression: bool = False,
    velocity_floor: int = 0,
    delta_scale: float = 1.0,
) -> Tuple[List[Event], dict]:
    """v1 transcription: original merge path (closed hat only, crash dedupe)."""
    stems_path = Path(stems_dir)
    manifest_path = stems_path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        stem_files = {
            name: stems_path / filename
            for name, filename in manifest.get("stems", {}).items()
        }
    else:
        stem_files = {
            name: stems_path / f"{name}.wav"
            for name in STEM_PRESETS
            if (stems_path / f"{name}.wav").exists()
        }

    stem_events: Dict[str, List[Event]] = {}
    summary: dict = {"stems": {}, "tom_k": None}
    temp_detect_paths: List[str] = []
    cymbals_path = stem_files.get("cymbals")
    hihat_path = stem_files.get("hihat")

    for name in STEM_ORDER:
        path = stem_files.get(name)
        if path is None or not path.exists():
            continue
        detect_path = str(path)
        if (
            name == "cymbals"
            and cymbals_path is not None
            and hihat_path is not None
            and hihat_path.exists()
        ):
            detect_path, temp_path = _cymbals_detect_path(cymbals_path, hihat_path)
            if temp_path is not None:
                temp_detect_paths.append(temp_path)
        events, info = detect_stem_events(detect_path, name, delta_scale=delta_scale)
        stem_events[name] = events
        summary["stems"][name] = info
        if name == "toms":
            summary["tom_k"] = info.get("tom_k")

    for temp_path in temp_detect_paths:
        Path(temp_path).unlink(missing_ok=True)

    merged = merge_events(
        stem_events,
        bleed_suppression=bleed_suppression,
        velocity_floor=velocity_floor,
    )

    note_counts: Dict[int, int] = {}
    for _t, note, _v in merged:
        note_counts[note] = note_counts.get(note, 0) + 1
    summary["total_events"] = len(merged)
    summary["note_counts"] = note_counts
    summary["transcription_version"] = "v1"
    return merged, summary


def transcribe_stems(
    stems_dir: str,
    *,
    bleed_suppression: bool = False,
    velocity_floor: int = 0,
    delta_scale: float = 1.0,
    transcription_version: str = "v2",
) -> Tuple[List[Event], dict]:
    """Transcribe a Phase 2 separated-stems directory into merged GM events.

    ``transcription_version`` selects v1 (classic) or v2 (open-hat rerouting).
    """
    if transcription_version == "v1":
        return _transcribe_stems_v1(
            stems_dir,
            bleed_suppression=bleed_suppression,
            velocity_floor=velocity_floor,
            delta_scale=delta_scale,
        )
    from pipeline.transcription_v2 import transcribe_stems_v2  # noqa: WPS433

    return transcribe_stems_v2(
        stems_dir,
        bleed_suppression=bleed_suppression,
        velocity_floor=velocity_floor,
        delta_scale=delta_scale,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe a separated-stems directory (Phase 2 output) into a single "
            "General MIDI .mid (Phase 3: multi-stem onset detection + merge)."
        )
    )
    parser.add_argument("stems_dir", type=Path, help="Directory of separated stem WAVs.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .mid path (default: <stems_dir>.mid).",
    )
    parser.add_argument(
        "--bleed-suppression",
        action="store_true",
        help="Enable cross-stem same-timestamp bleed suppression (default off).",
    )
    parser.add_argument(
        "--velocity-floor",
        type=int,
        default=0,
        help="Drop events below this velocity (default 0 = keep ghost notes).",
    )
    parser.add_argument(
        "--tempo",
        type=float,
        default=DEFAULT_TEMPO,
        help=f"Initial tempo written to the MIDI file (default: {DEFAULT_TEMPO}).",
    )
    parser.add_argument(
        "--delta-scale",
        type=float,
        default=1.0,
        help=(
            "Onset sensitivity multiplier over the tuned per-stem thresholds "
            "(default 1.0; <1 detects more hits, >1 fewer; clamped 0.25-2.0)."
        ),
    )
    parser.add_argument(
        "--plugin",
        default=None,
        help=(
            "Optional plugin profile (file stem or display name from mappings/) to "
            "remap GM notes to the plugin's note numbers. Default: pure General MIDI."
        ),
    )
    parser.add_argument(
        "--transcription-version",
        choices=("v1", "v2"),
        default="v2",
        help=(
            "Transcription algorithm: v1 (classic closed-hat + crash) or "
            "v2 (open-hat rerouting and open/closed classification). Default: v2."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if not args.stems_dir.is_dir():
        print(f"error: stems directory not found: {args.stems_dir}", file=sys.stderr)
        return 2

    events, summary = transcribe_stems(
        str(args.stems_dir),
        bleed_suppression=args.bleed_suppression,
        velocity_floor=args.velocity_floor,
        delta_scale=args.delta_scale,
        transcription_version=args.transcription_version,
    )

    if args.plugin:
        try:
            profile = load_profile(args.plugin)
        except (ValueError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        events = remap_events(events, profile)
        print(f"Remapped to plugin: {profile['plugin']} (confidence: {profile['confidence']})")

    output = args.output or args.stems_dir.with_suffix(".mid")
    write_midi(events, output, tempo=args.tempo)

    print(f"Transcribed '{args.stems_dir.name}' -> {output}")
    print(f"  total events: {summary['total_events']}  (tom clusters k={summary['tom_k']})")
    for name, info in summary["stems"].items():
        extra = f", tom_k={info['tom_k']}" if info.get("tom_k") is not None else ""
        print(f"  {name}: {info['onsets']} onsets{extra}")
    if summary["note_counts"]:
        counts = ", ".join(
            f"{note}:{count}" for note, count in sorted(summary["note_counts"].items())
        )
        print(f"  note counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
