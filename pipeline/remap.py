"""General MIDI to plugin-specific note remapping.

Phase 4: load a plugin's JSON profile from ``mappings/`` and remap the General
MIDI note numbers this pipeline emits (36, 38, 45, 47, 49, 50) to the note
numbers a specific drum plugin expects.

Design:
- Transcription stays plugin-agnostic and always emits clean GM (see
  ``pipeline.merge``). This module is a thin, data-driven layer applied
  afterward, so the whole pipeline left of it is reusable for any target.
- A GM note with no entry in a profile's ``map`` passes through unchanged
  (identity), with a one-time warning per distinct note. Profiles are never
  required to be complete, and hits are never dropped.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import pretty_midi

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.midi_writer import Event  # noqa: E402

logger = logging.getLogger(__name__)

MAPPINGS_DIR = REPO_ROOT / "mappings"
REQUIRED_KEYS = ("plugin", "confidence", "map")


def available_profiles() -> List[str]:
    """Return the sorted profile names (JSON file stems) available in mappings/."""
    if not MAPPINGS_DIR.is_dir():
        return []
    return sorted(p.stem for p in MAPPINGS_DIR.glob("*.json"))


def _resolve_profile_path(plugin: str) -> Path:
    """Resolve a profile identifier to a JSON path.

    Accepts a file stem (e.g. ``superior_drummer_3``) or a plugin display name
    (e.g. ``Superior Drummer 3``, matched case-insensitively against each
    profile's ``plugin`` field).
    """
    candidate = MAPPINGS_DIR / f"{plugin}.json"
    if candidate.is_file():
        return candidate

    target = plugin.strip().lower()
    for path in sorted(MAPPINGS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(data.get("plugin", "")).strip().lower() == target:
            return path

    raise FileNotFoundError(
        f"No mapping profile for '{plugin}'. Available: {', '.join(available_profiles())}"
    )


def load_profile(plugin: str) -> dict:
    """Load and validate a plugin mapping profile from ``mappings/``.

    Returns the parsed profile with ``map`` normalized to ``{int: int}``.

    Raises:
        FileNotFoundError: if no profile matches ``plugin``.
        ValueError: if the matched file is malformed.
    """
    path = _resolve_profile_path(plugin)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in mapping profile {path.name}: {exc}") from exc

    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(
            f"Mapping profile {path.name} is missing required key(s): {', '.join(missing)}"
        )

    raw_map = data["map"]
    if not isinstance(raw_map, dict):
        raise ValueError(f"Mapping profile {path.name} 'map' must be a JSON object.")

    note_map: Dict[int, int] = {}
    for src, dst in raw_map.items():
        try:
            note_map[int(src)] = int(dst)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Mapping profile {path.name} has a non-integer note entry: "
                f"{src!r} -> {dst!r}"
            ) from exc

    data["map"] = note_map
    return data


def remap_note(
    note: int, note_map: Dict[int, int], *, warned: Optional[Set[int]] = None
) -> int:
    """Return the remapped note, or the original note (identity) if unmapped.

    When ``warned`` is provided, log one warning per distinct unmapped note.
    """
    if note in note_map:
        return note_map[note]
    if warned is not None and note not in warned:
        logger.warning(
            "GM note %d has no entry in mapping profile; passing through unchanged.",
            note,
        )
        warned.add(note)
    return note


def remap_events(events: Sequence[Event], profile: dict) -> List[Event]:
    """Remap the note number of each ``(time, note, velocity)`` event.

    Unmapped notes pass through unchanged (identity) with a one-time warning per
    distinct note. Timing and velocity are preserved.
    """
    note_map = profile.get("map", {})
    warned: Set[int] = set()
    return [
        (time_s, remap_note(int(note), note_map, warned=warned), velocity)
        for time_s, note, velocity in events
    ]


def remap_midi_file(in_mid, out_mid, plugin: str) -> Path:
    """Load a GM ``.mid``, remap its drum note pitches per ``plugin``, and write it.

    Timing, velocity, tempo and note durations are preserved. Drum instruments
    (``is_drum``) are remapped; if the file has none, all instruments are remapped
    (our own output is always a single drum track).
    """
    profile = load_profile(plugin)
    note_map = profile["map"]
    pm = pretty_midi.PrettyMIDI(str(in_mid))

    drum_instruments = [inst for inst in pm.instruments if inst.is_drum]
    targets = drum_instruments if drum_instruments else pm.instruments
    if not drum_instruments:
        logger.warning(
            "No drum (is_drum) instrument found in %s; remapping all instruments.",
            Path(in_mid).name,
        )

    warned: Set[int] = set()
    for instrument in targets:
        for midi_note in instrument.notes:
            midi_note.pitch = remap_note(int(midi_note.pitch), note_map, warned=warned)

    out = Path(out_mid)
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out))
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remap a General MIDI drum .mid to a plugin's note numbers using a "
            "JSON profile from mappings/ (Phase 4)."
        )
    )
    parser.add_argument(
        "input", nargs="?", type=Path, help="Input General MIDI .mid file."
    )
    parser.add_argument(
        "-p",
        "--plugin",
        help="Plugin profile: file stem (e.g. superior_drummer_3) or display name.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .mid path (default: <input>.<profile>.mid).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available plugin profiles (with confidence) and exit.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_arg_parser().parse_args(argv)

    if args.list:
        for name in available_profiles():
            try:
                profile = load_profile(name)
                print(f"{name:22s} [{profile.get('confidence', '?'):6s}] {profile.get('plugin', name)}")
            except (ValueError, FileNotFoundError) as exc:
                print(f"{name:22s} [error] {exc}")
        return 0

    if args.input is None or args.plugin is None:
        print(
            "error: an input .mid and --plugin are required (or use --list).",
            file=sys.stderr,
        )
        return 2
    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    try:
        profile = load_profile(args.plugin)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    profile_stem = _resolve_profile_path(args.plugin).stem
    output = args.output or args.input.with_name(f"{args.input.stem}.{profile_stem}.mid")
    remap_midi_file(args.input, output, args.plugin)

    print(
        f"Remapped '{args.input.name}' -> {output}  "
        f"(profile: {profile['plugin']}, confidence: {profile['confidence']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
