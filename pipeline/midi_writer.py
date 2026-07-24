"""MIDI file writer using pretty_midi.

Phase 1: turn a list of ``(time, note, velocity)`` events into a drum ``.mid``.

Events are written to a single drum instrument (General MIDI channel 10, i.e.
``is_drum=True``) so the output is directly playable by any GM-aware drum plugin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Union

import pretty_midi

Event = Tuple[float, int, int]
PathLike = Union[str, Path]

DEFAULT_TEMPO = 120.0
DEFAULT_NOTE_DURATION = 0.1  # seconds; drums are one-shots, exact length is cosmetic


def write_midi(
    events: Iterable[Event],
    output_path: PathLike,
    *,
    tempo: float = DEFAULT_TEMPO,
    note_duration: float = DEFAULT_NOTE_DURATION,
    is_drum: bool = True,
    program: int = 0,
    instrument_name: str = "Drums",
) -> Path:
    """Write ``(time, note, velocity)`` events to a MIDI file.

    Args:
        events: Iterable of ``(time_seconds, midi_note, velocity)`` tuples.
        output_path: Destination ``.mid`` path (parent dirs are created).
        tempo: Initial tempo written to the file. Event times are absolute
            seconds in memory; this BPM controls tick encoding. Many DAWs
            schedule those ticks at the *project* tempo, so this value must
            match the track BPM or playback speed will be wrong.
        note_duration: Fixed note length in seconds for each one-shot hit.
        is_drum: Place notes on the GM drum channel when True.
        program: GM program number (ignored by most hosts when ``is_drum``).
        instrument_name: Track name written into the file.

    Returns:
        The ``Path`` that was written.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    instrument = pretty_midi.Instrument(
        program=int(program), is_drum=is_drum, name=instrument_name
    )

    for time_s, note, velocity in events:
        start = float(time_s)
        end = start + float(note_duration)
        clamped_velocity = int(max(1, min(127, int(velocity))))
        instrument.notes.append(
            pretty_midi.Note(
                velocity=clamped_velocity,
                pitch=int(note),
                start=start,
                end=end,
            )
        )

    pm.instruments.append(instrument)

    out = Path(output_path)
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out))
    return out
