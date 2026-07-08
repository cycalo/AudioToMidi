"""Drum voice names and GM-note mapping shared by UI and preview."""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Sequence, Tuple

Event = Tuple[float, int, int]

VOICE_ORDER = ("kick", "snare", "toms", "cymbals", "hihat")
ALL_VOICES: FrozenSet[str] = frozenset(VOICE_ORDER)

# General MIDI notes emitted by the transcription pipeline.
GM_NOTE_TO_VOICE: Dict[int, str] = {
    36: "kick",
    38: "snare",
    45: "toms",
    47: "toms",
    50: "toms",
    49: "cymbals",
    42: "hihat",
    46: "hihat",
}

STEM_NAME_BY_VOICE: Dict[str, str] = {
    "kick": "kick",
    "snare": "snare",
    "toms": "toms",
    "cymbals": "cymbals",
    "hihat": "hihat",
}


def is_all_voices(voices: FrozenSet[str]) -> bool:
    return voices == ALL_VOICES


def toggle_voice_filter(current: FrozenSet[str], voice: str) -> FrozenSet[str]:
    """Toggle one voice in the preview filter (additive from ALL, exclusive first pick)."""
    if voice not in ALL_VOICES:
        return current
    if is_all_voices(current):
        return frozenset({voice})
    if voice in current:
        remaining = frozenset(v for v in current if v != voice)
        return ALL_VOICES if not remaining else remaining
    return frozenset(set(current) | {voice})


def filter_gm_events_by_voices(
    events: Sequence[Event],
    voices: FrozenSet[str],
) -> List[Event]:
    """Keep only events whose GM note maps to an active voice."""
    if is_all_voices(voices):
        return list(events)
    return [
        event
        for event in events
        if GM_NOTE_TO_VOICE.get(int(event[1]), "cymbals") in voices
    ]


def voices_label(voices: FrozenSet[str]) -> str:
    if is_all_voices(voices):
        return "all"
    return "+".join(v for v in VOICE_ORDER if v in voices)
