# Drum Stem to MIDI Converter — Project Specification

## Goal
Build a Windows application that converts a WAV drum stem into MIDI, mapped to specific virtual drum plugin formats.

## Tech Stack
- **IDE**: Cursor AI (Agent Mode)
- **AI Assistant**: Claude Code
- **OS**: Windows (future update macOS)
- **Language/Framework**: [To be determined by research]

## User Flow
1. User uploads a WAV file of a drum kit stem
2. User selects target virtual drum instrument from a dropdown
3. User clicks "Convert" to generate MIDI file in the selected plugin's format

## Target Virtual Drum Instruments & MIDI Mapping

| Plugin | Developer |
|---|---|
| GGD (GetGood Drums) | GetGood / Misha Mansoor |
| Superior Drummer 3 | Toontrack |
| EZdrummer 3 | Toontrack |
| Addictive Drums 2 | XLN Audio |
| Steven Slate Drums 5.5 | Steven Slate Audio |
| BFD3 | FXpansion/inMusic |
| Drumforge | Drumforge |

## Technical Requirements
- Use each plugin's default/general MIDI mapping as the baseline
- Transient detection to identify drum hits from the audio stem
- Velocity mapping based on hit amplitude
- Handle overlapping hits (e.g., cymbal bleed, ghost notes)
- Output standard MIDI file (.mid) compatible with the selected plugin

## Research Needed
- Best audio-to-MIDI transient detection algorithms
- How to handle multi-drum stems (kick, snare, toms, cymbals in one file)
- General MIDI drum mapping vs. plugin-specific mappings
- Recommended libraries
- GUI framework