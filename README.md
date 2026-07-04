# Drum Stem to MIDI

Convert a WAV drum stem into MIDI mapped for popular virtual drum plugins.

## Status

**Phase 3** — full transcription pipeline (CLI). Runs per-stem onset detection
with per-voice tuned parameters across the separated stems, adaptively clusters
tom hits into low/mid/high notes, merges everything into one General MIDI
timeline (with double-trigger suppression and optional bleed handling), and
writes a `.mid`. Builds on Phase 2 separation and the Phase 1 transcriber. The
PySide6 window from Phase 0 still launches but is not yet wired to the pipeline.

## Requirements

- Python 3.13+ (3.11+ supported)
- Windows (macOS planned for a later release)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Run from source

```bash
python app/main.py
```

## Transcribe a drum voice to MIDI (CLI)

Phase 1 provides a command-line transcriber for a single pre-isolated drum voice.
Feed it a solo drum WAV (e.g. an isolated kick) and it writes a GM `.mid`:

```bash
python pipeline/onset_detection.py path/to/kick.wav -o kick.mid
```

Useful options:

- `-n, --note` GM note assigned to every hit (default `36`, kick)
- `--delta` onset threshold; raise it to suppress spurious re-triggers
- `--window-ms` velocity measurement window after each onset
- `--tempo` initial tempo written into the file

Velocities are normalized per file: the quietest hit maps toward the velocity
floor and the loudest toward 127, so velocity visibly tracks loudness in a DAW
piano roll.

## Split a full-kit drum WAV into stems (CLI)

Phase 2 separates a full drum-bus WAV into per-voice stems, written alongside a
`manifest.json` describing the run.

```bash
# Demucs drumsep backend (best quality; 4 stems: kick, snare, toms, cymbals)
python pipeline/separation.py drums.wav -o drums_stems --backend demucs

# DSP fallback (fast, no download, CPU-only; 5 stems: adds hihat)
python pipeline/separation.py drums.wav -o drums_stems --backend dsp
```

Notes and current limitations:

- The `demucs` backend downloads a one-time ~167 MB checkpoint on first use into
  `models/drumsep/` (SHA-256 verified; source overridable via
  `AUDIOTOMIDI_DRUMSEP_URL`).
- The drumsep model does **not** separate hi-hat — it stays inside the cymbals
  stem. A dedicated hi-hat stem is only produced by the `dsp` backend (a v1
  limitation).
- Device is auto-detected (`--device auto|cpu|cuda`). CPU separation of a 3-4
  minute stem realistically takes minutes; a GPU cuts this to well under a
  minute.
- Backend can also be set via `AUDIOTOMIDI_SEPARATION_BACKEND=demucs|dsp`.

## Transcribe separated stems to a single MIDI (CLI)

Phase 3 turns a separated-stems directory (Phase 2 output, with its
`manifest.json`) into one General MIDI file:

```bash
python pipeline/merge.py drums_stems -o drums.mid
```

What it does:

- Detects onsets per stem with per-voice tuned settings (kick/snare use a low
  threshold; cymbals use a higher threshold and longer wait to avoid retriggering
  on the decay wash).
- Splits the toms stem by pitch, adaptively choosing 2 or 3 tom notes per file.
- Merges into one timeline with a per-voice minimum inter-onset interval.

Options:

- `--bleed-suppression` drop a quiet onset coincident with a much louder hit in
  another stem (off by default).
- `--velocity-floor N` drop hits below velocity `N` (default 0 = keep ghost notes).
- `--tempo T` initial tempo written into the file.

The full end-to-end path is: `separation.py` (WAV -> stems) then `merge.py`
(stems -> GM `.mid`).

## Run tests

```bash
pytest -q
```

## Build (PyInstaller, one-folder)

```bash
pyinstaller build.spec
dist\AudioToMidi\AudioToMidi.exe
```

## Project layout

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) Section 4 for the full repository structure.

## License

TBD
