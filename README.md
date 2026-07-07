# Drum Stem to MIDI

Convert a WAV drum stem into MIDI mapped for popular virtual drum plugins.

## Status

**Phase 5** — desktop GUI (PySide6). The full pipeline now runs from a window:
pick a WAV, choose a target plugin (annotated by mapping confidence), click
Convert to separate and transcribe in the background, review the detected onsets
on a waveform, tune detection sensitivity with a slider, then Save the
plugin-mapped MIDI. Builds on the Phase 4 remap layer, the Phase 3 full pipeline
(per-stem onset detection, floor/rack tom clustering, merge), and Phase 2
separation. The CLI tools from earlier phases still work standalone.

## Requirements

- Python 3.13+ (3.11+ supported)
- Windows (macOS planned for a later release)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Run the app (GUI)

```bash
python app/main.py
```

The window drives the whole pipeline in two steps:

1. Browse to a drum-stem WAV and pick a target plugin from the dropdown. Plugins
   with medium/low mapping confidence show an inline note (see the remap section
   for what the tiers mean).
2. Click **Convert**. Separation and onset detection run on a background thread
   with a staged progress bar, so the window stays responsive. When it finishes,
   the waveform shows color-coded onset markers (kick/snare/toms/cymbals).
3. Drag the **sensitivity** slider to re-run detection live (fast — it reuses the
   already-separated stems). Left detects only the strongest hits; right detects
   more, quieter hits.
4. Click **Save MIDI...** to write the plugin-remapped `.mid`.

Expected wait for Convert: separation dominates and is CPU-bound. A ~30s clip
typically takes roughly 15-50s on a CPU without a GPU (onset detection, merge,
and remap together add well under 2s). The first ever run also downloads a
one-time ~167 MB separation model. Live sensitivity re-runs take under a second
since they skip separation.

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
- Splits the toms stem by pitch into floor tom (GM 45) and rack tom (GM 50).
- Merges into one timeline with a per-voice minimum inter-onset interval.

Options:

- `--bleed-suppression` drop a quiet onset coincident with a much louder hit in
  another stem (off by default).
- `--velocity-floor N` drop hits below velocity `N` (default 0 = keep ghost notes).
- `--delta-scale S` onset sensitivity multiplier over the tuned thresholds
  (default 1.0; below 1 detects more hits, above 1 fewer; clamped 0.25-2.0). This
  is the same knob the GUI sensitivity slider drives.
- `--tempo T` initial tempo written into the file.
- `--plugin NAME` remap the output to a plugin profile in one step (see below).
  Omit it for pure General MIDI.

The full end-to-end path is: `separation.py` (WAV -> stems) then `merge.py`
(stems -> GM `.mid`).

## Remap General MIDI to a plugin's note numbers (CLI)

Phase 4 keeps transcription plugin-agnostic (always GM) and applies a separate
remap layer driven by JSON profiles in `mappings/`. Remap an existing GM `.mid`,
or use `merge.py --plugin` to do it in one pass.

```bash
# List available plugin profiles and their confidence tiers
python pipeline/remap.py --list

# Remap a GM .mid to a plugin (accepts a file stem or the display name)
python pipeline/remap.py drums.mid --plugin superior_drummer_3 -o drums_sd3.mid

# ...or transcribe and remap in one step
python pipeline/merge.py drums_stems --plugin ezdrummer_3 -o drums.mid
```

Only the notes this pipeline emits need coverage: `36` (kick), `38` (snare),
`45` (floor tom), `50` (rack tom), `49` (crash), and `47` (single-cluster tom
fallback). A GM note with no entry in a profile passes through unchanged (with a
warning); hits are never dropped.

Confidence tiers reflect how well each plugin's GM compatibility is documented,
not output quality:

- **high** — Superior Drummer 3, EZdrummer 3, Addictive Drums 2, BFD3: documented
  built-in GM keymap presets, so the profiles are pass-through.
- **medium** — Steven Slate Drums 5.5: not native GM; load the community Groove
  Monkee GM IOMap in its Map tab so the pass-through profile lines up.
- **low** — GetGood Drums (ships a "GM Mapping" preset but assignments vary per
  library title — verify in the Mapping tab) and Drumforge (proprietary factory
  map where GM 49/50 may hit the wrong piece — load a GM-compatible preset in its
  Map page or remap toms/cymbals manually).

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
