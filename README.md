# Drum Stem to MIDI

Convert a WAV drum stem into MIDI mapped for popular virtual drum plugins.

## Status

**Phase 5** — desktop GUI (PySide6). The full pipeline runs from a window: pick a
WAV, choose a target plugin (dropdown labels and inline hints come from each
profile's JSON in `mappings/`), click **Convert** to separate and transcribe in
the background, review detected onsets on a waveform, optionally tune sensitivity,
then **Save MIDI**. Builds on the Phase 4 remap layer, Phase 3 transcription
(per-stem onset detection, floor/rack tom clustering, merge), and Phase 2
separation. CLI tools from earlier phases still work standalone. Phase 6
(PyInstaller packaging) is not yet done.

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

The window drives the pipeline in two steps (**Convert**, then **Save MIDI**):

1. **Browse** to a drum-stem WAV and pick a target plugin. Medium/low-confidence
   plugins show a one-line inline hint (full detail lives in each profile's
   `notes` field under `mappings/`).
2. Click **Convert**. Separation and onset detection run on a background thread
   with a staged progress bar. When it finishes, the waveform shows the input
   audio with color-coded onset markers (kick / snare / toms / cymbals). Use
   **Reset View** or double-click the plot to fit the full clip; pan/zoom is
   clamped to the audio so you cannot scroll into empty space.
3. **Sensitivity** (optional): the slider starts centered on the tuned per-stem
   defaults — most material should not need adjustment. Drag and **release** to
   re-run detection on the cached stems (left = fewer/stronger hits only, right
   = more including quieter hits). Ghost notes are kept by default; per-stem
   presets already bias kick/snare toward sensitivity and cymbals/toms toward
   rejecting bleed.
4. Click **Save MIDI...** to write the plugin-remapped `.mid`.

**Timing:** separation dominates and is CPU-bound. A ~30s clip typically takes
roughly 15–50s on CPU without a GPU; onset detection, merge, and remap add well
under 2s. Sensitivity re-runs skip separation and take under a second. The first
Convert on a fresh install also downloads a one-time ~167 MB Demucs checkpoint.

**Temp files:** separated stems are cached under `%TEMP%\audiotomidi_*` for the
session (so the sensitivity slider can re-detect without re-separating). They are
removed when you quit normally. On startup, any leftover `audiotomidi_*` folders
from a crashed prior session are cleaned up automatically.

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

Velocities are normalized per file: the quietest hit maps to a floor of **20**
(not 1) and the loudest toward 127, so ghost notes stay audible in a DAW piano
roll while still tracking relative loudness.

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
one-time console warning per distinct note); hits are never dropped.

Each profile JSON may also include `ui_label_suffix` and `ui_hint` for the GUI
dropdown label and one-line warning; the full `notes` field holds detailed
mapping rationale.

Confidence tiers reflect how well each plugin's mapping is documented and
verified, not transcription quality:

- **high** — Superior Drummer 3, EZdrummer 3, Addictive Drums 2, BFD3: documented
  built-in GM keymap presets; profiles are pass-through for all six emitted notes.
- **medium** — Steven Slate Drums 5.5 (not native GM; load the Groove Monkee GM
  IOMap in its Map tab) and GetGood Drums (verified against Modern & Massive's
  built-in **GM** preset only — select **GM** in the plugin, not Halpern or the
  default GGD preset; kick/snare/crash/rack tom pass through, but our floor tom
  and tom fallback `45`/`47` remap to GGD note **43** / Floor Tom 1).
- **low** — Drumforge (proprietary factory map; kick/snare pass through but
  toms/cymbals may land on the wrong piece without a GM-compatible preset in
  Drumforge's Map page).

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
