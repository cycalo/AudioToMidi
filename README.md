# AudioToMidi — Drum Stem to MIDI Converter
.venv/Scripts/python.exe app/main.py
AudioToMidi is a Windows desktop application that converts a WAV drum stem into a standard MIDI file, remapped for popular virtual drum plugins. Upload a full-kit drum bus, review detected hits on an interactive waveform, tune sensitivity, and export a `.mid` ready to drag into your DAW.

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Run the GUI](#run-the-gui)
- [Repository Structure](#repository-structure)
- [Architecture Overview](#architecture-overview)
- [Supported Plugins](#supported-plugins)
- [CLI Tools](#cli-tools)
- [Environment Variables](#environment-variables)
- [Testing](#testing)
- [Build & Deployment](#build--deployment)
- [Contributing](#contributing)
- [License](#license)



## Features

- **Kick / snare / tom transcription** — Separates a drum stem (Demucs still produces full stems internally), then detects onsets for kick, snare, and toms only, with bleed hardening (relative peak floor + cross-stem dominance).
- **Tom pitch clustering** — Splits the toms stem into floor tom and rack tom using relative pitch clustering (k=2).
- **Velocity mapping** — Normalizes hit amplitudes to MIDI velocities with a configurable floor so ghost notes stay audible.
- **Plugin remapping** — Transcribes to General MIDI first, then applies a thin JSON-driven remap layer for seven target plugins.
- **Desktop GUI** — PySide6 window with staged progress, waveform review, color-coded onset markers, sensitivity slider, and GGD sample-based preview playback before export.
- **CLI pipeline** — Each processing stage is also available as a standalone script for scripting and debugging.
- **Dual separation backends** — Demucs drumsep (best quality, GPU-friendly) or a lightweight DSP fallback (no model download).
- **Session caching** — Separated stems are cached under `%TEMP%\audiotomidi_`* so sensitivity tweaks are fast; orphaned temp dirs from crashed sessions are cleaned on startup.



## Tech Stack


| Layer             | Technology                       |
| ----------------- | -------------------------------- |
| Language          | Python 3.11+ (3.13+ recommended) |
| GUI               | PySide6, pyqtgraph               |
| Audio I/O         | librosa, soundfile               |
| Source separation | demucs-infer, PyTorch            |
| Onset detection   | librosa                          |
| MIDI              | pretty_midi                      |
| Preview playback  | sounddevice                      |
| Numerics          | NumPy, SciPy                     |
| Packaging         | PyInstaller                      |
| Testing           | pytest                           |




## Quick Start



### Prerequisites

- **Python 3.11+** (3.13+ tested)
- **Windows** (macOS support planned)
- **~2 GB disk space** for Python dependencies and the one-time Demucs checkpoint (~167 MB)
- **Optional GPU** — CUDA speeds up separation significantly; CPU-only works but is slower



### Installation

1. **Clone the repository**
  ```bash
   git clone https://github.com/YOUR_USERNAME/AudioToMidi.git
   cd AudioToMidi
  ```
2. **Create a virtual environment and install dependencies**
  ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
  ```
3. **Run tests** (optional but recommended)
  ```bash
   pytest -q
  ```



### Run the GUI

```bash
python app/main.py
```

**Workflow:**

1. **Browse** to a drum-stem WAV, pick a target plugin, and choose a **Device** for separation: **Auto** (GPU if available), **CPU**, or **GPU** (shown only when CUDA is detected). Medium/low-confidence plugins show an inline hint (full detail lives in each profile's `notes` field under `mappings/`).
2. Click **Convert**. Separation and onset detection run on a background thread with a staged progress bar. When finished, the waveform shows color-coded onset markers for **kick / snare / toms**.
3. **Sensitivity** (optional) — The slider starts at tuned per-stem defaults. Drag and release to re-run detection on cached stems (left = fewer hits, right = more including quieter hits). Re-runs skip separation and typically finish in under a second.
4. **Voice filter** (preview) — Click a drum voice label above the waveform (**kick**, **snare**, **toms**) to filter preview playback. The first click isolates that voice; further clicks add voices. Click **All** to reset. Works with MIDI, Original (isolated stems), and Both sources. Onset markers on the waveform follow the same filter.
5. **Preview** (GetGood Drums only) — Select **GetGood Drums**, choose a **Source** mode (**MIDI** default, **Original**, or **Both**), then click **Play** to hear the remapped transcription through the bundled [Preview Kit](Preview%20Kit/) samples. Use **Both** to compare timing against the original stem. A playhead tracks playback on the waveform.
6. Click **Save MIDI...** to write the plugin-remapped `.mid`.

### GGD Preview Kit mapping

The [Preview Kit](Preview%20Kit/) uses post-remap GGD Modern & Massive GM note numbers (same as export when GGD is selected):

| GGD note | Voice | Sample file |
| -------- | ----- | ----------- |
| 36 | Kick (C1) | `kick.wav` |
| 38 | Snare (D1) | `snare.wav` |
| 43, 48 | Floor Tom 1 (G1 / C2) | `low tom.wav` |
| 50 | Rack Tom (D2) | `high tom.wav` |
| 49 | Crash L (C#2) | `crash cymbal.wav` |
| 54 | Hat Closed (F#2) | `hi hat closed.wav` |
| 58 | Hat Open0 (A#2) | `hi hat open.wav` |

**Detection scope:** Transcription emits kick / snare / toms only. Hi-hat and cymbal stems may still be written by separation for debugging, but they are not onset-detected. Bleed into snare/toms is reduced with a relative peak floor and cross-stem energy dominance gate.

**Timing expectations:**


| Step                                    | Typical duration (~30s clip)       |
| --------------------------------------- | ---------------------------------- |
| First Convert (includes model download) | 15–50s on CPU + one-time ~167 MB download; much faster on GPU |
| Subsequent Converts                     | ~15–50s on CPU; typically well under a minute on GPU |
| Sensitivity re-run                      | < 1s                               |
| Merge + remap                           | < 2s                               |




## Repository Structure

```
AudioToMidi/
├── app/
│   ├── main.py                  # PySide6 entry point; startup temp cleanup
│   ├── controller.py            # Wires UI to pipeline stages
│   └── ui/
│       ├── main_window.py       # Main window, controls, progress
│       └── waveform_view.py     # pyqtgraph onset review widget
├── pipeline/
│   ├── preprocess.py            # Resample, normalize, mono/stereo handling
│   ├── separation.py            # Demucs drumsep or DSP fallback
│   ├── onset_detection.py       # Per-stem onset + velocity extraction
│   ├── merge.py                 # Timeline merge, tom clustering, GM output
│   ├── transcription_v2.py      # v2 open-hat rerouting + open/closed classification
│   ├── remap.py                 # GM → plugin note remap
│   └── midi_writer.py           # pretty_midi file writer
├── mappings/                    # JSON plugin profiles (one per target)
│   ├── general_midi.json
│   ├── superior_drummer_3.json
│   ├── ezdrummer_3.json
│   ├── addictive_drums_2.json
│   ├── bfd3.json
│   ├── steven_slate_5_5.json
│   ├── ggd.json
│   └── drumforge.json
├── models/                      # Downloaded separation checkpoints (gitignored)
├── tests/
│   ├── fixtures/                # Short sample stems for regression tests
│   └── test_*.py
├── docs/
│   ├── PROJECT_SPEC.MD          # Original project specification
│   └── IMPLEMENTATION_PLAN.md   # Architecture and phase plan
├── requirements.txt
├── build.spec                   # PyInstaller one-folder spec
└── README.md
```



## Architecture Overview

The pipeline always produces a clean **General MIDI intermediate representation** first. Plugin selection only affects the final remap step, keeping transcription reusable and testable.

```
WAV drum stem
     │
     ▼
[1] Preprocess — resample, normalize
     │
     ▼
[2] Source separation (Demucs drumsep + DSP hi-hat, or full DSP)
     │   → kick, snare, toms (+ unused hihat/cymbals stems)
     ▼
[3] Per-stem onset detection on kick / snare / toms only
     │   → relative peak floor drops weak bleed ghosts
     ▼
[3a] Toms — pitch estimate + k=2 cluster → floor tom / rack tom
     │
     ▼
[4] Velocity extraction (amplitude → MIDI velocity curve)
     │
     ▼
[5] Cross-stem dominance gate + timeline merge
     │   → (time, GM note, velocity) events
     ▼
[6] GM → plugin remap (JSON profile)
     │
     ▼
[7] MIDI file writer → .mid
```

**Emitted GM notes:**


| Note | Drum voice                    |
| ---- | ----------------------------- |
| 36   | Kick                          |
| 38   | Snare                         |
| 45   | Floor tom                     |
| 47   | Tom fallback (single cluster) |
| 50   | Rack tom                      |




## Supported Plugins

Profiles live in `mappings/` as JSON files. Confidence reflects mapping documentation quality, not transcription accuracy.


| Plugin                 | Confidence | Notes                                                              |
| ---------------------- | ---------- | ------------------------------------------------------------------ |
| Superior Drummer 3     | High       | GM pass-through; built-in GM keymap preset                         |
| EZdrummer 3            | High       | GM pass-through                                                    |
| Addictive Drums 2      | High       | Load the GM map preset in AD2                                      |
| BFD3                   | High       | Select General MIDI keymap in BFD3                                 |
| Steven Slate Drums 5.5 | Medium     | Load Groove Monkee GM IOMap in SSD5.5                              |
| GetGood Drums          | Medium     | Select **GM** preset in plugin; in-app preview via Preview Kit      |
| Drumforge              | Low        | Proprietary factory map; toms/cymbals may need manual verification |


List all profiles from the CLI:

```bash
python pipeline/remap.py --list
```



## CLI Tools

Each pipeline stage can be run standalone. The full end-to-end path is: `separation.py` → `merge.py` (optionally with `--plugin`).

### Transcribe a single drum voice

```bash
python pipeline/onset_detection.py path/to/kick.wav -o kick.mid
```

Useful options: `-n` (GM note, default 36), `--delta` (onset threshold), `--window-ms` (velocity window), `--tempo`.

### Split a full-kit WAV into stems

```bash
# Demucs drumsep + hybrid hi-hat extract (default; 5 stems)
python pipeline/separation.py drums.wav -o drums_stems --backend demucs

# DSP fallback (fast, no download; 5 stems)
python pipeline/separation.py drums.wav -o drums_stems --backend dsp
```

Demucs runs ML separation for kick/snare/toms/cymbals, then extracts `hihat.wav` from the original mix. The full DSP backend uses frequency masking for all five stems.

### Transcribe stems to General MIDI

```bash
python pipeline/merge.py drums_stems -o drums.mid
```

Options: `--bleed-suppression`, `--velocity-floor`, `--delta-scale` (sensitivity, same knob as the GUI slider), `--relative-peak-floor`, `--dominance-ratio`, `--tempo`, `--plugin NAME`.
### Remap GM MIDI to a plugin

```bash
python pipeline/remap.py drums.mid --plugin superior_drummer_3 -o drums_sd3.mid

# Or transcribe and remap in one step
python pipeline/merge.py drums_stems --plugin ezdrummer_3 -o drums.mid
```



## Environment Variables


| Variable                         | Description                                                                 |
| -------------------------------- | --------------------------------------------------------------------------- |
| `AUDIOTOMIDI_SEPARATION_BACKEND` | Default separation backend: `demucs` or `dsp`                               |
| `AUDIOTOMIDI_DRUMSEP_URL`        | Override URL for the Demucs drumsep checkpoint download                     |
| `AUDIOTOMIDI_RUN_DEMUCS_TESTS`   | Set to `1` to run integration tests that require the cached checkpoint      |
| `AUDIOTOMIDI_SELFTEST`           | Set to `1` for a headless startup self-test (used in CI-style smoke checks) |


Never commit API keys or secrets. This project does not require external API keys.

## Testing

```bash
pytest -q
```

Tests cover separation, transcription, merge, remap, controller logic, and startup cleanup. Demucs integration tests are skipped unless the drumsep checkpoint is cached locally or `AUDIOTOMIDI_RUN_DEMUCS_TESTS=1` is set.

## Build & Deployment

Package a one-folder Windows executable with PyInstaller:

```bash
pyinstaller build.spec
dist\AudioToMidi\AudioToMidi.exe
```

The first run inside the packaged app still downloads the Demucs checkpoint on demand. Phase 6 (full PyInstaller packaging polish) is in progress.


| Component   | Platform | Notes                               |
| ----------- | -------- | ----------------------------------- |
| Desktop app | Windows  | PyInstaller one-folder build        |
| macOS       | Planned  | Same Python codebase; packaging TBD |




## Contributing

Contributions are welcome. To get started:

1. **Fork** the repository on GitHub.
2. **Clone** your fork and create a feature branch:
  ```bash
   git checkout -b feat/your-feature-name
  ```
3. **Set up** the environment (see [Quick Start](#quick-start)).
4. **Make changes** and add or update tests where behavior changes.
5. **Run the test suite**:
  ```bash
   pytest -q
  ```
6. **Commit** with clear messages (`feat:`, `fix:`, `docs:`, `refactor:`).
7. **Open a pull request** with a short description of what changed and why.

For larger changes, read [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) first to understand the phased architecture.

## License

TBD