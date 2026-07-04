# Drum Stem → MIDI Converter — Implementation Plan for Cursor AI Agent Mode

## 1. Research Summary

### 1.1 Transient detection: what actually works
Full-kit "automatic drum transcription" (classifying every hit in a summed mix directly) is still an open research problem, especially for toms and cymbals where frequency overlap and bleed are worst. The practical workaround the industry has converged on is to **separate the stem into individual drum voices first, then run simple per-voice onset detection**. Single-instrument onset detection (just "did the kick hit or not") is a solved problem; multi-class classification from a mixed signal is not.

For per-voice onset detection, the strongest open-source options are:
- **madmom** — CNN-based onset detector trained on ~26,000 annotated onsets, plus spectral-flux based onset processors. Best accuracy for percussive material.
- **librosa** — `onset_detect` with spectral flux / high-frequency-content backends. Easier to install, good enough once the signal is a single isolated drum voice.
- **aubio** — lightweight, real-time capable, good fallback if madmom's dependency chain is a problem on Windows.
- **ADTLib** — a purpose-built automatic drum transcription library (CNN-based) that outputs kick/snare/hi-hat onsets directly, useful as a reference implementation or fallback path.

### 1.2 Handling a multi-drum stem in one file
A single WAV containing the whole kit needs to be split into per-instrument stems before transcription. This is exactly what the "drumsep" family of tools does:
- **inagoy/drumsep** — a Hybrid Demucs model fine-tuned specifically on drum recordings, splitting a drum bus into kick, snare, cymbals, and toms.
- Community-trained variants hosted on MVSep (MDX23C, SCNet XL, MelBand RoFormer architectures) extend this to 5–6 stems by separating hi-hat from cymbals, and crash from ride.
- **cukas/drumsep** — a pure-DSP alternative (harmonic-percussive separation + frequency masking + transient gating) that needs no ML model or GPU, at the cost of separation quality, useful as a lightweight fallback mode.
- Across all of these, kick separates cleanest, snare is moderate, and toms/cymbals are the weakest link because of their broad/overlapping frequency content. Plan the UI and QA process around this reality (see Section 7).

### 1.2.1 Splitting the toms stem by pitch
Source separation gives you one "toms" stem, not one stem per tom, so a second, cheaper step is needed on top of it: for each detected onset in the toms stem, estimate the pitch of that hit (a short pitch/fundamental-frequency estimate right after the transient, e.g. via `librosa.pyin` or autocorrelation, skipping the initial broadband click) and cluster the results into low/mid/high. Use **relative clustering** over the pitches found in that specific file rather than fixed frequency thresholds, since tom tuning varies enormously between kits. Rather than forcing a fixed `k=3`, **decide the number of clusters adaptively per stem**: run k-means at both `k=2` and `k=3` on the per-onset pitch estimates, compare cluster separation (e.g. `silhouette_score`, or the relative gap between cluster centers), and pick whichever better fits that stem's actual pitch spread (preferring `k=2` unless `k=3` wins by a clear margin, and collapsing to a single group when there is only one usable pitch or negligible spread). This keeps the split working whether the kit has 2 toms or 5 — a 2-tom kit gets 2 clusters instead of a spurious third — at the cost of not guaranteeing "low tom" always means the same absolute pitch across different songs. Flag this as a v1 simplification (see Section 7).

The same clustering idea does **not** get applied to cymbals: cymbal decay/timbre differences (crash vs ride vs splash) aren't a clean pitch axis the way toms are, which is why cymbals stay lumped into one bucket for v1 while toms get split three ways.

### 1.3 General MIDI vs plugin-specific mapping
This turned out to be the single most important finding for the architecture. Every major plugin researched treats **General MIDI as a common interchange format**, not just Superior Drummer 3/EZdrummer 3:
- **Superior Drummer 3 / EZdrummer 3**: internal "TDM" mapping, but SD3 ships built-in mapping presets including one for Addictive Drums 2, and supports "Add Suggested Note" remapping. GM-based MIDI is the standard way patterns get shared between DAWs and Toontrack products.
- **Addictive Drums 2**: ships a dedicated **GM Map Preset** specifically so users can drop in GM-standard MIDI or grooves from other software.
- **BFD3**: its Key Map panel explicitly lists **General MIDI** as one of the pre-loaded keymaps to try first before manual mapping.
- **Steven Slate Drums 5.5**: does not ship a GM preset out of the box, but the community (Groove Monkee) publishes free "GM IOMaps" for exactly this reason, because the built-in IOMaps (Toontrack, EZdrummer, AD2, various e-kits) don't include one natively.
- **GetGood Drums (Kontakt-based)**: no single canonical mapping. It varies per GGD library/title. Third-party sites (midiremap.com) maintain conversion tables between GM and specific GGD, EZdrummer, and Superior Drummer libraries.
- **Drumforge**: smallest/least standardized of the seven, also Kontakt-based, no public canonical map found.

**Implication for the architecture**: build the pipeline to always output a clean **General MIDI intermediate representation** first, then apply a thin, swappable, data-driven remap layer per plugin. For SD3/EZ3/AD2/BFD3 this remap layer can genuinely just be "load GM," which the plugin already understands natively. For SSD5.5/GGD/Drumforge, ship a best-effort JSON mapping seeded from public keymap docs and community tables, and be upfront in the UI that the user may need to load a matching keymap preset inside the plugin itself for perfect results.

### 1.4 Recommended libraries (Python)
| Purpose | Library | Notes |
|---|---|---|
| Audio I/O, resampling | `soundfile`, `librosa` | WAV read/write, normalization |
| Source separation (kit → voices) | `demucs` (Hybrid Demucs) + a drumsep-style fine-tuned checkpoint | PyTorch dependency, ~1GB+ install |
| Onset/transient detection | `madmom`, `librosa.onset`, `aubio` | madmom for accuracy, librosa as simpler fallback |
| MIDI file construction | `pretty_midi` or `mido` | `pretty_midi` is friendlier for note/velocity/timing objects |
| Numerics | `numpy`, `scipy` | peak/RMS analysis for velocity mapping |
| Visualization (in-app waveform + detected hits) | `pyqtgraph` | fast enough for interactive review before export |

### 1.5 GUI framework
Because the processing pipeline is Python/PyTorch-heavy, the pragmatic choice is to keep the GUI in the same process rather than bridging languages:
- **Recommended: PySide6 (Qt for Python)**. Mature, native Windows look and feel, well-documented PyInstaller packaging path for exactly this kind of "Python + heavy ML dependency" desktop app, and `pyqtgraph` integrates directly for the waveform/onset review view.
- **Alternative worth considering given your Flutter background: Flet.** Flet wraps the Flutter engine with a Python API, so the UI code would feel familiar from your other projects. The trade-off is that desktop packaging (bundling PyTorch + Flet's Flutter runtime into one Windows exe) is less battle-tested than PyInstaller + PySide6, and you'd be debugging two different packaging systems (Flutter engine + Python/PyTorch) instead of one. Reasonable as a v2 experiment once the core pipeline is proven, not for the first build.

---

## 2. Recommended Architecture

```
WAV drum stem
     │
     ▼
[1] Preprocess: resample/normalize, mono/stereo handling
     │
     ▼
[2] Source separation (Demucs/drumsep checkpoint)
     │   → kick.wav, snare.wav, toms.wav, hihat.wav, cymbals.wav
     ▼
[3] Per-stem onset detection (madmom/librosa, tuned per drum type)
     │
     ▼
[3a] Toms-only: per-onset pitch estimate + cluster into high/mid/low tom
     │
     ▼
[4] Velocity extraction (amplitude at transient → MIDI velocity curve)
     │
     ▼
[5] Timeline merge + overlap/bleed handling
     │   → unified list of (time, GM note, velocity) events
     ▼
[6] GM → plugin-specific remap (JSON profile per plugin)
     │
     ▼
[7] MIDI file writer (pretty_midi) → .mid
     │
     ▼
Output file, ready to drag into the selected plugin
```

Everything left of step 6 is plugin-agnostic. Step 6 is the only place plugin selection matters, which keeps the core transcription engine reusable and testable independent of which of the seven plugins the user picks.

---

## 3. Tech Stack Decision

- **Language**: Python 3.11 (best ecosystem support for the audio/ML libraries above)
- **GUI**: PySide6
- **Audio/ML**: demucs, librosa, madmom, soundfile, numpy, scipy
- **MIDI**: pretty_midi
- **Packaging**: PyInstaller (one-folder build first, one-file exe once stable, since PyTorch's DLLs are large and one-file mode slows startup)
- **macOS follow-up**: same codebase; PySide6 and the audio/ML stack are all cross-platform, so the main macOS-specific work later is packaging (`py2app` or PyInstaller's macOS target) and code-signing/notarization, not a rewrite.

---

## 4. Repository Structure

```
drum-stem-to-midi/
├── app/
│   ├── main.py                  # PySide6 entry point
│   ├── ui/
│   │   ├── main_window.py
│   │   └── waveform_view.py     # pyqtgraph onset review widget
│   └── controller.py             # wires UI to pipeline
├── pipeline/
│   ├── preprocess.py
│   ├── separation.py             # Demucs/drumsep wrapper
│   ├── onset_detection.py        # per-stem onset + velocity extraction
│   ├── merge.py                  # timeline merge + overlap/bleed handling
│   ├── remap.py                  # GM -> plugin note remap
│   └── midi_writer.py
├── mappings/
│   ├── general_midi.json
│   ├── superior_drummer_3.json
│   ├── ezdrummer_3.json
│   ├── addictive_drums_2.json
│   ├── bfd3.json
│   ├── steven_slate_5_5.json
│   ├── ggd.json
│   └── drumforge.json
├── models/                       # downloaded separation checkpoints (gitignored)
├── tests/
│   ├── fixtures/                 # short sample drum stems for regression tests
│   └── test_*.py
├── requirements.txt
├── build.spec                    # PyInstaller spec
└── README.md
```

---

## 5. MIDI Mapping Strategy

### 5.1 Canonical General MIDI drum map (the internal representation)
| Note | Sound | Note | Sound |
|---|---|---|---|
| 36 | Kick | 47 | Low-Mid Tom |
| 38 | Snare (acoustic) | 48 | Hi-Mid Tom |
| 40 | Snare (electric/rim variant) | 49 | Crash Cymbal 1 |
| 41 | Low Floor Tom | 50 | High Tom |
| 42 | Closed Hi-Hat | 51 | Ride Cymbal 1 |
| 44 | Pedal Hi-Hat | 52 | Chinese Cymbal |
| 45 | Low Tom | 53 | Ride Bell |
| 46 | Open Hi-Hat | 57 | Crash Cymbal 2 |
| 37 | Side Stick | 59 | Ride Cymbal 2 |

**Tom assignment**: the 3-way pitch cluster from Section 1.2.1 maps to `low → 45 (Low Tom)`, `mid → 47 (Low-Mid Tom)`, `high → 50 (High Tom)`. Cymbal stem defaults to `49 (Crash Cymbal 1)` as the single representative note for v1, per Section 1.2.1.

### 5.2 Per-plugin JSON profile format
```json
{
  "plugin": "Superior Drummer 3",
  "source": "General MIDI (built-in SD3 keymap preset)",
  "confidence": "high",
  "map": {
    "36": 36,
    "38": 38,
    "42": 42,
    "46": 46,
    "49": 49,
    "51": 51
  },
  "notes": "SD3 accepts GM MIDI directly via its built-in mapping presets. No remap needed; this file is effectively pass-through."
}
```
```json
{
  "plugin": "GetGood Drums",
  "source": "community reference (midiremap.com), best effort",
  "confidence": "low — varies per GGD library title",
  "map": {
    "36": 36,
    "38": 38,
    "42": 44,
    "46": 46,
    "49": 49,
    "51": 51
  },
  "notes": "GGD does not have one canonical map across all libraries. Ship this as a starting point and tell the user in the UI to verify against the specific GGD title's included keymap, or load the matching keymap preset inside Kontakt."
}
```
Mark each profile's `confidence` field (`high` / `medium` / `low`) and surface it in the UI dropdown, e.g. "GetGood Drums (mapping may need verification)". This sets honest expectations instead of implying every plugin is equally well supported on day one.

---

## 6. Step-by-Step Phases for Cursor Agent

### Phase 0 — Project scaffolding
- Initialize repo with the structure in Section 4.
- Set up `requirements.txt`: `pyside6`, `librosa`, `soundfile`, `numpy`, `scipy`, `pretty_midi`, `pyqtgraph`.
- Get a "hello window" PySide6 app running and packaged once with PyInstaller early, to surface Windows packaging issues before the codebase gets complex.
- **Acceptance criteria**: empty PySide6 window launches from source and from a PyInstaller build.

### Phase 1 — Core transcription MVP (CLI, single drum type, GM only)
- Build `pipeline/onset_detection.py` using `librosa.onset.onset_detect` on a single pre-isolated drum sample (e.g., a solo kick loop) as a CLI script.
- Build `pipeline/midi_writer.py` to turn a list of `(time, note, velocity)` into a `.mid` using `pretty_midi`.
- Velocity mapping: extract peak/RMS amplitude in a small window around each onset, normalize per-stem (calibrate min/max from the file itself), map to MIDI velocity 1-127.
- **Acceptance criteria**: feed in a solo kick WAV, get out a `.mid` with correctly timed note-36 hits and velocities that visibly track loudness when viewed in a DAW piano roll.

### Phase 2 — Source separation integration
- Wrap Demucs + a drumsep-style checkpoint in `pipeline/separation.py`. Download/cache the model checkpoint on first run (flag this clearly in the UI, it's a multi-hundred-MB download).
- Add the `cukas/drumsep`-style DSP-only path as a "fast/no-download" fallback mode.
- Test separation quality against a handful of real drum stems (rock kit, more cymbal-heavy jazz-ish kit) and log SNR/quality notes in `tests/fixtures/README.md` since quality will vary by genre.
- **Acceptance criteria**: given one full-kit WAV, produces separate kick/snare/toms/hihat/cymbals WAVs a human can identify by ear.

### Phase 3 — Full pipeline: multi-stem onset detection, merge, overlap/bleed handling
- Run Phase 1's onset detector against each separated stem from Phase 2, with per-stem tuned parameters (cymbals/hihat need different onset backend settings than kick/snare, per Section 1.1).
- For the toms stem specifically: after onsets are found, run pitch estimation on each hit and adaptively choose 2 or 3 clusters via silhouette comparison (Section 1.2.1), then map the clusters (ascending by center pitch) to low/mid/high tom notes before it goes into the merge step.
- Build `pipeline/merge.py`:
  - Merge all stems' onsets into one sorted timeline.
  - Apply a minimum inter-onset interval per voice (~30-40ms) to suppress double-triggers from decay/ringing.
  - Add a user-configurable minimum-velocity threshold to optionally filter bleed, but default to keeping ghost notes.
  - Add a same-timestamp bleed suppression heuristic: if a much louder onset in one stem coincides with a much quieter onset in another stem within a few ms, treat the quiet one as bleed and drop it (configurable, off by default until validated).
- **Acceptance criteria**: run against a real multi-drum stem end-to-end and get a single GM-mapped `.mid` file that a human can play back over the original audio and roughly recognize the groove.

### Phase 4 — Plugin-specific mapping layer
- Implement `pipeline/remap.py`: load the selected plugin's JSON profile from `mappings/`, remap each GM note event to the plugin's note number(s).
- Populate the seven JSON profiles per Section 5.2, marking confidence levels honestly (SD3/EZ3/AD2/BFD3 = high confidence via documented GM presets; SSD5.5 = medium via community GM IOMap; GGD/Drumforge = low, best-effort).
- **Acceptance criteria**: same input stem produces different `.mid` note numbers depending on selected plugin, verified against each plugin's documented/community keymap.

### Phase 5 — GUI
- Build `app/main_window.py`: file picker for WAV upload, dropdown for plugin selection (using the `confidence` field to annotate low-confidence options), a "Convert" button, and a progress indicator (separation + detection can take real time on CPU).
- Add the `waveform_view.py` review screen: show the waveform with detected onsets overlaid before final export, with a sensitivity slider that re-runs onset detection live. This directly addresses the fact that transient detection will need per-song tuning (Section 7).
- Wire everything through `controller.py`.
- **Acceptance criteria**: full user flow from Section "User Flow" in the original spec works end-to-end through the GUI.

### Phase 6 — Windows packaging and QA pass
- Finalize `build.spec` for PyInstaller, test the packaged exe on a clean Windows VM (no Python/dev tools installed) to catch missing-DLL issues, especially around PyTorch/Demucs.
- Run the full pipeline against at least one real stem per genre archetype (straightforward rock, busy fills, cymbal-heavy) and log known failure modes.
- Write the README with setup instructions and known limitations (Section 7).
- **Acceptance criteria**: a fresh Windows machine can run the installed app, upload a WAV, pick any of the 7 plugins, and get a usable `.mid` out, without needing Python installed separately.

---

## 7. Known Risks and Limitations (be upfront about these in the UI/README)
- **Toms and cymbals separate worse than kick/snare.** Expect more manual cleanup needed on busy fills and cymbal-heavy passages regardless of which separation model is used, this is a known limit of current source separation research, not a bug to chase indefinitely.
- **GGD and Drumforge mappings are genuinely not standardized** across their own product lines. Treat the shipped JSON profiles as a documented starting point, not a guarantee, and make that visible in the UI.
- **CPU-only inference will be slow** for Demucs-based separation on longer stems. Set expectations in the UI (a progress bar with an honest time estimate) rather than letting the app appear frozen.
- **Tom clustering is relative, not absolute.** Because low/mid/high tom is decided by k-means/percentile over pitches found in that one file, a kit with only 2 toms will still get split into 3 buckets (one will just be sparsely populated or a near-duplicate of a neighbor), and "low tom" in one song isn't guaranteed to match the same physical drum as "low tom" in another. Fine for v1; a per-kit calibration step is a reasonable post-MVP improvement.
- **Tempo/quantization is out of scope for MVP.** The plan above outputs note events at their literal detected times; tempo detection and quantization-to-grid is a reasonable Phase 7 addition later, not a blocker for a usable v1.

## 8. Post-MVP Roadmap
- Open hi-hat vs closed hi-hat state detection (spectral centroid heuristic).
- Per-kit-title mapping profiles for GGD (since it varies by library, not just by plugin).
- Tempo detection and note quantization options.
- macOS build via PyInstaller/py2app once the Windows build is stable.
- Batch processing (drop a folder of stems, get a folder of `.mid` files back).

## 9. Sources
- Toontrack Superior Drummer 3 forums, MIDI mapping and "Add Suggested Note" behavior: toontrack.com/forums
- XLN Audio Addictive Drums 2 support docs, GM Map Preset: support.xlnaudio.com
- BFD3 keymap support docs, General MIDI as a pre-loaded keymap: support.bfddrums.com
- Steven Slate Drums 5.5 GM IOMap (community, Groove Monkee): groovemonkee.com
- midiremap.com, community drum map conversion tables
- inagoy/drumsep and cukas/drumsep (GitHub), MVSep drum separation model comparisons: mvsep.com
- madmom (arXiv:1605.07008), librosa, ADTLib documentation
