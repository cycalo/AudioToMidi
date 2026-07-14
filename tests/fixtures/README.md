# Test fixtures

Drop **real full-kit drum stems** (`.wav`) here to exercise source separation.
Audio files in this folder are gitignored (they can be large or copyrighted), so
they stay local to your machine. When at least one `.wav` is present, the
real-fixture tests in [../test_separation.py](../test_separation.py) run
automatically; otherwise they skip.

## How to test separation

```bash
# DSP fallback (fast, no download, 5 stems)
python pipeline/separation.py tests/fixtures/your_kit.wav -o tests/fixtures/your_kit_stems --backend dsp

# Demucs drumsep (best quality, 4 stems, one-time ~167 MB download)
python pipeline/separation.py tests/fixtures/your_kit.wav -o tests/fixtures/your_kit_stems --backend demucs
```

Then listen to each stem and confirm it is identifiable by ear (the true
Phase 2 acceptance bar).

## Backend stem sets

- `demucs`: kick, snare, toms, hihat (DSP extract), cymbals.
- `dsp`: kick, snare, toms, hihat, cymbals.

## Separation quality notes (log per stem / genre)

Fill this in as real stems are tested, since quality varies by genre and by drum
voice (kick separates cleanest; toms/cymbals are the weakest link).

| Stem file | Genre / kit | Backend | Notes (SNR, bleed, artifacts) |
|---|---|---|---|
| _example.wav_ | _rock_ | _demucs_ | _pending_ |

## Phase 3 transcription notes (stems -> GM MIDI)

Transcription detects **kick / snare / toms only** (hi-hat and cymbals are
ignored). Bleed is reduced with a relative peak floor and cross-stem dominance
gate.

```bash
python pipeline/merge.py tests/fixtures/drums_1_stems -o tests/fixtures/drums_1.mid
```

`Drum Sample.wav` (acoustic kit, ~164s) after snare peak-window fix:

| Events | kick(36) | snare(38) | toms(45/50) | hats/cymbals |
|---|---|---|---|---|
| ~518 | ~322 | ~112 | ~13/71 | none |

Snare look-ahead for the relative-peak gate is 50 ms (kick 25 ms) so backtracked
onsets still measure the real attack. Sensitivity also eases that gate toward
"More".
